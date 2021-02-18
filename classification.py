
# coding: utf-8

# In[77]:
# Import all used python libraries.

import random
from keras import models
from keras.layers import Dropout, Dense
import tensorflow as tf
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_selection import SelectKBest, f_classif
import multiprocessing as mp
from math import ceil

# In[78]:
# Load all used functions.


def preprocess_text(text):
    """Remove non-characters and lower case the text"""
    # replace non characers with space and lower case
    temp = re.sub(r"[/W/D/S.,-]+", " ", str(text).lower())
    # merge multiple spaces to a single one
    return re.sub(r"[ ]+", " ", temp)


def buckets(data, n):
    """Return a factory that yields buckets with size n."""
    # Shuffle all datasets to get a more consistent workload for all threads.
    random.shuffle(data)

    for i in range(0, len(data), n):
        yield data[i:i + n]


def sample_datasets(datasets, language, targets, sampling, tfidf_parameters, sample_reruns):
    """Sample the datasets."""
    # Load the csv
    data = pd.read_csv(TRAIN_TEST_PATH+language+'.csv')

    # preprocess the text
    data['clean'] = data['all_text_orig_lang_lemma'].apply(
        lambda x: preprocess_text(x))

    for target in targets:
        print(target)
        for s in sampling:
            positive = data[data[target] == 1]

            if s <= len(positive):
                for _ in range(sample_reruns):
                    if s == 'max_pos':
                        # Sample as many text as there are positive samples
                        positive = data[data[target] == 1]
                        negative = data[data[target] == 0]

                        sample = positive
                        sample = sample.append(negative.sample(len(positive)))

                        sampled_data = sample
                    elif s > 0:
                        # Sample a specific number of texts
                        positive = data[data[target] == 1]
                        negative = data[data[target] == 0]

                        if len(positive) > s:
                            sample = positive.sample(s)
                            sample = sample.append(negative.sample(s))
                        else:
                            continue

                        sampled_data = sample
                    else:
                        # Use all the available data
                        # Triggered if sampling is set to 0
                        sampled_data = data

                    positive_count = len(
                        sampled_data[sampled_data[target] == 1])
                    negative_count = len(
                        sampled_data[sampled_data[target] == 0])

                    datasets.append(prepare_datasets(
                        sampled_data,
                        target,
                        tfidf_parameters,
                        positive_count,
                        negative_count
                    ))


def prepare_datasets(data, target, tfidf_parameters, positive_count, negative_count):
    """Create the tfidf vectors for a specific dataset and return metadata, vectors, and labels."""
    vectorizer = TfidfVectorizer(**tfidf_parameters)

    # Learn vocabulary from training texts and vectorize training texts.
    x_train = vectorizer.fit_transform(data['clean'])
    train_labels = data[target]

    # Select top words of the vectorized features.
    selector = SelectKBest(f_classif, k=min(20000, x_train.shape[1]))
    selector.fit(x_train, train_labels)
    x_train = selector.transform(x_train).astype('float32')

    output = [
        ' '.join([
            language,
            target,
            str(positive_count),
            str(negative_count)
        ]),
        x_train,
        train_labels,
    ]

    return output


def train_classifiers(datasets):
    """Train the classifiers on all datasets."""
    # Create result dataframe
    out = pd.DataFrame(
        columns=["Dataset", "Classifier", "Accuracy", "F1", "Precision", "Recall"])

    # Iterate the datasets
    for data_id, dataset in enumerate(datasets):
        dataset_name = dataset[0]
        data = dataset[1]
        y = np.array(dataset[2])
        skf = StratifiedKFold(n_splits=4)
        split_indices = []
        print(dataset_name)

        for train_indices, test_indices in skf.split(data, y):
            split_indices.append((train_indices, test_indices))

            print("datasets: ", str(data_id+1), "/", str(len(datasets)))

            clf_params = [
                'mlp',
                'svm',
                'rf'
            ]

            # Iterate classifiers
            for model_type in clf_params:
                print("Classifier: ", str(model_type))

                acc_scores = []
                pre_scores = []
                rec_scores = []
                f1_scores = []

                # Iterate splits
                for train_index, test_index in split_indices:
                    global X_train
                    X_train, X_test = data[train_index], data[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    y_pred = None

                    if model_type == 'mlp':
                        # Create model instance.
                        model = mlp_model(layers=2, units=64, dropout_rate=0.2,
                                          input_shape=X_train.shape[1:], num_classes=2)
                        optimizer = tf.keras.optimizers.Adam(lr=1e-3)
                        model.compile(optimizer=optimizer,
                                      loss='binary_crossentropy', metrics=['acc'])

                        # Stop training is validation loss doesnt decrease for 3 steps
                        callbacks = [tf.keras.callbacks.EarlyStopping(
                            monitor='val_loss', patience=3)]

                        # Train and validate model.
                        history = model.fit(
                            X_train,
                            y_train,
                            epochs=100,
                            callbacks=callbacks,
                            validation_data=(X_test, y_test),
                            verbose=0,
                            batch_size=512)

                        acc_scores.append(history.history['val_acc'][-1])
                        y_pred = [round(a[0]) for a in model.predict(X_test)]
                    elif model_type == 'svm':
                        # Linear SVM
                        model = SVC(gamma=0.001, kernel='linear')

                        model.fit(X_train, y_train)

                        acc_scores.append(model.score(X_test, y_test))
                        y_pred = model.predict(X_test)
                    elif model_type == 'rf':
                        # Random Forest Classifier
                        model = RandomForestClassifier(
                            criterion='entropy', n_estimators=200)

                        model.fit(X_train, y_train)

                        acc_scores.append(model.score(X_test, y_test))
                        y_pred = model.predict(X_test)

                    # Compute the results
                    prfs = precision_recall_fscore_support(
                        y_test, y_pred, warn_for=[])

                    pre_scores.append(prfs[0].mean())
                    rec_scores.append(prfs[1].mean())
                    f1_scores.append(prfs[2].mean())

                # Append average scores
                clf_acc = np.array(acc_scores).mean()
                clf_pre = np.array(pre_scores).mean()
                clf_rec = np.array(rec_scores).mean()
                clf_f1 = np.array(f1_scores).mean()

                out = out.append(pd.DataFrame(
                    [[dataset_name, model_type, clf_acc, clf_f1, clf_pre, clf_rec]], columns=out.columns), ignore_index=True)

    return out


def _get_last_layer_units_and_activation(num_classes):
    """Gets the # units and activation function for the last network layer.

    # Arguments
        num_classes: int, number of classes.

    # Returns
        units, activation values.
    """
    # https://developers.google.com/machine-learning/guides/text-classification/step-4

    if num_classes == 2:
        activation = 'sigmoid'
        units = 1
    else:
        activation = 'softmax'
        units = num_classes
    return units, activation


def mlp_model(layers, units, dropout_rate, input_shape, num_classes):
    """Creates an instance of a multi-layer perceptron model.

    # Arguments
        layers: int, number of `Dense` layers in the model.
        units: int, output dimension of the layers.
        dropout_rate: float, percentage of input to drop at Dropout layers.
        input_shape: tuple, shape of input to the model.
        num_classes: int, number of output classes.

    # Returns
        An MLP model instance.
    """
    # https://developers.google.com/machine-learning/guides/text-classification/step-4

    op_units, op_activation = _get_last_layer_units_and_activation(num_classes)
    model = models.Sequential()
    model.add(Dropout(rate=dropout_rate, input_shape=input_shape))

    for _ in range(layers-1):
        model.add(Dense(units=units, activation='relu'))
        model.add(Dropout(rate=dropout_rate))

    model.add(Dense(units=op_units, activation=op_activation))
    return model


# In[79]:
# Prepare all datasets.
# Be careful this step can take a considerable amount of time.
datasets = []

SUB_SAMPLE_RERUNS = 5
tfidf_parameters = {
    'ngram_range': (1, 2),
    'dtype': 'int32',
    'strip_accents': 'unicode',
    'decode_error': 'replace',
    'analyzer': 'word',
    'min_df': 2,
}

TRAIN_TEST_PATH = 'data/articles_dictionary_annotated_'
languages = ['de', 'es', 'pl', 'ro', 'sv', 'uk']
targets = ['d_fr_eco', 'd_fr_lab', 'd_fr_sec', 'd_fr_wel']
sampling = [100, 150, 200, 250, 300, 350,
            400, 450, 500, 600, 700, 800, 900, 1000, 'max_pos', 0]

for language in languages:
    print(language)
    sample_datasets(datasets, language, targets,
                    sampling, tfidf_parameters, SUB_SAMPLE_RERUNS)

# In[80]:
# Run all the different dataset and model combinations.
# Fast version using data parallelism.
# Use this cell OR the cell below.

pool = mp.Pool(processes=(mp.cpu_count()))
results = pool.map(train_classifiers, buckets(
    datasets, ceil(len(datasets)/(mp.cpu_count()))))
pool.close()
pool.join()

output = pd.concat(results)
output.to_csv(('results_classifications.csv'), index=False)

print('DONE')
# In[81]:
# Single threaded version.
# Be careful: this might take very long!
output = train_classifiers(datasets)
output.to_csv(('results_all_.csv'), index=False)

print('DONE')
