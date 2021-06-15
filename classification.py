
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


def sample_datasets(language, target, sampling, tfidf_parameters, top_k, sample_reruns):
    """Sample the datasets."""
    # Load the csv
    data = pd.read_csv(TRAIN_TEST_PATH+language+'.csv')
    datasets = []

    # preprocess the text
    data['clean'] = data['all_text_orig_lang_lemma'].apply(
        lambda x: preprocess_text(x))

    for s in sampling:
        positive = data[data[target] == 1]

        if s <= len(positive):
            for _ in range(sample_reruns):
                if s == 'max_pos':
                    # Sample as many text as there are positive samples
                    positive = data[data[target] == 1]
                    negative = data[data[target] == 0]

                    sample = positive
                    sample = sample.append(
                        negative.sample(len(positive)))

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
                    language,
                    target,
                    tfidf_parameters,
                    top_k,
                    positive_count,
                    negative_count
                ))
    return datasets


def prepare_datasets(data, language, target, tfidf_parameters, top_k_words, positive_count, negative_count):
    """Create the tfidf vectors for a specific dataset and return metadata, vectors, and labels."""
    vectorizer = TfidfVectorizer(**tfidf_parameters)

    # Learn vocabulary from training texts and vectorize training texts.
    x_train = vectorizer.fit_transform(data['clean'])
    train_labels = data[target]

    # Select top words of the vectorized features.
    selector = SelectKBest(f_classif, k=min(top_k_words, x_train.shape[1]))
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


def train_classifiers(params):
    """Train the classifiers on all datasets."""
    # Create result dataframe
    out = pd.DataFrame(
        columns=["Dataset", "Classifier", "ModelParams", "Accuracy", "F1", "Precision", "Recall"])

    for model_type, all_languages in params.items():
        print("Classifier: ", str(model_type))

        for language, all_targets in all_languages.items():
            for target, model_params in all_targets.items():

                datasets = sample_datasets(
                    language, target, SAMPLING, TFIDF, model_params['top_k_words'], SUB_SAMPLE_RERUNS)

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

                        print("datasets: ", str(data_id+1),
                              "/", str(len(datasets)))

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

                            if model_type == 'MLP':
                                # Create model instance.
                                model = mlp_model(layers=model_params['hidden_layers'], units=model_params['hidden_units'], dropout_rate=model_params['dropout_rate'],
                                                  input_shape=X_train.shape[1:], num_classes=2)
                                optimizer = tf.keras.optimizers.Adam(
                                    lr=model_params['learning_rate'])
                                model.compile(optimizer=optimizer,
                                              loss='binary_crossentropy', metrics=['acc'])

                                # Stop training is validation loss doesnt decrease for 3 steps
                                callbacks = [tf.keras.callbacks.EarlyStopping(
                                    monitor='val_loss', patience=3)]

                                # Train and validate model.
                                history = model.fit(
                                    X_train,
                                    y_train,
                                    epochs=model_params['epochs'],
                                    callbacks=callbacks,
                                    validation_data=(X_test, y_test),
                                    verbose=0,
                                    batch_size=512)

                                acc_scores.append(
                                    history.history['val_acc'][-1])
                                y_pred = [round(a[0])
                                          for a in model.predict(X_test)]
                            elif model_type == 'SVM':
                                # Linear SVM
                                model = SVC(
                                    C=model_params['C'], kernel=model_params['kernel'])

                                model.fit(X_train, y_train)

                                acc_scores.append(model.score(X_test, y_test))
                                y_pred = model.predict(X_test)
                            elif model_type == 'Random Forest':
                                # Random Forest Classifier
                                model = RandomForestClassifier(
                                    criterion=model_params['criterion'], n_estimators=model_params['n_estimators'])

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
                            [[dataset_name, model_type, str(model_params), clf_acc, clf_f1, clf_pre, clf_rec]], columns=out.columns), ignore_index=True)

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
SUB_SAMPLE_RERUNS = 5
TFIDF = {
    'ngram_range': (1, 2),
    'dtype': 'int32',
    'strip_accents': 'unicode',
    'decode_error': 'replace',
    'analyzer': 'word',
    'min_df': 2,
}

TRAIN_TEST_PATH = 'data/articles_dictionary_annotated_'
SAMPLING = [100, 150, 200, 250, 300, 350,
            400, 450, 500, 600, 700, 800, 900, 1000]

# for language in languages:
#     print(language)
#     sample_datasets(datasets, language, targets,
#                     sampling, tfidf_parameters, TOP_K_WORDS, SUB_SAMPLE_RERUNS)


# In[79]
best_classifiers_rm = {'de': {'d_fr_eco': {"n_estimators": 100, "criterion": "entropy", "top_k_words": 10000},
                              'd_fr_lab': {"n_estimators": 300, "criterion": "gini", "top_k_words": 10000},
                              'd_fr_sec': {"n_estimators": 50, "criterion": "gini", "top_k_words": 20000},
                              'd_fr_wel': {"n_estimators": 300, "criterion": "entropy", "top_k_words": 30000}},
                       'es': {'d_fr_eco': {"n_estimators": 300, "criterion": "entropy", "top_k_words": 30000},
                              'd_fr_lab': {"n_estimators": 300, "criterion": "entropy", "top_k_words": 10000},
                              'd_fr_sec': {"n_estimators": 100, "criterion": "entropy", "top_k_words": 20000},
                              'd_fr_wel': {"n_estimators": 200, "criterion": "entropy", "top_k_words": 30000}},
                       'pl': {'d_fr_eco': {"n_estimators": 200, "criterion": "gini", "top_k_words": 20000},
                              'd_fr_lab': {"n_estimators": 100, "criterion": "gini", "top_k_words": 20000},
                              'd_fr_sec': {"n_estimators": 300, "criterion": "gini", "top_k_words": 30000},
                              'd_fr_wel': {"n_estimators": 200, "criterion": "entropy", "top_k_words": 30000}},
                       'ro': {'d_fr_eco': {"n_estimators": 100, "criterion": "gini", "top_k_words": 20000},
                              'd_fr_lab': {"n_estimators": 200, "criterion": "entropy", "top_k_words": 10000},
                              'd_fr_sec': {"n_estimators": 300, "criterion": "gini", "top_k_words": 20000},
                              'd_fr_wel': {"n_estimators": 200, "criterion": "gini", "top_k_words": 10000}},
                       'sv': {'d_fr_eco': {"n_estimators": 300, "criterion": "entropy", "top_k_words": 30000},
                              'd_fr_lab': {"n_estimators": 300, "criterion": "gini", "top_k_words": 10000},
                              'd_fr_sec': {"n_estimators": 300, "criterion": "entropy", "top_k_words": 20000},
                              'd_fr_wel': {"n_estimators": 200, "criterion": "entropy", "top_k_words": 10000}},
                       'en': {'d_fr_eco': {"n_estimators": 200, "criterion": "entropy", "top_k_words": 30000},
                              'd_fr_lab': {"n_estimators": 300, "criterion": "entropy", "top_k_words": 20000},
                              'd_fr_sec': {"n_estimators": 300, "criterion": "entropy", "top_k_words": 10000},
                              'd_fr_wel': {"n_estimators": 300, "criterion": "gini", "top_k_words": 10000}}}

best_classifiers_svm = {'de': {'d_fr_eco': {"C": 5, "kernel": "rbf", "top_k_words": 20000},
                               'd_fr_lab': {"C": 5, "kernel": "rbf", "top_k_words": 30000},
                               'd_fr_sec': {"C": 1, "kernel": "rbf", "top_k_words": 20000},
                               'd_fr_wel': {"C": 5, "kernel": "rbf", "top_k_words": 30000}},
                        'es': {'d_fr_eco': {"C": 5, "kernel": "rbf", "top_k_words": 30000},
                               'd_fr_lab': {"C": 3, "kernel": "rbf", "top_k_words": 30000},
                               'd_fr_sec': {"C": 3, "kernel": "rbf", "top_k_words": 20000},
                               'd_fr_wel': {"C": 3, "kernel": "rbf", "top_k_words": 30000}},
                        'pl': {'d_fr_eco': {"C": 5, "kernel": "linear", "top_k_words": 30000},
                               'd_fr_lab': {"C": 3, "kernel": "sigimoid", "top_k_words": 10000},
                               'd_fr_sec': {"C": 5, "kernel": "rbf", "top_k_words": 10000},
                               'd_fr_wel': {"C": 3, "kernel": "sigimoid", "top_k_words": 20000}},
                        'ro': {'d_fr_eco': {"C": 3, "kernel": "sigimoid", "top_k_words": 20000},
                               'd_fr_lab': {"C": 5, "kernel": "sigimoid", "top_k_words": 20000},
                               'd_fr_sec': {"C": 5, "kernel": "sigimoid", "top_k_words": 30000},
                               'd_fr_wel': {"C": 5, "kernel": "sigimoid", "top_k_words": 20000}},
                        'sv': {'d_fr_eco': {"C": 5, "kernel": "linear", "top_k_words": 30000},
                               'd_fr_lab': {"C": 5, "kernel": "rbf", "top_k_words": 10000},
                               'd_fr_sec': {"C": 3, "kernel": "linear", "top_k_words": 30000},
                               'd_fr_wel': {"C": 5, "kernel": "linear", "top_k_words": 30000}},
                        'en': {'d_fr_eco': {"C": 5, "kernel": "linear", "top_k_words": 30000},
                               'd_fr_lab': {"C": 5, "kernel": "rbf", "top_k_words": 30000},
                               'd_fr_sec': {"C": 3, "kernel": "rbf", "top_k_words": 10000},
                               'd_fr_wel': {"C": 5, "kernel": "linear", "top_k_words": 30000}}}

best_classifiers_mlp = {'de': {'d_fr_eco': {"hidden_layers": 2, "hidden_units": 16, "dropout_rate": 0.2, "learning_rate": 1e-2, "epochs": 100, "top_k_words": 20000},
                               'd_fr_lab': {"hidden_layers": 2, "hidden_units": 16, "dropout_rate": 0.2, "learning_rate": 1e-2, "epochs": 100, "top_k_words": 30000},
                               'd_fr_sec': {"hidden_layers": 2, "hidden_units": 32, "dropout_rate": 0.2, "learning_rate": 1e-2, "epochs": 100, "top_k_words": 20000},
                               'd_fr_wel': {"hidden_layers": 2, "hidden_units": 16, "dropout_rate": 0.2, "learning_rate": 1e-2, "epochs": 100, "top_k_words": 30000}},
                        'es': {'d_fr_eco': {"hidden_layers": 2, "hidden_units": 32, "dropout_rate": 0.2, "learning_rate": 1e-3, "epochs": 100, "top_k_words": 30000},
                               'd_fr_lab': {"hidden_layers": 2, "hidden_units": 32, "dropout_rate": 0.2, "learning_rate": 1e-3, "epochs": 100, "top_k_words": 30000},
                               'd_fr_sec': {"hidden_layers": 2, "hidden_units": 16, "dropout_rate": 0.2, "learning_rate": 1e-3, "epochs": 100, "top_k_words": 30000},
                               'd_fr_wel': {"hidden_layers": 2, "hidden_units": 32, "dropout_rate": 0.2, "learning_rate": 1e-3, "epochs": 100, "top_k_words": 30000}},
                        'pl': {'d_fr_eco': {"hidden_layers": 2, "hidden_units": 32, "dropout_rate": 0.2, "learning_rate": 1e-2, "epochs": 100, "top_k_words": 30000},
                               'd_fr_lab': {"hidden_layers": 3, "hidden_units": 64, "dropout_rate": 0.2, "learning_rate": 1e-2, "epochs": 100, "top_k_words": 20000},
                               'd_fr_sec': {"hidden_layers": 3, "hidden_units": 32, "dropout_rate": 0.2, "learning_rate": 1e-2, "epochs": 100, "top_k_words": 30000},
                               'd_fr_wel': {"hidden_layers": 2, "hidden_units": 32, "dropout_rate": 0.2, "learning_rate": 1e-2, "epochs": 100, "top_k_words": 20000}},
                        'ro': {'d_fr_eco': {"hidden_layers": 3, "hidden_units": 32, "dropout_rate": 0.2, "learning_rate": 1e-3, "epochs": 100, "top_k_words": 30000},
                               'd_fr_lab': {"hidden_layers": 3, "hidden_units": 64, "dropout_rate": 0.2, "learning_rate": 1e-3, "epochs": 100, "top_k_words": 20000},
                               'd_fr_sec': {"hidden_layers": 2, "hidden_units": 16, "dropout_rate": 0.2, "learning_rate": 1e-2, "epochs": 100, "top_k_words": 30000},
                               'd_fr_wel': {"hidden_layers": 2, "hidden_units": 64, "dropout_rate": 0.2, "learning_rate": 1e-3, "epochs": 100, "top_k_words": 20000}},
                        'sv': {'d_fr_eco': {"hidden_layers": 3, "hidden_units": 32, "dropout_rate": 0.2, "learning_rate": 1e-3, "epochs": 100, "top_k_words": 20000},
                               'd_fr_lab': {"hidden_layers": 2, "hidden_units": 128, "dropout_rate": 0.2, "learning_rate": 1e-2, "epochs": 100, "top_k_words": 10000},
                               'd_fr_sec': {"hidden_layers": 2, "hidden_units": 32, "dropout_rate": 0.2, "learning_rate": 1e-3, "epochs": 100, "top_k_words": 30000},
                               'd_fr_wel': {"hidden_layers": 2, "hidden_units": 32, "dropout_rate": 0.2, "learning_rate": 1e-3, "epochs": 100, "top_k_words": 30000}},
                        'en': {'d_fr_eco': {"hidden_layers": 2, "hidden_units": 16, "dropout_rate": 0.2, "learning_rate": 1e-2, "epochs": 100, "top_k_words": 30000},
                               'd_fr_lab': {"hidden_layers": 2, "hidden_units": 16, "dropout_rate": 0.2, "learning_rate": 1e-2, "epochs": 100, "top_k_words": 30000},
                               'd_fr_sec': {"hidden_layers": 3, "hidden_units": 32, "dropout_rate": 0.2, "learning_rate": 1e-2, "epochs": 100, "top_k_words": 30000},
                               'd_fr_wel': {"hidden_layers": 3, "hidden_units": 32, "dropout_rate": 0.2, "learning_rate": 1e-2, "epochs": 100, "top_k_words": 30000}}}

best_classifier_params = {"Random Forest": best_classifiers_rm,
                          "SVM": best_classifiers_svm, "MLP": best_classifiers_mlp}

# In[80]:
# Run all the different dataset and model combinations.
# Fast version using data parallelism.
# Use this cell OR the cell below.

# pool = mp.Pool(processes=(mp.cpu_count()))
# results = pool.map(train_classifiers, buckets(
#     datasets, ceil(len(datasets)/(mp.cpu_count()))))
# pool.close()
# pool.join()

# output = pd.concat(results)
# output.to_csv(('results_classifications.csv'), index=False)

# print('DONE')
# In[81]:
# Single threaded version.
# Be careful: this might take very long!
output = train_classifiers(best_classifier_params)
output.to_csv('results_best_TEST.csv', index=False)

print('DONE')

# %%
