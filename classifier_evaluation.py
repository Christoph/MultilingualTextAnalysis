# coding: utf-8
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
    datasets = []
    # Load the csv
    data = pd.read_csv(TRAIN_TEST_PATH+language+'.csv')
    validation_data = pd.read_csv(VALIDATION_SET_PATH+language+'.csv') 

    # preprocess the text
    data['clean'] = data['all_text_orig_lang_lemma'].apply(
        lambda x: preprocess_text(x))

    validation_data['clean'] = validation_data['all_text_orig_lang_lemma'].apply(
        lambda x: preprocess_text(x))
    
    # Rename columns
    validation_data.rename(columns={'m_fr_eco': 'd_fr_eco'}, inplace=True)
    validation_data.rename(columns={'m_fr_lab': 'd_fr_lab'}, inplace=True)
    validation_data.rename(columns={'m_fr_sec': 'd_fr_sec'}, inplace=True)
    validation_data.rename(columns={'m_fr_wel': 'd_fr_wel'}, inplace=True)

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

                positive_count = len(sampled_data[sampled_data[target] == 1])
                negative_count = len(sampled_data[sampled_data[target] == 0])

                datasets.append(prepare_datasets(
                    sampled_data,
                    validation_data,
                    language,
                    target,
                    tfidf_parameters,
                    top_k,
                    positive_count,
                    negative_count
                ))
    return datasets


def prepare_datasets(data, validation_data, language, target, tfidf_parameters, top_k_words, positive_count, negative_count):
    """Create the tfidf vectors for a specific dataset and return metadata, vectors, and labels."""
    vectorizer = TfidfVectorizer(**tfidf_parameters)

    # Learn vocabulary from training texts and vectorize training texts.
    x_train = vectorizer.fit_transform(data['clean'])
    train_labels = data[target]
    
    # Vectorize validation texts.
    x_val = vectorizer.transform(validation_data['clean'])
    val_labels = validation_data[target]

    # Select top words of the vectorized features.
    selector = SelectKBest(f_classif, k=min(top_k_words, x_train.shape[1]))
    selector.fit(x_train, train_labels)
    x_train = selector.transform(x_train).astype('float32')
    x_val = selector.transform(x_val).astype('float32')

    output = [
        ' '.join([
            language,
            target,
            str(positive_count),
            str(negative_count)
        ]),
        x_train,
        train_labels,
        x_val,
        val_labels
    ]

    return output

def train_classifiers(params):
    """Train the classifiers on all datasets, return performance data."""
    # Create result dataframe
    out = pd.DataFrame(
        columns=["Dataset", "Classifier", "Accuracy", "F1", "Precision", "Recall"])

    for model_type, all_languages in params.items():
        print("Classifier: ", str(model_type))

        for language, all_targets in all_languages.items():
            print(language)
            for target, model_params in all_targets.items():
                print(target)
                print(model_params)

                datasets = sample_datasets(
                    language, target, SAMPLING, TFIDF, model_params['top_k_words'], SUB_SAMPLE_RERUNS)

                # Iterate the datasets
                for data_id, dataset in enumerate(datasets):
                    dataset_name = dataset[0]
                    data = dataset[1]
                    y = np.array(dataset[2])
                    val_data = dataset[3]
                    val_y = np.array(dataset[4])

                    acc_scores = []
                    pre_scores = []
                    rec_scores = []
                    f1_scores = []
                
                    global X_train
                    X_train, X_test = data, val_data
                    y_train, y_test = y, val_y
                    y_pred = None

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
    """Gets the # units and activation function for the last network layer."""
    # https://developers.google.com/machine-learning/guides/text-classification/step-4

    if num_classes == 2:
        activation = 'sigmoid'
        units = 1
    else:
        activation = 'softmax'
        units = num_classes
    return units, activation


def mlp_model(layers, units, dropout_rate, input_shape, num_classes):
    """Creates an instance of a multi-layer perceptron model."""
    # https://developers.google.com/machine-learning/guides/text-classification/step-4

    op_units, op_activation = _get_last_layer_units_and_activation(num_classes)
    model = models.Sequential()
    model.add(Dropout(rate=dropout_rate, input_shape=input_shape))

    for _ in range(layers-1):
        model.add(Dense(units=units, activation='relu'))
        model.add(Dropout(rate=dropout_rate))

    model.add(Dense(units=op_units, activation=op_activation))
    return model

# best classifier parameters
best_classifiers_mlp = {'de':{'d_fr_eco':{"hidden_layers": 2, "hidden_units": 16, "dropout_rate": 0.2, "learning_rate": 1e-2, "epochs": 100, "top_k_words":20000}, 
                             'd_fr_lab':{"hidden_layers": 2, "hidden_units": 16, "dropout_rate": 0.2, "learning_rate": 1e-2, "epochs": 100, "top_k_words":30000}, 
                             'd_fr_sec':{"hidden_layers": 2, "hidden_units": 32, "dropout_rate": 0.2, "learning_rate": 1e-2, "epochs": 100, "top_k_words":20000}, 
                             'd_fr_wel':{"hidden_layers": 2, "hidden_units": 16, "dropout_rate": 0.2, "learning_rate": 1e-2, "epochs": 100, "top_k_words":30000}}, 
                       'es':{'d_fr_eco':{"hidden_layers": 2, "hidden_units": 32, "dropout_rate": 0.2, "learning_rate": 1e-3, "epochs": 100, "top_k_words":30000}, 
                             'd_fr_lab':{"hidden_layers": 2, "hidden_units": 32, "dropout_rate": 0.2, "learning_rate": 1e-3, "epochs": 100, "top_k_words":30000}, 
                             'd_fr_sec':{"hidden_layers": 2, "hidden_units": 16, "dropout_rate": 0.2, "learning_rate": 1e-3, "epochs": 100, "top_k_words":30000}, 
                             'd_fr_wel':{"hidden_layers": 2, "hidden_units": 32, "dropout_rate": 0.2, "learning_rate": 1e-3, "epochs": 100, "top_k_words":30000}}, 
                       'pl':{'d_fr_eco':{"hidden_layers": 2, "hidden_units": 32, "dropout_rate": 0.2, "learning_rate": 1e-2, "epochs": 100, "top_k_words":30000}, 
                             'd_fr_lab':{"hidden_layers": 3, "hidden_units": 64, "dropout_rate": 0.2, "learning_rate": 1e-2, "epochs": 100, "top_k_words":20000}, 
                             'd_fr_sec':{"hidden_layers": 3, "hidden_units": 32, "dropout_rate": 0.2, "learning_rate": 1e-2, "epochs": 100, "top_k_words":30000}, 
                             'd_fr_wel':{"hidden_layers": 2, "hidden_units": 32, "dropout_rate": 0.2, "learning_rate": 1e-2, "epochs": 100, "top_k_words":20000}}, 
                       'ro':{'d_fr_eco':{"hidden_layers": 3, "hidden_units": 32, "dropout_rate": 0.2, "learning_rate": 1e-3, "epochs": 100, "top_k_words":30000}, 
                             'd_fr_lab':{"hidden_layers": 3, "hidden_units": 64, "dropout_rate": 0.2, "learning_rate": 1e-3, "epochs": 100, "top_k_words":20000}, 
                             'd_fr_sec':{"hidden_layers": 2, "hidden_units": 16, "dropout_rate": 0.2, "learning_rate": 1e-2, "epochs": 100, "top_k_words":30000}, 
                             'd_fr_wel':{"hidden_layers": 2, "hidden_units": 64, "dropout_rate": 0.2, "learning_rate": 1e-3, "epochs": 100, "top_k_words":20000}}, 
                       'sv':{'d_fr_eco':{"hidden_layers": 3, "hidden_units": 32, "dropout_rate": 0.2, "learning_rate": 1e-3, "epochs": 100, "top_k_words":20000}, 
                             'd_fr_lab':{"hidden_layers": 2, "hidden_units": 128, "dropout_rate": 0.2, "learning_rate": 1e-2, "epochs": 100, "top_k_words":10000}, 
                             'd_fr_sec':{"hidden_layers": 2, "hidden_units": 32, "dropout_rate": 0.2, "learning_rate": 1e-3, "epochs": 100, "top_k_words":30000}, 
                             'd_fr_wel':{"hidden_layers": 2, "hidden_units": 32, "dropout_rate": 0.2, "learning_rate": 1e-3, "epochs": 100, "top_k_words":30000}}, 
                       'uk':{'d_fr_eco':{"hidden_layers": 2, "hidden_units": 16, "dropout_rate": 0.2, "learning_rate": 1e-2, "epochs": 100, "top_k_words":30000}, 
                             'd_fr_lab':{"hidden_layers": 2, "hidden_units": 16, "dropout_rate": 0.2, "learning_rate": 1e-2, "epochs": 100, "top_k_words":30000}, 
                             'd_fr_sec':{"hidden_layers": 3, "hidden_units": 32, "dropout_rate": 0.2, "learning_rate": 1e-2, "epochs": 100, "top_k_words":30000}, 
                             'd_fr_wel':{"hidden_layers": 3, "hidden_units": 32, "dropout_rate": 0.2, "learning_rate": 1e-2, "epochs": 100, "top_k_words":30000}},
                       'hu':{'d_fr_eco':{"hidden_layers": 3, "hidden_units": 64, "dropout_rate": 0.2, "learning_rate": 1e-2, "epochs": 100, "top_k_words":30000}, 
                             'd_fr_lab':{"hidden_layers": 2, "hidden_units": 16, "dropout_rate": 0.2, "learning_rate": 1e-2, "epochs": 100, "top_k_words":30000}, 
                             'd_fr_sec':{"hidden_layers": 3, "hidden_units": 32, "dropout_rate": 0.2, "learning_rate": 1e-3, "epochs": 100, "top_k_words":30000}, 
                             'd_fr_wel':{"hidden_layers": 3, "hidden_units": 64, "dropout_rate": 0.2, "learning_rate": 1e-2, "epochs": 100, "top_k_words":20000}}}

best_classifier_params = {"MLP": best_classifiers_mlp}
# Settings
SUB_SAMPLE_RERUNS = 3
TFIDF = {
    'ngram_range': (1, 2),
    'dtype': 'int32',
    'strip_accents': 'unicode',
    'decode_error': 'replace',
    'analyzer': 'word',
    'min_df': 2,
}


TRAIN_TEST_PATH = f'/articles_dictionary_annotated_'
VALIDATION_SET_PATH = f'/articles_manual_annotated_925_'

SAMPLING = [100, 150, 200, 250, 300, 350,
            400, 450, 500, 600, 700, 800, 900, 1000]

# Single threaded version.
# Be careful: this might take very long!
output = train_classifiers(best_classifier_params)
output.to_csv(('results_evaluation_MLP.csv'), index=False)

print('DONE')