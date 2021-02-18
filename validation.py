
# coding: utf-8

# In[135]:

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from keras import models
from keras.layers import Dropout
from keras.layers import Dense
import tensorflow as tf
from math import ceil
import multiprocessing as mp
import random
# In[107]:
# Load all functions


def preprocess_text(text):
    """Remove non-characters and lower case the text"""
    # replace non characers with space and lower case
    temp = re.sub(r"[/W/D/S.,-]+", " ", str(text).lower())
    # merge multiple spaces to a single one
    return re.sub(r"[ ]+", " ", temp)


def train_classifiers(datasets):
    out = pd.DataFrame(
        columns=["Dataset", "Classifier", "Accuracy", "F1", "Precision", "Recall"])

    # Iterate datasets
    for data_id, dataset in enumerate(datasets):
        dataset_name = dataset[0]
        data = dataset[1]
        y = np.array(dataset[2])
        val_data = dataset[3]
        val_y = np.array(dataset[4])

        print(data_id+1, '/', len(datasets), ':', dataset_name)

        acc_scores = []
        pre_scores = []
        rec_scores = []
        f1_scores = []

        # Iterate splits
        global X_train
        X_train, X_test = data, val_data
        y_train, y_test = y, val_y
        y_pred = None

        # model = return_mlp()
        model = mlp_model(layers=2, units=64, dropout_rate=0.2,
                          input_shape=X_train.shape[1:])
        optimizer = tf.keras.optimizers.Adam(lr=1e-3)
        model.compile(optimizer=optimizer,
                      loss='binary_crossentropy', metrics=['acc'])

        callbacks = [tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=3)]

        # Train and validate model.
        history = model.fit(
            X_train,
            y_train,
            epochs=100,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=0,  # use 2 to log once per epoch.
            batch_size=512)

        acc_scores.append(history.history['val_acc'][-1])
        y_pred = [round(a[0]) for a in model.predict(X_test)]

        prfs = precision_recall_fscore_support(
            y_test, y_pred, warn_for=[])

        pre_scores.append(prfs[0].mean())
        rec_scores.append(prfs[1].mean())
        f1_scores.append(prfs[2].mean())

        clf_acc = np.array(acc_scores).mean()
        clf_pre = np.array(pre_scores).mean()
        clf_rec = np.array(rec_scores).mean()
        clf_f1 = np.array(f1_scores).mean()
        out = out.append(pd.DataFrame(
            [[dataset_name, 'MLP', clf_acc, clf_f1, clf_pre, clf_rec]], columns=out.columns), ignore_index=True)

    return out


def buckets(data, n):
    """Return a factory that yields buckets with size n."""
    # Shuffle all datasets to get a more consistent workload for all threads.
    random.shuffle(data)

    for i in range(0, len(data), n):
        yield data[i:i + n]


def sample_datasets(datasets, language, targets, sampling, tfidf_parameters, sample_reruns):
    """Create all sample datasets."""
    data = pd.read_csv(TRAIN_TEST_PATH+language+'.csv')
    validation_data = pd.read_csv(
        VALIDATION_SET_PATH+language+'.csv')

    data['clean'] = data['all_text_orig_lang_lemma'].apply(
        lambda x: preprocess_text(x))
    validation_data['clean'] = validation_data['all_text_orig_lang_lemma'].apply(
        lambda x: preprocess_text(x))

    # Rename columns
    data.rename(columns={'d_fr_eco': 'fr_eco'}, inplace=True)
    data.rename(columns={'d_fr_lab': 'fr_lab'}, inplace=True)
    data.rename(columns={'d_fr_sec': 'fr_sec'}, inplace=True)
    data.rename(columns={'d_fr_wel': 'fr_wel'}, inplace=True)

    validation_data.rename(columns={'m_fr_eco': 'fr_eco'}, inplace=True)
    validation_data.rename(columns={'m_fr_lab': 'fr_lab'}, inplace=True)
    validation_data.rename(columns={'m_fr_sec': 'fr_sec'}, inplace=True)
    validation_data.rename(columns={'m_fr_wel': 'fr_wel'}, inplace=True)

    for target in targets:
        print(target)
        for s in sampling:
            for _ in range(sample_reruns):
                if s == 'max_pos':
                    positive = data[data[target] == 1]
                    negative = data[data[target] == 0]

                    sample = positive
                    sample = sample.append(negative.sample(len(positive)))

                    sampled_data = sample
                elif s > 0:
                    positive = data[data[target] == 1]
                    negative = data[data[target] == 0]

                    if len(positive) > s:
                        sample = positive.sample(s)
                        sample = sample.append(negative.sample(s))
                    else:
                        continue

                    sampled_data = sample
                else:
                    sampled_data = data

                positive_count = len(sampled_data[sampled_data[target] == 1])
                negative_count = len(sampled_data[sampled_data[target] == 0])

                print('Pos:', positive_count, 'Neg:',  negative_count)

                datasets.append(prepare_datasets(
                    sampled_data,
                    validation_data,
                    target,
                    tfidf_parameters,
                    positive_count,
                    negative_count
                ))


def prepare_datasets(data, validation_data, target, tfidf_parameters, positive_count, negative_count):
    """Create the tfidf vectors for a specific dataset and return metadata, vectors, and labels."""
    vectorizer = TfidfVectorizer(**tfidf_parameters)

    # Learn vocabulary from training texts and vectorize training texts.
    x_train = vectorizer.fit_transform(data['clean'])
    train_labels = data[target]

    # Vectorize validation texts.
    x_val = vectorizer.transform(validation_data['clean'])
    val_labels = validation_data[target]

    # Select top 'k' of the vectorized features.
    selector = SelectKBest(f_classif, k=min(20000, x_train.shape[1]))
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


def mlp_model(layers, units, dropout_rate, input_shape):
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

    op_units = 1
    op_activation = 'sigmoid'
    model = models.Sequential()
    model.add(Dropout(rate=dropout_rate, input_shape=input_shape))

    for _ in range(layers-1):
        model.add(Dense(units=units, activation='relu'))
        model.add(Dropout(rate=dropout_rate))

    model.add(Dense(units=op_units, activation=op_activation))
    return model


# In[193]:
# Prepare all datasets

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
VALIDATION_SET_PATH = "validate/articles_manual_annotated_925_"
languages = ['de', 'hu', 'pl', 'ro', 'uk', 'sv', 'es']
targets = ['fr_eco', 'fr_lab', 'fr_sec', 'fr_wel']
sampling = [100, 150, 200, 250, 300, 350,
            400, 450, 500, 600, 700, 800, 900, 1000]

for language in languages:
    print(language)
    sample_datasets(datasets, language, targets,
                    sampling, tfidf_parameters, SUB_SAMPLE_RERUNS)


# %%
# Run all the different dataset and model combinations.
# Fast version using data parallelism.
# Use this cell OR the cell below.

pool = mp.Pool(processes=(mp.cpu_count() - 1))
results = pool.map(train_classifiers, buckets(
    datasets, ceil(len(datasets)/(mp.cpu_count() - 1))))
pool.close()
pool.join()

output = pd.concat(results)

output.to_csv(('results_validation.csv'), index=False)

print('DONE')
# %%
# Single threaded version.
# Be careful: this might take very long!
output = train_classifiers(datasets)
output.to_csv(('results_validation.csv'), index=False)
