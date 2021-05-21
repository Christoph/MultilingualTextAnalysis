# coding: utf-8

# In[77]:

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_recall_fscore_support
import random
from keras import models
from keras.layers import Dropout, Dense
import pandas as pd
import numpy as np
import re
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_selection import SelectKBest, f_classif
import multiprocessing as mp
from math import ceil
import tensorflow as tf


# In[78]:
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

# # Load data
# data = pd.read_csv('data/articles_dictionary_annotated_'+language+'.csv')


def sample_datasets(datasets, language, targets, tfidf_parameters, sample_reruns):
    """Sample the datasets."""
    # Load the csv
    data = pd.read_csv(TRAIN_TEST_PATH+language+'.csv')

    # preprocess the text
    data['clean'] = data['all_text_orig_lang_lemma'].apply(
        lambda x: preprocess_text(x))

    for target in targets:
        print(target)
        for tfidf in tfidf_parameters:
            for _ in range(sample_reruns):
                # Sample as many text as there are positive samples
                positive = data[data[target] == 1]
                negative = data[data[target] == 0]

                sample = positive
                sample = sample.append(negative.sample(len(positive)))

                sampled_data = sample

                positive_count = len(
                    sampled_data[sampled_data[target] == 1])
                negative_count = len(
                    sampled_data[sampled_data[target] == 0])

                datasets.append(prepare_datasets(
                    sampled_data,
                    target,
                    tfidf,
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
    selector = SelectKBest(f_classif, k=min(TOP_K_WORDS, x_train.shape[1]))
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


def hyperparameter_sampling(datasets):
    # Create result dataframe
    out = pd.DataFrame(
        columns=["Dataset", "Classifier", "Params", "Accuracy", "F1", "Precision", "Recall"])

    # Iterate the datasets
    for data_id, dataset in enumerate(datasets):
        dataset_name = dataset[0]
        data = dataset[1]
        y = np.array(dataset[2])
        skf = StratifiedKFold(n_splits=SUMBER_OF_KFOLD_SPLITS)
        split_indices = []
        print(dataset_name)

        for train_indices, test_indices in skf.split(data, y):
            split_indices.append((train_indices, test_indices))

            print("datasets: ", str(data_id+1), "/", str(len(datasets)))

            # Iterate classifications
            for cls_id, classification in enumerate(classifications):
                clf_name = classification[0]
                clf_params = classification[2]

                print("classifier: ", clf_name, ", ", str(
                    cls_id+1), "/", len(classifications))

                # Iterate parametrizations
                for p_id, param in enumerate(clf_params):
                    print("Params: ", param, ", ", str(
                        p_id+1), "/"+str(len(clf_params)))

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

                        if clf_name == 'MLP':
                            # Create model instance.
                            model = mlp_model(layers=param["hidden_layers"], units=param["hidden_units"], dropout_rate=param["dropout_rate"],
                                              input_shape=X_train.shape[1:], num_classes=2)
                            optimizer = tf.keras.optimizers.Adam(
                                lr=param["learning_rate"])
                            model.compile(optimizer=optimizer,
                                          loss='binary_crossentropy', metrics=['acc'])

                            # Stop training is validation loss doesnt decrease for 3 steps
                            callbacks = [tf.keras.callbacks.EarlyStopping(
                                monitor='val_loss', patience=2)]

                            # Train and validate model.
                            history = model.fit(
                                X_train,
                                y_train,
                                epochs=param["epochs"],
                                callbacks=callbacks,
                                validation_data=(X_test, y_test),
                                verbose=0,
                                batch_size=512)

                            acc_scores.append(
                                history.history['val_acc'][-1])
                            y_pred = [round(a[0])
                                      for a in model.predict(X_test)]
                        else:
                            model = classification[1](**param)

                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            prfs = precision_recall_fscore_support(
                                y_test, y_pred, warn_for=[])

                            acc_scores.append(
                                model.score(X_test, y_test))
                            y_pred = model.predict(X_test)

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
                        [[dataset_name, clf_name, str(clf_params), clf_acc, clf_f1, clf_pre, clf_rec]], columns=out.columns), ignore_index=True)

    return out


# %%
classifications = [
    # ["DecisionTree", DecisionTreeClassifier, [
    #     #    {"criterion": "gini", "min_samples_split": 1e-2},
    #     #    {"criterion": "entropy", "min_samples_split": 1e-2},
    #     #    {"criterion": "gini", "min_samples_split": 0.05},
    #     #    {"criterion": "entropy", "min_samples_split": 0.05},
    #     #    {"criterion": "gini"},
    #     {"criterion": "entropy"},
    # ]],
    # ["AdaBoost", AdaBoostClassifier, [
    #     {"n_estimators": 25, "learning_rate": 1},
    #     # {"n_estimators": 25, "learning_rate": 0.5},
    #     # {"n_estimators": 50, "learning_rate": 1},
    #     # {"n_estimators": 100, "learning_rate": 1},
    #     # {"n_estimators": 200, "learning_rate": 1},
    #     # {"n_estimators": 300, "learning_rate": 1},
    # ]],
    # ["GradientBoostingClassifier", GradientBoostingClassifier, [
    #     #     {"n_estimators": 25},
    #     # {"n_estimators": 50},
    #     #     {"n_estimators": 100},
    #     {"n_estimators": 200},
    #     # {"n_estimators": 300},
    # ]],
    ["SVM", SVC, [
        {"C": 1, "kernel": "rbf"},
        {"C": 1, "kernel": "linear"},
        {"C": 1, "kernel": "sigmoid"},
        {"C": 1, "kernel": "poly"},
        {"C": 3, "kernel": "rbf"},
        {"C": 3, "kernel": "linear"},
        {"C": 3, "kernel": "sigmoid"},
        {"C": 3, "kernel": "poly"},
        {"C": 5, "kernel": "rbf"},
        {"C": 5, "kernel": "linear"},
        {"C": 5, "kernel": "sigmoid"},
        {"C": 5, "kernel": "poly"}
    ]],
    # ["Nearest Neighbor", KNeighborsClassifier, [
    #     {'n_neighbors': 3},
    #     #    {'n_neighbors':4},
    #     #    {'n_neighbors':5},
    #     #    {'n_neighbors':6},
    # ]],
    # ["Naive Bayes Gaussian", MultinomialNB, [
    #     {}]],
    ["Random Forest", RandomForestClassifier, [
        {"n_estimators": 50, "criterion": "entropy"},
        {"n_estimators": 100, "criterion": "entropy"},
        {"n_estimators": 200, "criterion": "entropy"},
        {"n_estimators": 300, "criterion": "entropy"},
        {"n_estimators": 50, "criterion": "gini"},
        {"n_estimators": 100, "criterion": "gini"},
        {"n_estimators": 200, "criterion": "gini"},
        {"n_estimators": 300, "criterion": "gini"},
    ]],
    ["MLP", "Keras", [
        {'hidden_layers': 2, 'hidden_units': 16, 'dropout_rate': 0.2,
            'learning_rate': 1e-2, 'epochs': 100}, 
        {'hidden_layers': 2, 'hidden_units': 16, 'dropout_rate': 0.2,
            'learning_rate': 1e-3, 'epochs': 100}, 
        {'hidden_layers': 2, 'hidden_units': 16, 'dropout_rate': 0.2,
            'learning_rate': 1e-4, 'epochs': 100}, 
        {'hidden_layers': 2, 'hidden_units': 32, 'dropout_rate': 0.2,
            'learning_rate': 1e-2, 'epochs': 100}, 
        {'hidden_layers': 2, 'hidden_units': 32, 'dropout_rate': 0.2,
            'learning_rate': 1e-3, 'epochs': 100}, 
        {'hidden_layers': 2, 'hidden_units': 32, 'dropout_rate': 0.2,
            'learning_rate': 1e-4, 'epochs': 100}, 
        {'hidden_layers': 2, 'hidden_units': 64, 'dropout_rate': 0.2,
            'learning_rate': 1e-2, 'epochs': 100}, 
        {'hidden_layers': 2, 'hidden_units': 64, 'dropout_rate': 0.2,
            'learning_rate': 1e-3, 'epochs': 100}, 
        {'hidden_layers': 2, 'hidden_units': 64, 'dropout_rate': 0.2,
            'learning_rate': 1e-4, 'epochs': 100}, 
        {'hidden_layers': 2, 'hidden_units': 128, 'dropout_rate': 0.2,
            'learning_rate': 1e-2, 'epochs': 100}, 
        {'hidden_layers': 2, 'hidden_units': 128, 'dropout_rate': 0.2,
            'learning_rate': 1e-3, 'epochs': 100}, 
        {'hidden_layers': 2, 'hidden_units': 128, 'dropout_rate': 0.2,
            'learning_rate': 1e-4, 'epochs': 100}, 
        {'hidden_layers': 3, 'hidden_units': 16, 'dropout_rate': 0.2,
            'learning_rate': 1e-2, 'epochs': 100}, 
        {'hidden_layers': 3, 'hidden_units': 16, 'dropout_rate': 0.2,
            'learning_rate': 1e-3, 'epochs': 100}, 
        {'hidden_layers': 3, 'hidden_units': 16, 'dropout_rate': 0.2,
            'learning_rate': 1e-4, 'epochs': 100}, 
        {'hidden_layers': 3, 'hidden_units': 32, 'dropout_rate': 0.2,
            'learning_rate': 1e-2, 'epochs': 100}, 
        {'hidden_layers': 3, 'hidden_units': 32, 'dropout_rate': 0.2,
            'learning_rate': 1e-3, 'epochs': 100}, 
        {'hidden_layers': 3, 'hidden_units': 32, 'dropout_rate': 0.2,
            'learning_rate': 1e-4, 'epochs': 100}, 
        {'hidden_layers': 3, 'hidden_units': 64, 'dropout_rate': 0.2,
            'learning_rate': 1e-2, 'epochs': 100}, 
        {'hidden_layers': 3, 'hidden_units': 64, 'dropout_rate': 0.2,
            'learning_rate': 1e-3, 'epochs': 100}, 
        {'hidden_layers': 3, 'hidden_units': 64, 'dropout_rate': 0.2,
            'learning_rate': 1e-4, 'epochs': 100}, 
        {'hidden_layers': 3, 'hidden_units': 128, 'dropout_rate': 0.2,
            'learning_rate': 1e-2, 'epochs': 100}, 
        {'hidden_layers': 3, 'hidden_units': 128, 'dropout_rate': 0.2,
            'learning_rate': 1e-3, 'epochs': 100}, 
        {'hidden_layers': 3, 'hidden_units': 128, 'dropout_rate': 0.2,
            'learning_rate': 1e-4, 'epochs': 100}
    ]]
]

tfidf_parameters = [{
    'ngram_range': (1, 2),
    'dtype': 'int32',
    'strip_accents': 'unicode',
    'decode_error': 'replace',
    'analyzer': 'word',
    'min_df': 2,
}]


# In[79]:
datasets = []

SUMBER_OF_KFOLD_SPLITS = 3
SUB_SAMPLE_RERUNS = 1
TOP_K_WORDS = 20000


TRAIN_TEST_PATH = 'data/articles_dictionary_annotated_'
languages = ['de', 'es', 'pl', 'ro', 'sv', 'uk']
targets = ['d_fr_eco', 'd_fr_lab', 'd_fr_sec', 'd_fr_wel']

for language in languages:
    print(language)
    sample_datasets(datasets, language, targets,
                    tfidf_parameters, SUB_SAMPLE_RERUNS)

print("Datasets are ready")

# In[ ]:
pool = mp.Pool(processes=(mp.cpu_count()))
results = pool.map(hyperparameter_sampling, buckets(
    datasets, ceil(len(datasets)/(mp.cpu_count()))))
pool.close()
pool.join()

output = pd.concat(results)
output.to_csv(('results_hyperparamter_sampling.csv'), index=False)

print('Hyperparameter sampling is finished')
