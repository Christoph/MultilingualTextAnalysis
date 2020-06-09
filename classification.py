import pandas as pd
import numpy as np
import re
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer


def preprocess_text(text):
    # replace non characers with space
    # regexr = re.sub(r"[^a-zA-Z0-9.!? ]+", " ", text)
    temp = re.sub(r"[/W/D/S.,-]+", " ", text.lower())
    # merge multiple spaces to a single one
    return re.sub(r"[ ]+", " ", temp)


# language = 'es'

# # Load data
# data = pd.read_csv('data/articles_dictionary_annotated_'+language+'.csv')


def prepare_datasets(language, target, sampling, max_df_list):
    data = pd.read_csv('data/articles_dictionary_annotated_'+language+'.csv')
    output = []

    # Preprocess text
    data['clean'] = data['all_text_orig_lang_lemma'].apply(
        lambda x: preprocess_text(x))

    positive = data[data[target] == 1]
    negative = data[data[target] == 0]

    for (positive_count, negative_count) in sampling:
        try:
            sample = positive.sample(positive_count)
            sample = sample.append(negative.sample(negative_count))

            for maxdf in max_df_list:
                tfidf = TfidfVectorizer(max_df=maxdf)
                vecs = tfidf.fit_transform(sample['clean'])
                target_vec = sample[target]

                output.append([
                    ' '.join([
                        language,
                        target,
                        'tfidf-'+str(maxdf),
                        str(positive_count),
                        str(negative_count)
                    ]),
                    vecs,
                    target_vec
                ])
        except:
            print('Too many samples:', target, sampling, max_df_list)

    return output


sampling = [
    [1000, 1000],
    [500, 1000],
    [500, 500],
    [250, 250],
]
tfidf_maxdf = [0.5, 0.7, 0.9]

classifications = [
    ["DecisionTree", DecisionTreeClassifier, [
        {"criterion": "gini", "min_samples_split": 0.01},
        {"criterion": "entropy", "min_samples_split": 0.01},
        {"criterion": "gini", "min_samples_split": 0.05},
        {"criterion": "entropy", "min_samples_split": 0.05},
        # {"criterion": "gini"},
        # {"criterion": "entropy"},
    ]],
    # ["AdaBoost", AdaBoostClassifier, [
    #     {"n_estimators": 25, "learning_rate": 1},
    #     # {"n_estimators": 25, "learning_rate": 0.5},
    #     # {"n_estimators": 50, "learning_rate": 1},
    #     # {"n_estimators": 100, "learning_rate": 1},
    #     # {"n_estimators": 200, "learning_rate": 1},
    #     # {"n_estimators": 300, "learning_rate": 1},
    # ]],
    # ["GradientBoostingClassifier", GradientBoostingClassifier, [
    #     {"n_estimators": 25},
    #     # {"n_estimators": 50},
    #     {"n_estimators": 100},
    #     # {"n_estimators": 200},
    #     # {"n_estimators": 300},
    # ]],
    # ["SVM", SVC, [
    #     {"gamma": "scale", "kernel": "rbf"},
    #     {"gamma": "scale", "kernel": "linear"},
    # ]],
    ["Random Forest", RandomForestClassifier, [
        # {"n_estimators": 200, "criterion": "entropy", "min_samples_split": 0.01},
        # {"n_estimators": 200, "criterion": "entropy", "min_samples_split": 0.05},
        {"n_estimators": 100, "criterion": "gini"},
        {"n_estimators": 100, "criterion": "entropy"},
        # {"n_estimators": 200, "criterion": "gini"},
        #     {"n_estimators": 200, "criterion": "entropy"},
        #     {"n_estimators": 300, "criterion": "gini"},
        #     {"n_estimators": 300, "criterion": "entropy"},
        #     {"n_estimators": 200, "criterion": "gini", "max_leaf_nodes": 179},
        #     {"n_estimators": 200, "criterion": "entropy", "max_leaf_nodes": 179},
    ]],
    # ["MLP", MLPClassifier, [
    #     {"hidden_layer_sizes": 5, "activation": "relu",
    #         "solver": "lbfgs", "max_iter": 200},
    #     # {"hidden_layer_sizes": 10, "activation": "relu",
    #     #     "solver": "lbfgs", "max_iter": 200},
    #     # {"hidden_layer_sizes": 20, "activation": "relu",
    #     #     "solver": "lbfgs", "max_iter": 200},
    #     # {"hidden_layer_sizes": 20, "activation": "relu",
    #     #     "solver": "lbfgs", "max_iter": 300},
    #     # {"hidden_layer_sizes": 50, "activation": "relu",
    #     #     "solver": "lbfgs", "max_iter": 200},
    #     # {"hidden_layer_sizes": (10, 10), "activation": "relu",
    #     #     "solver": "lbfgs", "max_iter": 200},
    #     # {"hidden_layer_sizes": (20, 20, 20, 20, 5), "activation": "relu",
    #     #  "solver": "lbfgs", "max_iter": 200},
    #     # {"hidden_layer_sizes": (50, 50, 50), "activation": "relu",
    #     #  "solver": "lbfgs", "max_iter": 200},
    #     # {"hidden_layer_sizes": (50, 20, 10), "activation": "relu",
    #     #  "solver": "lbfgs", "max_iter": 200},
    #     {"hidden_layer_sizes": (20, 20, 20), "activation": "relu",
    #      "solver": "lbfgs", "max_iter": 200},
    #     # {"hidden_layer_sizes": (20, 20, 10), "activation": "relu",
    #     #  "solver": "lbfgs", "max_iter": 200},
    #     # {"hidden_layer_sizes": (20, 20, 20), "activation": "relu",
    #     #  "solver": "lbfgs", "max_iter": 300},
    #     # {"hidden_layer_sizes": (20, 20, 20), "activation": "relu",
    #     #  "solver": "lbfgs", "max_iter": 400},
    # ]]
]


def find_best_classifier(classifications):
    out = pd.DataFrame(
        columns=["Dataset", "Method", "Params", "Accuracy", "Precision", "Recall"])

    # Iterate datasets
    for target_id, target in enumerate(['d_fr_eco', 'd_fr_lab', 'd_fr_sec']):
        target_name = target[0]
        y = target[1]

        print("targets: ", str(target_id+1), "/", str(3))

        datasets = prepare_datasets(
            'es',
            target,
            sampling,
            tfidf_maxdf)

        for data_id, dataset in enumerate(datasets):
            dataset_name = dataset[0]
            data = dataset[1]
            y = np.array(dataset[2])
            skf = ShuffleSplit(n_splits=3)
            split_indices = []

            for train_index, test_index in skf.split(data, y):
                split_indices.append((train_index, test_index))

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

                        # Iterate splits
                        for train_index, test_index in split_indices:

                            X_train, X_test = data[train_index], data[test_index]
                            y_train, y_test = y[train_index], y[test_index]

                            if len(y.shape) > 1:
                                clf = MultiOutputClassifier(
                                    classification[1](**param))
                            else:
                                clf = classification[1](**param)
                            try:
                                clf.fit(X_train, y_train)
                                y_pred = clf.predict(X_test)
                                prfs = precision_recall_fscore_support(
                                    y_test, y_pred, warn_for=[])

                                acc_scores.append(clf.score(X_test, y_test))
                                pre_scores.append(prfs[0].mean())
                                rec_scores.append(prfs[1].mean())
                            except:
                                print("Exception during fitting")
                                acc_scores.append(0)
                                pre_scores.append(0)
                                rec_scores.append(0)

                        clf_acc = np.array(acc_scores).mean()
                        clf_pre = np.array(pre_scores).mean()
                        clf_rec = np.array(rec_scores).mean()
                        out = out.append(pd.DataFrame([[dataset_name, clf_name, str(
                            param), clf_acc, clf_pre, clf_rec]], columns=out.columns), ignore_index=True)

                    out.to_csv("results.csv", index=False)

    # Final save
    out.to_csv("results.csv", index=False)

    print("DONE!")


find_best_classifier(classifications)
