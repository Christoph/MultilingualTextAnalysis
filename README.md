# MultilingualTextAnalysis

This is the repository for the article:
Greasing the wheels for comparative communication research: Supervised text classification for multilingual corpora https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3840747.
The article is under review at Computational Communication Research (https://computationalcommunication.org/ccr/preprint).


The repository contains seven scripts (four for annotation with dictionaries, one for hyperparameter selection, one for classifier selection, and one for additional classifier evaluation with manual test data), two data files, and five files with results.


### Dictionary Annotation

Economy_Dictionary_Annotation.R\
Labor_Market_Dictionary_Annotation.R\
Security_Dictionary_Annotation.R\
Welfare_Dictionary_Annotation.R

were used to annotate the English documents regarding the occurance of the Economy & budget, Labor market, Security, and Welfare frame. 

### Hyperparameter selection
The hyperparameter_sampling.py file was used to identify the best hyperparameters per algorithm for each frame and language.

Structure:
- Everything before line 239 loads libraries and defines functions.
- Lines 239 - 314 define parameters for the three different classifiers, parameters are varied.
- Lines 316 - 334 define constant settings for preprocessing (tfidf; top k words were manually varied between 10,000 and 30,000 in each run) and sampling (kfold split, max number of observations).
- Lines 336 - 339 sample the data based on the sampling parameters.
- Lines 343 - 350 train classifiers and return information on the performance of each setting.

### Classifier Selection
The classification.py file was used to compare classification methods by algorithm and number of training documents.

Structure:
- Everything before line 254 loads libraries and defines functions.
- Lines 258 - 343 define the best performing settings for each classifier, language, and target (here frame) as dervied from the hyperparameter sampling procedure.
- Lines 350 - 358 define the constant parameters for the subsample reruns and the preprocessing (tfidf).
- Lines 361 - 362 define the varied sampling settings concerning the number of observations used to train the classifiers
- Lines 366 - 367 sample the datasets based on the sampling parameters, trains classifiers according to the predefined settings on the sampled datasets and returns the performance scores.

### Additional Classifier evaluation with separate manually labeled test data
The classifier_evaluation.py file was used to evaluate the best classifiers using separate manually annotated test data.

Structure:
- Everything before line 246 loads libraries and defines functions.
- Lines 246 - 273 define the best performing settings for the MLP classifier for each language and target (here frame).
- Lines 277 - 285 define the constant parameters for the subsample reruns and the preprocessing (tfidf).
- Lines 291 - 292 define the varied sampling settings concerning the number of observations used to train the classifiers
- Lines 296 - 297 train the MLP models on the sampled datasets and evaluate their performances using manually coded datasets.

### Data

The articles_dictionary_annotated_train_test_set.csv file includes training and test data. It was automatically annotated with the four dictionaries.

The articles_manual_annotated_test_set.csv file is the separate test data. It was manually annotated by native speakers.

### Results

results_hyperparamter_selection_rm.csv\
results_hyperparamter_selection_svm.csv\
results_hyperparamter_selection_mlp.csv 

include the results for hyperparamter selection.

dictionary_training_test_results.csv includes the results for classfier selection.

manual_test_results.csv includes the results for the classifiers which were trained with dictionary annotated date and tested with manually annotated test data.




