# MultilingualTextAnalysis

This is the repository for the article:
Greasing the wheels for comparative communication research: Supervised text classification for multilingual corpora https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3840747.
The article is under review at Computational Communication Research (https://computationalcommunication.org/ccr/preprint).


The repository contains seven scripts (four for annotation with dictionaries, one for hyperparameter selection, one for classifier selection, and one for additional classifier evaluation with manual test data), two data files, and five files with results.


### Dictionary Annotation

Economy_Dictionary_Annotation.R\
Labor_Market_Dictionary_Annotation.R\
Security_Dictionary_Annotation.R\
Welfare_Dictionary_Annotation.R\

were used to annotate the English documents regarding the occurance of the Economy & budget, Labor market, Security, and Welfare frame. 

### Hyperparameter selection
The hyperparameter_selection.py file was used to identify the best hyperparameter per algorithm for each frame and language.

Structure:
- Everything before line 239 loads libraries and defines functions.
- Lines 239 - 334 define parameters and settings for the three different classifiers that are varied as well as preprocessing steps (tfidf) and sampling.
- Lines 336 - 339 sample the data based on the sampling parameters.
- Lines 343 - 350 train classifiers and return information on the performance of each setting.

### Classifier Selection
The classification.py file was used to compare classification methods by algorithm and number of training documents.

Structure:
- Everything before line 245 loads libraries and defines functions.
- Lines 246 - 275 define the best performing settings for the MLP classifier for each language and target
- Lines 275 - 292 define the parameters for the preprocessing and sampling procedure
- Lines 296 - 297 sample the datasets based on the sampling parameters, trains classifiers according to the predefined settings on the sampled datasets and returns the performance scores.

### Additional Classifier evaluation with separate manually labeled test data
The classifier_evaluation.py file was used to evaluate the best classifiers using separate manually annotated test data.

Structure:
- Everything before line 264 loads libraries and defines functions.
- Lines 265 - 354 define the best performing settings for each classifier, language, and target as dervied from the hyperparameter sampling procedure
- Lines 355 - 317 define the parameters for the preprocessing and sampling procedure
- Lines 375 - 376 train the MLP models on the sampled datasets and evaluate their performances using manually coded datasets.

### Data

The articles_dictionary_annotated_train_test_set.csv file includes training and test data. It was automatically annotated with the four dictionaries.

The articles_manual_annotated_test_set.csv file is the separate test data. It was manually annotated by native speakers.

### Results

The files results_hyperparamter_selection_rm.csv, results_hyperparamter_selection_svm.csv, and results_hyperparamter_selection_mlp.csv include the results for hyperparamter selection.

The dictionary_training_test_results.csv file includes the results for classfier selection.

The manual_test_results.csv file the results for the testing classifiers which were trained with dictionary annotated date with manually annotated test data.




