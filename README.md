# MultilingualTextAnalysis

This is the repository for the article:
Greasing the wheels for comparative communication research: Supervised text classification for multilingual corpora https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3840747.
The article is under review at Computational Communication Research (https://computationalcommunication.org/ccr/preprint).


The repository contains seven scripts (four for annotation with dictionaries, one for hyperparameter selection, one for classifier selection, and one for strategy evaluation with manual test data), two data files, and three files with results.


### Dictionary Annotation

Economy_Dictionary_Annotation.R\
Labor_Market_Dictionary_Annotation.R\
Security_Dictionary_Annotation.R\
Welfare_Dictionary_Annotation.R\

were used to annotate the English documents regarding the occurance of the Economy & budget, Labor market, Security, and Welfare frame. 


### Classifier Selection

The hyperparameter_selection.py file was used to identify the best hyperparameter per algorithm for each frame and language.

The classification.py file was then used to compare classifier performance by algorithm and number of training documents.

Structure:
- Everything before line 280 loads libraries and functions.
- Lines 281 - 303 sample the datasets based on the sampling parameters. 
Sampling Parameters:
languages = ['de', 'es', 'pl', 'ro', 'sv', 'en']
targets = ['d_fr_eco', 'd_fr_lab', 'd_fr_sec', 'd_fr_wel']
sampling = [100, 150, 200, 250, 300, 350,
            400, 450, 500, 600, 700, 800, 900, 1000]
(Be careful if you use all parameters it will take a lot of time.)
- Lines 309 - 316 train the classification models on all sampled datasets using multithreading.
(Again be careful this step can take very long)

### Classifier evaluation with separate manually labeled test data
The manual_test.py file was used to evaluate the best classifiers using separate manually annotated test data.

Structure:
- Everything before line 233 loads libraries and functions.
- Lines 234 - 257 sample the datasets based on the sampling parameters. 
Sampling Parameters:
languages = ['de', 'es', 'pl', 'ro', 'sv', 'en']
targets = ['d_fr_eco', 'd_fr_lab', 'd_fr_sec', 'd_fr_wel']
sampling = [100, 150, 200, 250, 300, 350,
            400, 450, 500, 600, 700, 800, 900, 1000]
(Be careful if you use all parameters it will take a lot of time.)
- Lines 264 - 272 train the MLP model on the sampled datasets and validate using the validation datasets.
(Again be careful this step can take very long)

### Data

The articles_dictionary_annotated_train_test_set.csv file includes training and test data. It was automatically annotated with the four dictionaries.

The articles_manual_annotated_test_set.csv file is the separate test data. It was manually annotated by native speakers.

### Results

The files results_hyperparamter_selection_rm.csv, results_hyperparamter_selection_svm.csv, and results_hyperparamter_selection_mlp.csv include the results for hyperparamter selection.

The dictionary_training_test_results.csv file includes the results for classfier selection.

The manual_test_results.csv file the results for the testing classifiers which were trained with dictionary annotated date with manually annotated test data.




