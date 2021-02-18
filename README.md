# MultilingualTextAnalysis

This is the repository for the article:
Greasing the wheels for comparative communication research: Supervised text classification for multilingual corpora.


The repository contains six scripts (four for annotation with dictionaries, one for classifier selection and one for model validation), two data files, and two files with results.


### Dictionary Annotation

Economy_Dictionary_Annotation.R\
Labor_Market_Dictionary_Annotation.R\
Security_Dictionary_Annotation.R\
Welfare_Dictionary_Annotation.R\

were used to annotate the English documents regarding the occurance of the Economy & budget, Labor market, Security, and Welfare frame. 


### Classifier Selection
The classification.py file was used to compare classification methods for smaller datasets.

Structure:
- Everything before line 280 loads libraries and functions.
- Lines 281 - 303 sample the datasets based on the sampling parameters. 
Sampling Parameters:
languages = ['de', 'es', 'pl', 'ro', 'sv', 'uk']
targets = ['d_fr_eco', 'd_fr_lab', 'd_fr_sec', 'd_fr_wel']
sampling = [100, 150, 200, 250, 300, 350,
            400, 450, 500, 600, 700, 800, 900, 1000, 1500, 2000, 'max_pos', 0]
(Be careful if you use all parameters it will take a lot of time.)
- Lines 309 - 316 train the classification models on all sampled datasets using multithreading.
(Again be careful this step can take very long)

### Validation
The validation.py file was used to validate the best classification algorithm using separate validation data.

Structure:
- Everything before line 233 loads libraries and functions.
- Lines 234 - 257 sample the datasets based on the sampling parameters. 
Sampling Parameters:
languages = ['de', 'es', 'pl', 'ro', 'sv', 'uk']
targets = ['d_fr_eco', 'd_fr_lab', 'd_fr_sec', 'd_fr_wel']
sampling = [100, 150, 200, 250, 300, 350,
            400, 450, 500, 600, 700, 800, 900, 1000, 1500, 2000, 'max_pos', 0]
(Be careful if you use all parameters it will take a lot of time.)
- Lines 264 - 272 train the MLP model on the sampled datasets and validate using the validation datasets.
(Again be careful this step can take very long)

### Data

The file is the sperate validation data. It was automatically annotated by the four dictionaries.

The file is the sperate validation data. It was manually annotated by native speakers.

### Results

The file results for classfier selection and validation




