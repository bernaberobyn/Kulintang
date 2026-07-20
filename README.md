# Kulintang Project : From Oral Tradition to Digital Sheet: A Machine Learning Approach to Kulintang Music Transcription

## Oerview
Kulintang music is a central component of the indigenous musical heritage of the southern Philippines, yet its transmission is increasingly at risk due to the absence of standardized written notation and its reliance on oral pedagogy. This limitation makes long-term preservation and global accessibility challenging. To address this gap, we present the first automated framework for converting kulintang performance videos into musical notes using machine learning algorithms. We constructed a novel dataset of 1,920 audio segments derived from recordings of nine kulintang instruments, each exhibiting distinct tunings, timbral properties, and culturally specific acoustic characteristics. Audio was segmented through silence detection and transformed into spectral features optimized for per-note classification. We benchmarked three supervised learning models (k-Nearest Neighbors,  Random Forest, and Multilayer Perceptron) to evaluate their effectiveness in recognizing kulintang notes across heterogeneous instruments. Random Forest achieved the strongest performance, with 90.03% cross-validation accuracy and 87% test accuracy, indicating the feasibility of data-driven transcription for this musical form. This work represents one of the earliest computational approaches for automated kulintang transcription and contributes new resources, methodologies, and tools that support preservation, education, and broader engagement with Filipino cultural heritage.

## Dataset [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/datasets/maryrobynbernabe/kulintang-dataset)
The original dataset for this study was collected from video recordings of several kulintang players (with different kulintangs) across different universities and schools in the Philippines. Each performer played a full scale of the instrument, striking each individual gong from the lowest to the highest pitch. As a result, this produced a total of 1920 audio segments recorded from nine different kulintangs to provide a varied dataset. Since kulintangs are created by hand, each instrument is not standardized, where each can have tonal variability and different pitches and tunings. When preparing the data for analysis, the audio was extracted and labelled manually according to its pitch number (1-8). The full dataset can be found [here](https://www.kaggle.com/datasets/maryrobynbernabe/kulintang-dataset).

## Notebooks

### kulintang-1-video-segmentation-silence-interval [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/maryrobynbernabe/kulintang-1-video-segmentation-silence-interval)
This notebook breaks videos into segments where actual notes are played. The final output of the notebook is again uploaded as a dataset for user's conveneince [here](https://www.kaggle.com/datasets/maryrobynbernabe/kulintang-videosegments)

### kulintang-2-audio-extraction-and-featurization [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/maryrobynbernabe/kulintang-2-audio-extraction-and-featurization)
This notebook helps in extracting audios from the segments and convert them into MFCC features. The output is stored as a .csv file containing MFCC features with their labels.

### kulintang-3-train-test-split-csv [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/maryrobynbernabe/kulintang-3-train-test-split-csv)
This notebook helps in splitting the .csv file into train and test. The output is 3 separate .csv files.

### kulintang-4-classification-model-development [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/maryrobynbernabe/kulintang-4-classification-model-development)
This notebook helps in building various Machine Learning models using 5-fold cross validation. The output of this notebook is the best performing model saved a a pickle file.

### kulintang-5-testing-the-best-model [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/maryrobynbernabe/kulintang-5-testing-the-best-model)
This notebook used the test split from the dataset and passes it through the best model to check its performance. The performance report and confusion matrix can be obtained as an output.

### kulintang-6-inference-with-new-videos [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/maryrobynbernabe/kulintang-6-inference-with-new-videos)
This notebook helps in performing inference on a new video and provides and output video containing the predicted labels added as an overlay text.

## Web application
- The [app.py](https://github.com/bernaberobyn/Kulintang/blob/main/app.py) file contains the code for web application
- The application can be accessed [here](https://kulintangmusictranscription.streamlit.app/)


