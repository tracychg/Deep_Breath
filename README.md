# Project - Deep Breath

# Project aims

The objective of this project was to build a model capable of identifying breath abnormalities from audio recordings and predicting the most likely lung disease type accordingly. It was imagined as an aid and support to doctors in performing pulmonary auscultation, the accuracy of which can vary greatly from doctor to doctor according to clinical experience.

# Dataset and background

The dataset we used was obtained from Kaggle (https://www.kaggle.com/vbookshelf/respiratory-sound-database/) and consists of breath recordings from 126 patients collected by research teams in Greece and Portugal. In total, we processed 960 audio files and more than 6000 respiratory cycles.

# Libraries

Pandas, matplotlib, seaborn, scikit-learn, and tensorflow keras, the use of which was essential to our project in enabling us to convert audio to image through spectrograms and mel-spectrograms, while librosa was also used for exploratory data analysis.

# Model and performance

Our model is a CNN model which performs multi-class classification, determining whether a patient is healthy or ill with one of five respiratory diseases with an average precision of 86% and recall of 85%.

A particular aim we had in mind when working on this project, was to maximise recall in identifying and predicting COPD, a disease which accounts for about 5% of total UK deaths, and which our model achieves with a 96% average.

Notebooks can be found in the Notebooks folder in the project’s Github repository.

# Data engineering

Our model was deployed on Streamlit Cloud and can be accessed at the following address: https://share.streamlit.io/tracychg/deep_breath
