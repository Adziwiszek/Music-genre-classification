# Classification of music genre based on the audio

# Datasets

You can download GTZAN dataset from [here](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification?resource=download-directory), unzip it and move the `Data` directory to the root directory of this repo.

Before you run any code make sure to run `prepare_dataset.py` script for ease of use. It removes one song whose .wav file was corrupted.

## Feature engineering

To create datasets with DTW features run `python -m feature_engineering.create_dtw_distances`

To create datasets with time series features run `python -m feature_engineering.create_ts_data`