import os
import shutil
import operator
import random
from tqdm import tqdm
from .dataprep import *
import numpy as np
from scipy.io import wavfile
from ModelClass.Model import Model
import youtube_dl


def download_song_from_youtube(link):
    """
    Download_song_from_youtube downloads the a file from youtube using the youtube-dl API

    Args: link(string): The link to the video.

    """
    # Create the temp folder
    check_or_create_folder('./tempdownload')
    # Set up the download options
    options = {
        'format': 'bestaudio/best',
        'outtmpl': './tempdownload/%(title)s.%(ext)s',
        'noplaylist': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192', }]
    }
    # Download the song
    with youtube_dl.YoutubeDL(options) as ydl:
        ydl.download([link])


def get_song_label(file_path, model, guesses=20, dt=1.0, sr=16000, threshold=20):
    """
    get_song_label takes the path to the song, splits it then evaluates {guesses} times,
    storing the result of each prediction, returning the one with the maximal number guesses as the label.

    Args: file_path(string): The path to the file.
          model(ModelClass): The model which will perform the inference
          guesses(int): The number of files to be tested.
          dt(int): File duration time
          sr(int): Audio files sample rate
          threshold(int): The threshold magnitude value.
    """

    # Create the temp folder
    check_or_create_folder('./temp')
    # Split the song
    file_path_split = file_path.split('/')
    file_name = file_path_split[-1]
    split_audio_file(file_path, dt, sr, threshold, './temp', file_name)

    # Initialize the labels and the prediction stats
    dataset_labels = {0: 'Other',
                      1: 'Piano'}
    prediction_stats = {
        'Other': 0,
        'Piano': 0
    }
    # Choosing a random batch of indexes of size guesses
    indexes = random.sample(os.listdir('./temp'), guesses)

    # For each sample in the choosen indexes
    for sample in tqdm(indexes):

        # Read an audio file
        audio_file = wavfile.read(os.path.join('./temp', sample))

        # Create an empty data array
        X = np.empty((1, int(16000*1.0), 1), dtype=np.float32)

        # Assign the data
        X[0,] = audio_file[1].reshape(-1, 1)

        # Perform the inferrence
        confidences = model.predict(X)
        # Get prediction instead of confidence levels
        predictions = model.output_layer_activation.predictions(confidences)

        # Add the prediction to the stats
        prediction = dataset_labels[predictions[0]]
        prediction_stats[prediction] += 1
    # Remove temp folders
    shutil.rmtree('./temp')
    shutil.rmtree('./tempdownload')
    return prediction_stats
