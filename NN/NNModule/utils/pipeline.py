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

    check_or_create_folder('./tempdownload')
    options = {
        'format': 'bestaudio/best',
        'outtmpl': './tempdownload/%(title)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192', }]
    }
    with youtube_dl.YoutubeDL(options) as ydl:
        ydl.download([link])


def get_song_label(file_path, model, guesses = 20, dt=1.0, sr=16000, threshold=20):

    check_or_create_folder('./temp')
    # Split the song
    file_path_split = file_path.split('/')
    file_name = file_path_split[-1]
    split_audio_file(file_path, dt, sr, threshold, './temp', file_name)

    dataset_labels = {0: 'Other',
                      1: 'Piano'}
    prediction_stats = {
        'Other': 0,
        'Piano': 0
    }
    indexes = random.sample(os.listdir('./temp'), guesses)

    for sample in tqdm(indexes):

        audio_file = wavfile.read(os.path.join('./temp', sample))

        X = np.empty((1, int(16000*1.0), 1), dtype=np.float32)

        X[0, ] = audio_file[1].reshape(-1, 1)

        confidences = model.predict(X)
        # Get prediction instead of confidence levels
        predictions = model.output_layer_activation.predictions(confidences)

        prediction = dataset_labels[predictions[0]]
        prediction_stats[prediction] += 1
    shutil.rmtree('./temp')
    shutil.rmtree('./tempdownload')
    return prediction_stats