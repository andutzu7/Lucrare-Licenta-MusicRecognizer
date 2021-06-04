import os
import shutil
import operator
import random
from tqdm import tqdm
from .dataprep import *
import numpy as np
from scipy.io import wavfile
from ModelClass.Model import Model

def get_song_label(file_path,model,dt=1.0,sr=16000,threshold=20):
    check_or_create_folder('./temp')
    # Split the song
    file_path_split = file_path.split('/')
    file_name = file_path_split[-1]
    split_audio_file(file_path,dt,sr,threshold,'./temp',file_name)

    dataset_labels={0:'Other',
                    1:'Piano'}
    prediction_stats = {
        'Other':0,
        'Piano':0
    }
    total_number = 20 # total number of samples to be tested
    indexes = random.sample(os.listdir('./temp'),total_number)
    
    for sample in tqdm(indexes):
        
        audio_file = wavfile.read(os.path.join('./temp',sample))

        X = np.empty((1, int(16000*1.0), 1),dtype=np.float32)

        X[0, ] = audio_file[1].reshape(-1, 1)
    
        confidences = model.predict(X)
        # Get prediction instead of confidence levels
        predictions = model.output_layer_activation.predictions(confidences)

        prediction = dataset_labels[predictions[0]]
        prediction_stats[prediction] += 1
    shutil.rmtree('./temp')
    return prediction_stats

def parse_song(file_path):
    pass