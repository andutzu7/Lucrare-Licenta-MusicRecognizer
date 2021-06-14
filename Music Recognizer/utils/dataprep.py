from re import split
from scipy.io import wavfile
import os
import numpy as np
import pandas as pd
from librosa.core import resample, to_mono
import wavio


def split_by_sound_envelope(y, sr, threshold):
    """
    The split_by_sound_envelope function takes a wav file (represented as a numpy array), converts 
    it to a pandas series (applying the absolute value function over its elements) and extracts the 
    audio mask(the values above a given threshold) by computing the rolling window mean over the 
    array by a window of the sample rate divided by 20 (arbitrary rate) and comparing it to the 
    threshold. The mask represents the result of the comparasing between the value and the threshold (boolen)
    and marks the positions of the values of the sound file that don't contain noise/unnecessary 
    values such as audience clapping, silence, crowd noises etc.

    Split by soud envelope arguments :
        :param y(np.array) : np.array containing the wav file audio data
        :param sr(int) : the wav's sample rate
        :param threshold(int) : threshold magnitude value

    Returns:
    :mask(np.array): Clean audio file mask
    :y_mean(np.array): Rolling window mean of the audio

    Source:
        *https://www.youtube.com/user/seth8141/featured
        *https://github.com/seth814/Audio-Classification
        *https://kapre.readthedocs.io/en/latest/composed.html
"""
    # Initializing the mask array
    mask = []
    # Converting the audio to a pandas series
    y = pd.Series(y).apply(np.abs)
    # Computing the rolling window mean
    y_mean = y.rolling(window=int(sr/20),
                       min_periods=1,
                       center=True).max()
    # Comparing each value of the mean to the threshold
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask, y_mean


def downsample_mono(path, sr):
    """
    The downsample_mono function reads a file at a given path as a numpy array, mono-sampling the 
    its values and resampling it to the given sample rate.

    Downsample mono arguments :
        :param path(string) : the path of the file
        :param sr(int) : the wav's sample rate


    Returns:
    :param sr(int) : the wav's sample rate
    :param wav(np.array) : the mono-sampled converted wav file data
"""
    # Read the file
    file = wavio.read(path)
    # Convert the data to float
    wav = file.data.astype(np.float32)
    # Cheking if the file is mono channel or stereo and handling the audio accordingly
    channels = wav.shape[1]
    if channels == 2:
        wav = to_mono(wav.T)
    elif channels == 1:
        wav = to_mono(wav.reshape(-1))
    wav = to_mono(wav.reshape(-1))
    # Resampling the audio rate
    wav = resample(wav, file.rate, sr)
    # Converting the wav file values back to int
    wav = wav.astype(np.int16)
    return sr, wav


def save_sample(sample, sr, target_dir, file_name, index):
    """
    The save_sample function writes a given sample to the disk.

    Downsample mono arguments :
        :param sample(np.array) : wav audio file sample.
        :param sr(int) : the wav's sample rate
        :target_dir(string) : output directory
        :file_name(string) : output file name
        :index(int) : file index


    Returns:
"""
    # Get the file name
    file_name = file_name.split('.wav')[0]
    # Format the destination path
    destination_path = os.path.join(
        target_dir, f'{file_name}_{index}.wav')
    # Check if the file doesen't exist already
    if os.path.exists(destination_path):
        return
    # Write the file at the destination path
    wavfile.write(destination_path, sr, sample)


def check_or_create_folder(path):
    """
    The check_or_create_folder function checks if a folder at a given path exists and if not creates one.

    check_or_create_folder arguments :
        :param path(string) : the path of the folder


    Returns:
"""
    if os.path.exists(path) is False:
        os.mkdir(path)


def split_audio_files(audio_folder_root, destination_folder_root, dt=1.0, sr=16000, threshold=20):
    """
    The split_audio_files function performs the audio file splitting operation recursively throuought the
    given audio_folder_root. 
    The function gets the file names from the given folder (by class), then for each file found it performs
    the downsampling and splitting by envelope, cleaning the wav file with the mask.
    It then saves the cleaned audio samples in chunks of lenght dt.

    Split audio files arguments :
        :param audio_folder_root(string) : path to the folder data (The data have to be separated into classes).
        :param destination_folder_root(string) : path to the destination folder.
        :param dt(float) : delta time, the output file lenght.
        :param sr(int) : the wav's sample rate
        :param threshold(int) : threshold magnitude value

    Returns:

"""
    # Check if the destination folder exists and create it if it doesen't
    check_or_create_folder(destination_folder_root)
    # Get the classes names
    classes = os.listdir(audio_folder_root)
    # For each given class
    for category in classes:
        # Create the destination folder path
        destination_folder = os.path.join(destination_folder_root, category)
        # Check if the destination folder exists and create it if it doesen't
        check_or_create_folder(destination_folder)
        # Get the source directory path
        source_directory = os.path.join(audio_folder_root, category)
        # Iterate through all the files and for each file found apply the transformation
        for file in os.listdir(source_directory):
            # Generate the file name
            file_name = os.path.join(source_directory, file)
            split_audio_file(file_name, dt, sr, threshold,
                             destination_folder, file)


def split_audio_file(file_name, dt, sr, threshold, destination_folder, file):
    """
    Auxiliary function for split audio files.

    Split audio files arguments :
        :param file_name(string) : path to the file
        :param sr(int) : the wav's sample rate
        :param destination_folder(string) : output directory
        :param file(string) : isolated file name
    Returns:

"""
    # Downsample the file to mono channels
    rate, wav = downsample_mono(file_name, sr)
    # Compute the sound envelope mask
    mask, y_mean = split_by_sound_envelope(
        wav, rate, threshold)
    # Filter the wav file by only keeping the relevant audio values
    wav = wav[mask]
    # Compute the desired output shape
    delta_sample = int(dt*rate)
    # Compute the number of the splits of the file
    sample_difference = wav.shape[0] % delta_sample
    # Compute the range end by extracting the sample rate difference
    range_end = wav.shape[0]-sample_difference
    # Iterate through the wav file and extract all of the subsamples in regards to the delta_sample
    for count, step in enumerate(np.arange(start=0, stop=range_end, step=delta_sample)):
        # The start value is the current value from the range
        start = int(step)
        # The stop is proportional with the delta sample's shape
        stop = int(step + delta_sample)
        # Clustering the file
        sample = wav[start:stop]
        # Saving the new sample
        save_sample(sample, rate, destination_folder, file, count)


if __name__ == '__main__':
    src_root = './some test data'
    dst_root = './newe'
    dt = 1.0
    sr = 16000
    threshold = 20
    split_audio_files(src_root, dst_root, dt, sr, threshold)
