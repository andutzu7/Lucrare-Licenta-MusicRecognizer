import matplotlib.pyplot as plt
from scipy.io import wavfile
import argparse
import os
from glob import glob
import numpy as np
import pandas as pd
from librosa.core import resample, to_mono
from tqdm import tqdm
import wavio


def split_by_sound_envelope(y, rate, threshold):
    """
    The split_by_sound_envelope function takes a wav file (represented as a numpy array), converts 
    it to a pandas series (applying the absolute value function over its elements) and extracts the 
    audio mask(the values above a given threshold) by computing the rolling window mean over the 
    array by a window of the sample rate divided by 20 (arbitrary rate) and comparing it to the 
    threshold. The mask represents the values of the sound that don't contain noise/unnecessary 
    sound values such as audience clapping, silence, crowd noises etc.

    Split by soud envelope arguments :
        :param y(np.array) : np.array containing the wav file audio data
        :param rate(float) : the wav's sample rate
        :param threshold(np.array) : np.array containing the audio comparation threshold 

    Returns:
    :mask(np.array): Clean audio file mask
    :y_mean(np.array): Rolling window mean of the audio

"""
    # Initializing the mask array
    mask = []
    # Converting the audio to a pandas series
    y = pd.Series(y).apply(np.abs)
    # Computing the rolling window mean
    y_mean = y.rolling(window=int(rate/20),
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


def save_sample(sample, rate, target_dir, file_name, index):
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
        target_dir.split('.')[0], f'{file_name}_{index}')
    # Check if the file doesen't exist already
    if os.path.exists(destination_path):
        return
    # Write the file at the destination path
    wavfile.write(destination_path, rate, sample)


def check_dir(path):
    """
    The check_dir function checks if a folder at a given path exists and if not creates one.

    Downsample mono arguments :
        :param path(string) : the path of the folder


    Returns:
"""
    if os.path.exists(path) is False:
        os.mkdir(path)


def split_wavs(args):
    src_root = args.src_root
    dst_root = args.dst_root
    dt = args.delta_time

    wav_paths = glob('{}/**'.format(src_root), recursive=True)
    wav_paths = [x for x in wav_paths if '.wav' in x]
    dirs = os.listdir(src_root)
    check_dir(dst_root)
    classes = os.listdir(src_root)
    for _cls in classes:
        target_dir = os.path.join(dst_root, _cls)
        check_dir(target_dir)
        src_dir = os.path.join(src_root, _cls)
        for fn in tqdm(os.listdir(src_dir)):
            src_fn = os.path.join(src_dir, fn)
            rate, wav = downsample_mono(src_fn, args.sr)
            mask, y_mean = split_by_sound_envelope(
                wav, rate, threshold=args.threshold)
            wav = wav[mask]
            delta_sample = int(dt*rate)

            # cleaned audio is less than a single sample
            # pad with zeros to delta_sample size
            if wav.shape[0] < delta_sample:
                sample = np.zeros(shape=(delta_sample,), dtype=np.int16)
                sample[:wav.shape[0]] = wav
                save_sample(sample, rate, target_dir, fn, 0)
            # step through audio and save every delta_sample
            # discard the ending audio if it is too short
            else:
                trunc = wav.shape[0] % delta_sample
                for cnt, i in enumerate(np.arange(0, wav.shape[0]-trunc, delta_sample)):
                    start = int(i)
                    stop = int(i + delta_sample)
                    sample = wav[start:stop]
                    save_sample(sample, rate, target_dir, fn, cnt)


def test_threshold(args):
    src_root = args.src_root
    wav_paths = glob('{}/**'.format(src_root), recursive=True)
    wav_path = [x for x in wav_paths if args.fn in x]
    if len(wav_path) != 1:
        print('audio file not found for sub-string: {}'.format(args.fn))
        return
    rate, wav = downsample_mono(wav_path[0], args.sr)
    mask, env = split_by_sound_envelope(wav, rate, threshold=args.threshold)
    plt.style.use('ggplot')
    plt.title('Signal split_by_sound_envelope, Threshold = {}'.format(
        str(args.threshold)))
    plt.plot(wav[np.logical_not(mask)], color='r', label='remove')
    plt.plot(wav[mask], color='c', label='keep')
    plt.plot(env, color='m', label='split_by_sound_envelope')
    plt.grid(False)
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cleaning audio data')
    parser.add_argument('--src_root', type=str, default='./some test data',
                        help='directory of audio files in total duration')
    parser.add_argument('--dst_root', type=str, default='./newe',
                        help='directory to put audio files split by delta_time')
    parser.add_argument('--delta_time', '-dt', type=float, default=1.0,
                        help='time in seconds to sample audio')
    parser.add_argument('--sr', type=int, default=16000,
                        help='rate to downsample audio')
    parser.add_argument('--fn', type=str, default='3a3d0279',
                        help='file to plot over time to check magnitude')
    parser.add_argument('--threshold', type=str, default=20,
                        help='threshold magnitude for np.int16 dtype')
    args, _ = parser.parse_known_args()

    # test_threshold(args)
    split_wavs(args)
