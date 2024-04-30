import glob
from pydub import AudioSegment
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import librosa
import glob, os, shutil
import librosa
import numpy as np
import matplotlib.pyplot as plt
import subprocess


wav_dir = "../data/wav"
wav_files = glob.glob(wav_dir + "/**/*.wav", recursive=True)
print("Number of wavs: ", len(wav_files))

## Checking if there are audios of any other type
non_wavs = 0
for root_dir, cur_dir, files in os.walk(wav_dir):
    non_wavs += len(files)
print('Non_wavs: ', non_wavs)


def get_rms(wav_file):
    y, sr = librosa.load(wav_file)
    rms = librosa.feature.rms(y=y)
    return rms.mean()


sample_rates, durations = [], []
num_channels, bit_depths = [], []
rmss = []

for wav in tqdm(wav_files):
    
    audio_seg = AudioSegment.from_wav(wav)

    sample_rates.append(audio_seg.frame_rate)
    durations.append(audio_seg.duration_seconds)
    num_channels.append(audio_seg.channels)
    bit_depths.append(audio_seg.sample_width)
    try:
        rmss.append(get_rms(wav))
    except:
        rmss.append("RMS Could not be computed")


df = pd.DataFrame(columns = ['file_name', 'sample_rate', 'duration', 'num_channels', 'bit_depths'])
df['file_name'] = wav_files
df['sample_rate'] = sample_rates
df['duration'] = durations
df['num_channels'] = num_channels
df['bit_depths'] = bit_depths
df['rms'] = rmss

print(df.head())



## Check Sample Rates
print(df['sample_rate'].describe())
df['sample_rate'].hist(figsize=(4, 2))
plt.title("sample_rates")
plt.show()



## Check durations
print(df['duration'].describe())
df['duration'].hist(bins=50, figsize=(4, 2))
plt.title("duration")
plt.show()


## Check num_channels
print(df['num_channels'].describe())
df['num_channels'].hist(figsize=(4, 2))
plt.title("num_channels")
plt.show()


## Check bit_depths
print(df['bit_depths'].describe())
df['bit_depths'].hist(figsize=(4, 2))
plt.title("bit_depths")
plt.show()


df.to_csv("../eda.csv", index=None)




# Creating directory structure as required by denoiser package


noisy_wav_dir = "../data/noisy_wavs"
os.makedirs(noisy_wav_dir, exist_ok=True)
cleaned_wav_dir = "../data/cleaned_wavs"
os.makedirs(cleaned_wav_dir, exist_ok=True)

wav_files = glob.glob("../data/wav/**/*.wav", recursive=True)
print(len(wav_files))
for wav_file in tqdm(wav_files):

    dir_info = "_".join(wav_file.split("/")[3:])
    new_name = os.path.join(noisy_wav_dir, dir_info)
    shutil.copy(wav_file, new_name)


subprocess.run("bash run_denoising.sh")
# !bash run_denoising.sh



noisy_wavs = glob.glob(noisy_wav_dir + "/*.wav").sort(key=lambda x: str(x[0]))
cleaned_wavs = glob.glob(cleaned_wav_dir + "/*.wav").sort(key=lambda x: str(x[0]))

snrs = []
for i in range(len(noisy_wavs)):
    noisy_signal = librosa.load(noisy_wavs[i])[0]
    cleaned_signal = librosa.load(cleaned_wavs[i])[0]

    noisy_signal_energy = np.sum(noisy_signal ** 2)
    cleaned_signal_energy = np.sum(cleaned_signal ** 2)

    noise_energy = np.array(noisy_signal_energy - cleaned_signal_energy)
    snr = 10 * np.log10(noisy_signal_energy/noise_energy)

    snrs.append(snr)

original_names = []
for file_name in noisy_wavs:
    original_file_name = "../data/wav/" + "/".join(file_name.split("_"))
    original_names.append(original_file_name)

temp_df = pd.DataFrame(columns=['file_name', 'snr'])
temp_df['file_name_d'] = original_names
temp_df['snr'] = snrs



df = pd.read_csv("../eda.csv")
df = pd.merge(df, temp_df, on="file_name", how="inner")
print(df.shape)