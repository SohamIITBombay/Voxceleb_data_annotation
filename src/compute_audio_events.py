from pyannote.audio import Pipeline
import torch
import warnings
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")


device="cuda:1"
pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization-3.1', use_auth_token='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')
pipeline.to(torch.device(device))


df = pd.read_csv("../eda.csv")


def millisec(timeStr):
    spl = timeStr.split(":")
    s = (int)((int(spl[0]) * 60 * 60 + int(spl[1]) * 60 + float(spl[2]) )* 1000)
    return s


def get_speaker_overlap(s1, s2):
    i, j = 0, 0
    overlaps = []
    o_s, o_e = [], []
    # print(s1)
    while True:
        # print(i, j)
        if (i == len(s2)) or (j == len(s1)):
            break


        if s2[i][0] > s1[j][1]:
            j += 1
            continue   
        elif s2[i][0] < s1[j][0]:
            overlap_start = s1[j][0]
            o_s.append(overlap_start)
        else:
            overlap_start = s2[i][0]
            o_s.append(overlap_start)

        
        if s2[i][1] < s1[j][0]:
            i += 1
            continue
        elif s2[i][1] > s1[j][1]:
            overlap_end = s1[j][1]
            overlaps.append(overlap_end - overlap_start)
            o_e.append(overlap_end)
            j += 1
        else:
            overlap_end = s2[i][1]
            overlaps.append(overlap_end - overlap_start)
            o_e.append(overlap_end)
            i += 1

            
    if sum(overlaps) < 0:
        print("spk1: ", s1)
        print("spk2: ", s2)
        print("overlap_starts: ", o_s)
        print("Overlap_ends: ", o_e)
        print("overlaps: ", overlaps)
    return overlaps

def get_speech_activity(wav_file, duration):
    segmentations_ans = pipeline(wav_file)
    lines = str(segmentations_ans).split('\n')
    speech_duration = 0
    durs = {}
    spk1, spk2 = [], []
    for line in lines:
        starts = millisec(line.split()[1])
        ends = millisec(line.split()[3][:-1])
        speaker = line.split()[5]

        if speaker == "SPEAKER_01":
            spk1.append((starts, ends))
        else:
            spk2.append((starts, ends))

        if speaker not in durs.keys():
            durs[speaker] = 0
        dur = ends - starts
        # print(line, dur)
        durs[speaker] = durs[speaker] + dur

        speech_duration += dur

    num_speakers_present = len(durs)

    if num_speakers_present > 1:
        speaker_overlaps = get_speaker_overlap(spk1, spk2)
        total_speaker_overlap = sum(speaker_overlaps)/1000
    else:
        total_speaker_overlap = 0

    speaker_overlap_ratio = round(total_speaker_overlap/duration, 2)
    silent_duration = round(max(0, duration*1000 - speech_duration)/1000, 2)
    # print(duration*1000, speech_duration)

    return num_speakers_present, total_speaker_overlap, speaker_overlap_ratio, silent_duration


num_speakers_list, silent_durations = [], []
total_speaker_overlaps, speaker_overlap_ratios = [], []

for i, row in tqdm(df.iterrows(), total=len(df)):
    # print(row['file_name'])
    num_speakers, total_speaker_overlap, speaker_overlap_ratio, silent_duration = get_speech_activity(row['file_name'], row['duration'])
    num_speakers_list.append(num_speakers)
    silent_durations.append(silent_duration)
    total_speaker_overlaps.append(total_speaker_overlap)
    speaker_overlap_ratios.append(speaker_overlap_ratio)
    # print(total_speaker_overlap, row['duration'])
    # print(silent_duration)

df['num_speakers'] = num_speakers_list
df['total_speaker_overlap'] = total_speaker_overlaps
df['speaker_overlap_ratio'] = speaker_overlap_ratios
df['silent_duration'] = silent_durations


df.to_csv("../eda.csv", index=None)


print(df['silent_duration'].describe())
df['silent_duration'].hist(bins=50)
plt.title("Silent_Duration")
plt.show()


print(df['speaker_overlap_ratio'].describe())
df['speaker_overlap_ratio'].hist(bins=50)
plt.title("speaker_overlap_ratio")
plt.show()


print(df['num_speakers'].describe())
df['num_speakers'].hist(bins=50)
plt.title("num_speakers")
plt.show()



