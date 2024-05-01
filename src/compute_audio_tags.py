import torch
import warnings
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import whisper_at as whisper
warnings.filterwarnings("ignore")

df = pd.read_csv("../eda.csv")

device_idx = 3
audio_tagging_time_resolution = 10
top_k=1
model = whisper.load_model("large-v2", device=torch.device("cuda:" + str(device_idx)), download_root="/raid/soham.pendurkar/workspace/whisper-at")



def get_audio_tags(wav_file, duration, audio_tagging_time_resolution=10.0, top_k=2):
    # audio_tagging_time_resolution = min(duration, audio_tagging_time_resolution)
    result = model.transcribe(wav_file, at_time_res=audio_tagging_time_resolution)

    audio_tag_result = whisper.parse_at_label(result, language='follow_asr', top_k=top_k, p_threshold=-1, include_class_list=list(range(527)))
    # print(audio_tag_result)
    atags = []
    for audio_tags in audio_tag_result:
        for tag in audio_tags['audio tags']:
            atags.append(tag[0])
    # print(atags)
    return atags


tags = []
for i, row in tqdm(df.iterrows(), total=len(df)):
    tags.append(get_audio_tags(row['file_name'], row['duration']))

df['audio_events'] = tags
df.to_csv("../eda.csv", index=None)