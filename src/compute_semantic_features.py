import pandas as pd
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import whisper
from transformers import pipeline


df = pd.read_csv("../eda.csv")


model_size = "large-v3"
device_idx = 2
model = whisper.load_model(model_size, device=torch.device("cuda:" + str(device_idx)), \
                                    download_root="/raid/soham.pendurkar/workspace")


def get_transcription(wav_file):
    results = model.transcribe(wav_file, beam_size=5)

    overall_avg_log_prob = 0
    for segment in results['segments']:
        seg_avg_log_prob = segment['avg_logprob']
        overall_avg_log_prob += seg_avg_log_prob

    overall_avg_log_prob = (overall_avg_log_prob / len(results['segments']))

    return results['text'], overall_avg_log_prob, results['language']



transcriptions, avg_log_prob_list = [], []
languages, language_prob = [], []

for i, row in tqdm(df.iterrows(), total=len(df)):
    # print(row['duration'])
    transcript, avg_log_probs, language = get_transcription(row['file_name'])

    transcriptions.append(transcript)
    avg_log_prob_list.append(avg_log_probs)
    languages.append(language)

df['transcription'] = transcriptions
df['avg_log_probabilities'] = avg_log_prob_list
df['language'] = languages


df.to_csv("../eda.csv", index=None)


print(df['avg_log_probabilities'].describe())
df['avg_log_probabilities'].hist(bins=50)
plt.title("avg_log_probabilities")
plt.show()


print(df['language'].describe())
df['language'].hist(bins=50)
plt.title("language")
plt.show()


def count_words(text):
    return len(text.split(" "))

df['transcription_length'] = df['transcription'].apply(count_words)
df.to_csv("../eda.csv", index=None)


print(df['transcription_length'].describe())
df['transcription_length'].hist(bins=50)
plt.title("transcription_length")
plt.show()



classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=5, device=torch.device("cuda:3"))


def get_emotions(sentence, threshold):
    model_outs = classifier(sentence)
    detected = []
    for entry in model_outs[0]:
        if entry['score'] > threshold:
            detected.append((entry['label'], entry['score']))

    if len(detected) == 0:
        return "NA"

    return detected


emotions = []
for i, row in tqdm(df.iterrows(), total=len(df)):
    emotions.append(get_emotions(row['transcription'], threshold=0.5))


df['emotions'] = emotions
df.to_csv("../eda.csv", index=None)