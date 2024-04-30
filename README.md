# Voxceleb Dataset Labelling and Annotation

## Overview
This repository contains tools and guidelines for labelling/annotating a speech dataset - [voxceleb1](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html). This would involve comprehensively extracting information about the dataset such as Transcriptions, Audio Metadata, Speaker Characteristics, etc. <br><br>
The repository also contains the details on the different tools/libraries used and why they were used.


## Purpose
The main aim of this repository is to get a speech dataset labelled/annotated so as to enable custom filtration, quality control. Different filtration criteria based on the information extracted could then be used to train different speech models and study their performances.


## About the dataset: [Paper](https://www.robots.ox.ac.uk/~vgg/publications/2017/Nagrani17/nagrani17.pdf)

- Generated with an aim for enabling Speaker Identification and Verification in 'real world' conditions
- Automatic scalable pipeline consisting of Youtube crawling, Face tracking-based segmentation etc.
- Contains over 100,000 utterances for 1,251 celebrities, extracted from videos uploaded to YouTube

## Data Download

For this repo, only the test split of VoxCeleb1 **Speaker Verification** data will be used.<br>

- For downloading wavs, you need to register on [this](https://cn01.mmai.io/keyreq/voxceleb) link.
- After registration, an authenticated URL will be sent to the email Id for downloading.
- Download the ```test``` split data.
- Place the downloaded data into the ```data``` directory
- Unzip the folder


## Label/Annotation Details
1. Transcription
2. Speaker Identity
3. Number of Speakers in each Audio clip
4. Emotions contained in the utterance
5. Audio events (e.g. laughter, crowd cheers, music) along with their timestamps
6. Basic file metadata (sample rate, channels, duration)
7. Audio quality / naturalness / noise level (we will want to use this to filter based on quality later on).
8. Total Speech overlap in each audio clip (Total duration where more than one speaker is talking at the same time)
9. Language of each utterance
10. Incorrect chopping

## Labelling/Annotation categories

### Audio Features
- Sample rate
- Number of Channels
- Duration
- Bit Depth
- Noise level
- RMS (Correlated to loudness)

### Semantic Features
- Transcriptions
- Transcription length (Number of words in the transcription)
- Emotions
- Language
- Incorrect chopping (Abrupt end of sentence) 

### Audio Events
- Speaker Identity
- Number of speakers in each utterance
- Silence duration (No Speech Activity)
- Speech overlap
- Laughter, Crowd cheering, music



