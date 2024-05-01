#!/bin/bash

echo "Computing Audio Features"
source activate denoiser
python3 compute_audio_features.py


echo "Done Computing Audio features"
echo ""
echo ""

echo "Computing Semantic Features"

source activate data_annotation
python3 compute_semantic_features.py

echo "Done Computing Semantic features"
echo 
echo

echo "Computing audio events"
source activate data_annotation_2
python3 compute_audio_events.py

echo
echo
echo "Done"