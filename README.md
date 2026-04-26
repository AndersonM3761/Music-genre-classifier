# 🎵 Music Genre Classification — Deep Learning (PyTorch)

Classifies music into 10 genres using CNN, LSTM, 
and Hybrid CNN-LSTM trained from scratch.

## Genres
Blues, Classical, Country, Disco, Hip-Hop, 
Jazz, Metal, Pop, Reggae, Rock

## How to Run

Step 1 - Verify setup:
python step1_setup_check.py

Step 2 - Download GTZAN dataset from Kaggle:
https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
Extract to: data/genres_original/

Step 3 - Preprocess audio:
python step2_preprocess.py

Step 4 - Train models:
python step3_train.py

Step 5 - Run demo app:
streamlit run step4_demo_app.py

## Models
- CNN on Mel Spectrograms
- LSTM on MFCC features  
- Hybrid CNN-LSTM (best accuracy)

## Tech Stack
PyTorch, Librosa, Streamlit, Scikit-learn

## Dataset
GTZAN: 1000 songs, 10 genres, 30 seconds each