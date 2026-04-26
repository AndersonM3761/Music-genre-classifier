# 🎵 Music Genre Classification — Deep Learning (PyTorch)

Classifies music into 10 genres using three deep learning architectures trained from scratch on the GTZAN dataset.

## Problem Statement
Automatic music genre classification enables smarter music recommendation systems, streaming platform organization, and audio content tagging. This project compares CNN, LSTM, and hybrid CNN-LSTM architectures to identify the most effective deep learning approach for audio classification.

## Genres
Blues, Classical, Country, Disco, Hip-Hop, Jazz, Metal, Pop, Reggae, Rock

## Model Results

| Model | Test Accuracy |
|-------|--------------|
| CNN (Mel Spectrogram) | 90.52% |
| LSTM (MFCC Features) | 81.51% |
| **Hybrid CNN-LSTM** | **86.58%** |

CNN achieved the highest accuracy by effectively capturing spatial frequency patterns in mel spectrograms.

## Architecture

- **CNN** — 3 convolutional blocks on mel spectrograms for spatial feature extraction
- **LSTM** — 2-layer recurrent network on MFCC features for temporal pattern learning  
- **Hybrid CNN-LSTM** — CNN feature extractor feeding into LSTM for combined spatial + temporal modeling

## Results

![Model Comparison](results/comparison.png)
![CNN Confusion Matrix](results/CNN_confusion.png)
![LSTM Confusion Matrix](results/LSTM_confusion.png)
![Hybrid Confusion Matrix](results/Hybrid_CNN_LSTM_confusion.png)

## Dataset
GTZAN Dataset — 1000 audio clips, 10 genres, 30 seconds each  
Download: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification  
Extract to: `data/genres_original/`

## How to Run

```bash