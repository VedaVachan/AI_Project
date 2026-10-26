
# Speaker-Independent Spoken Digit Recognition

A machine learning project for recognizing spoken digits using MFCC features, Gaussian Mixture Models (GMM), and Hidden Markov Models (HMM).

## Project Overview

This project recognizes spoken digits (0–9) from audio recordings and evaluates generalization to previously unseen speakers.

## Key Results

| Model | Accuracy |
|---|---:|
| GMM | **85.8%** |
| HMM | **80.0%** |

## Features

- MFCC, Delta, and Delta-Delta feature extraction
- K-Means acoustic codebook with 48 centroids
- GMM implementation using expectation-maximization
- HMM implementation using forward-backward algorithms
- Speaker-independent model evaluation
- Accuracy comparison and performance analysis

## Dataset

- 3,000 audio recordings
- 10 digit classes (0–9)
- 6 speakers
- Training, validation, and testing splits based on speakers

## Technologies

- Python
- Librosa
- NumPy
- SoundFile
- Matplotlib

## Project Structure

```text
Solutions_DA24B029/
├── main.py
├── features.py
├── gmm.py
├── hmm.py
├── recordings/
├── Report.pdf
├── Details.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/VedaVachan/AI_Project.git
cd AI_Project
```

Install dependencies:

```bash
python -m pip install numpy librosa matplotlib soundfile
```

## Run the Project

Ensure the recordings directory and Python files are in the expected locations, then run:

```bash
python main.py
```

## Results

The reported evaluation achieved 85.8% accuracy with GMM and 80.0% with HMM. GMM performed better under the evaluated dataset and speaker-independent testing setup.

## Author

**Vangala VedaVachan Reddy**

B.Tech — Data Analytics and Artificial Intelligence  
Indian Institute of Technology Madras
