\# Speaker-Independent Spoken Digit Recognition



A machine learning project that recognizes spoken digits (0–9) from audio recordings using \*\*Mel-Frequency Cepstral Coefficients (MFCCs), Gaussian Mixture Models (GMMs), and Hidden Markov Models (HMMs)\*\*.



The project focuses on speaker-independent recognition: evaluating whether models trained on some speakers can recognize digits spoken by previously unseen speakers.



\## Project Overview



This project implements and compares two probabilistic approaches for spoken digit recognition using a dataset of \*\*3,000 audio recordings across 10 digit classes\*\*.



The system extracts acoustic features from each recording, trains a separate model for each digit, and predicts the digit corresponding to an unseen audio sample.



\### Key Results



| Model | Recognition Accuracy |

|---|---:|

| Gaussian Mixture Model (GMM) | \*\*85.8%\*\* |

| Hidden Markov Model (HMM) | \*\*80.0%\*\* |



\*Results reported for unseen-speaker evaluation.\*



\## Features



\- \*\*Audio feature extraction:\*\* Extracts 20 MFCC coefficients, along with Delta and Delta-Delta features, to represent spectral characteristics and their temporal changes.

\- \*\*Custom K-Means implementation:\*\* Builds a 48-centroid acoustic codebook for converting continuous audio features into discrete observation sequences.

\- \*\*GMM from scratch:\*\* Implements Gaussian mixture components, expectation-maximization (EM), covariance estimation, and log-likelihood scoring.

\- \*\*HMM from scratch:\*\* Implements a discrete Hidden Markov Model with forward-backward calculations, parameter estimation, and sequence likelihood scoring.

\- \*\*Speaker-independent evaluation:\*\* Separates training, validation, and test data by speaker rather than randomly splitting recordings.

\- \*\*Performance analysis:\*\* Reports accuracy, confusion matrices, and per-digit recognition performance.



\## Methodology



1\. \*\*Load audio:\*\* Read the WAV recordings and extract their digit and speaker labels from the filenames.

2\. \*\*Extract features:\*\* Calculate MFCC, Delta, and Delta-Delta features using Librosa.

3\. \*\*Normalize features:\*\* Standardize the extracted features using training-set statistics.

4\. \*\*Build the codebook:\*\* Train K-Means with 48 centroids to represent acoustic observations as discrete symbols.

5\. \*\*Train digit classifiers:\*\* Train one GMM and one discrete HMM for each digit class.

6\. \*\*Evaluate generalization:\*\* Evaluate both models on validation and test speakers that are excluded from training.

7\. \*\*Compare performance:\*\* Calculate accuracy, confusion matrices, and per-digit results.



\## Dataset and Evaluation



The project contains 3,000 WAV recordings distributed across 10 digit classes and 6 speakers.



| Dataset split | Speakers | Recordings |

|---|---|---:|

| Training | Speakers 1–4 | 2,000 |

| Validation | Speaker 5 | 500 |

| Testing | Speaker 6 | 500 |

| \*\*Total\*\* | \*\*6 speakers\*\* | \*\*3,000\*\* |



The speaker-based split helps evaluate generalization to voices that the models did not encounter during training.



\## Project Structure



```text

Solutions\_DA24B029/

├── main.py          # Training and evaluation pipeline

├── features.py      # Audio loading, MFCC features, and K-Means

├── gmm.py           # Custom Gaussian Mixture Model

├── hmm.py           # Custom discrete Hidden Markov Model

├── recordings/      # WAV audio dataset

├── Report.pdf       # Project report

├── Details.txt      # Project identification details

└── README.md        # Project documentation

```



\## Technologies Used



\- \*\*Language:\*\* Python

\- \*\*Audio processing:\*\* Librosa, SoundFile

\- \*\*Numerical computation:\*\* NumPy

\- \*\*Visualization:\*\* Matplotlib

\- \*\*Machine learning:\*\* Custom implementations of K-Means, GMM, and HMM



\## Installation and Setup



\### Prerequisites



\- Python 3.10 or later

\- pip

\- The audio recordings included in the `recordings/` directory



\### 1. Clone the repository



```bash

git clone https://github.com/VedaVachan/AI\_Project.git

cd AI\_Project

```



\### 2. Create and activate a virtual environment



\*\*Windows PowerShell\*\*



```powershell

python -m venv .venv

.\\.venv\\Scripts\\Activate.ps1

```



\*\*Linux / macOS\*\*



```bash

python3 -m venv .venv

source .venv/bin/activate

```



\### 3. Install dependencies



```bash

python -m pip install --upgrade pip

python -m pip install numpy librosa matplotlib soundfile

```



\### 4. Run the project



Ensure the `recordings/` directory is present in the project root, then run:



```bash

python main.py

```



The program loads the recordings, trains the models, evaluates validation and test performance, and displays confusion matrices and per-digit accuracy statistics.



\## Model Comparison



\### Gaussian Mixture Model (GMM)



The GMM models the distribution of acoustic feature vectors for each digit. The implementation uses the expectation-maximization algorithm to estimate mixture weights, means, and diagonal covariance matrices. Classification is performed using log-likelihood scores.



\### Hidden Markov Model (HMM)



The discrete HMM models temporal patterns in audio using hidden states and discrete observations derived from the K-Means codebook. The implementation uses forward-backward computations to estimate model parameters and sequence likelihoods.



\### Results



The reported evaluation shows that the GMM achieved 85.8% accuracy, compared with 80.0% for the HMM. On this dataset and evaluation setup, the GMM performed better at distinguishing spoken digits from unseen speakers.



\## Learning Outcomes



\- Audio signal processing and acoustic feature engineering

\- Probabilistic modelling for speech recognition

\- Implementing EM and forward-backward algorithms

\- Comparing generative models using held-out speakers

\- Evaluating classifiers using confusion matrices and per-class accuracy



\## Author



\*\*Vangala VedaVachan Reddy\*\*



B.Tech — Data Analytics and Artificial Intelligence  

Indian Institute of Technology Madras





