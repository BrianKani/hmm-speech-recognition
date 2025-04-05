# Speech Recognition System using HMM-DNN and HMM-GMM

A modular Python implementation of Hidden Markov Model (HMM)-based speech recognition with:
- **HMM-GMM** (Gaussian Mixture Model emissions)
- **HMM-DNN** (Deep Neural Network emissions using PyTorch)

## 🗂 Project Structure

speech-recognition/
├── main.py # Main entry point
├── models/ # Acoustic models
│ ├── hmm.py # Base HMM implementation
│ ├── dnn.py # DNN acoustic model (PyTorch)
│ └── gmm.py # GMM acoustic model
├── utils/ # Support utilities
│ ├── feature_extraction.py # MFCC feature extraction
│ └── data_loader.py # Dataset preprocessing
├── train.py # Model training scripts
├── evaluate.py # Accuracy metrics
└── config.py # Hyperparameters
