import torch
import os

class Config:
    # General
    RANDOM_SEED = 42
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CHECKPOINT_DIR = 'checkpoints'
    
    # Data
    SAMPLE_RATE = 16000
    FRAME_SIZE_MS = 25
    FRAME_STRIDE_MS = 10
    NUM_FEATURES = 39  # 13 MFCCs + delta + delta-delta
    
    # HMM
    NUM_STATES = 5
    NUM_ITERATIONS = 20
    
    # DNN
    DNN_HIDDEN_LAYERS = [512, 512, 512]
    DNN_DROPOUT = 0.2
    DNN_LEARNING_RATE = 0.001
    DNN_BATCH_SIZE = 64
    DNN_NUM_EPOCHS = 50
    
    # GMM
    GMM_NUM_COMPONENTS = 16
    GMM_MAX_ITERATIONS = 100
    GMM_CONVERGENCE_THRESHOLD = 1e-6
    
    @staticmethod
    def ensure_dirs():
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
