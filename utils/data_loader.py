import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from .feature_extraction import FeatureExtractor

class SpeechDataset(Dataset):
    def __init__(self, data_dir, config, transform=None):
        self.data_dir = data_dir
        self.config = config
        self.transform = transform
        self.feature_extractor = FeatureExtractor(config)
        
        # Load all file paths and labels
        self.samples = []
        self.labels = []
        self.label_to_idx = {}
        self.idx_to_label = {}
        
        self._load_data()
    
    def _load_data(self):
        """Load data paths and create label mappings for TIMIT dataset structure."""
        label_idx = 0
        phoneme_map = {}
        
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.WAV') or file.endswith('.wav'):
                    # Get the audio file path
                    wav_path = os.path.join(root, file)
                    
                    # Get the corresponding transcript file (.PHN for phoneme labels)
                    phn_file = os.path.splitext(file)[0] + '.PHN'
                    phn_path = os.path.join(root, phn_file)
                    
                    # If there's no phoneme file, try to find .TXT file
                    if not os.path.exists(phn_path):
                        txt_file = os.path.splitext(file)[0] + '.TXT'
                        txt_path = os.path.join(root, txt_file)
                        if os.path.exists(txt_path):
                            # Use the sentence ID (SA1, SA2, etc.) as the label
                            label = os.path.splitext(file)[0]
                            if label not in self.label_to_idx:
                                self.label_to_idx[label] = label_idx
                                self.idx_to_label[label_idx] = label
                                label_idx += 1
                            
                            self.samples.append(wav_path)
                            self.labels.append(self.label_to_idx[label])
                            
                    else:
                        # For phoneme classification, you might want to use the first phoneme
                        # or process each phoneme segment separately
                        try:
                            with open(phn_path, 'r') as f:
                                # Each line in PHN file: start_sample end_sample phoneme
                                lines = f.readlines()
                                if lines:
                                    # Just use the first phoneme as an example
                                    # You might want to handle this differently based on your needs
                                    first_line = lines[0].strip().split()
                                    if len(first_line) >= 3:
                                        phoneme = first_line[2]
                                        if phoneme not in self.label_to_idx:
                                            self.label_to_idx[phoneme] = label_idx
                                            self.idx_to_label[label_idx] = phoneme
                                            label_idx += 1
                                        
                                        self.samples.append(wav_path)
                                        self.labels.append(self.label_to_idx[phoneme])
                        except Exception as e:
                            print(f"Error processing {phn_path}: {e}")
        
        print(f"Loaded {len(self.samples)} samples with {len(self.label_to_idx)} unique labels")
        if len(self.samples) == 0:
            print(f"No samples found in {self.data_dir}. Check the directory structure.")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_path = self.samples[idx]
        label = self.labels[idx]
        
        # Extract features
        features = self.feature_extractor.extract_mfcc(file_path)
        
        if features is None:
            # Return a placeholder if feature extraction failed
            features = np.zeros((100, self.config.NUM_FEATURES))
        
        # Apply transformations if any
        if self.transform:
            features = self.transform(features)
        
        # Convert to tensor
        features = torch.FloatTensor(features)
        label = torch.LongTensor([label])[0]
        
        return features, label

def create_data_loaders(train_dir, test_dir, config):
    """Create train and test data loaders."""
    train_dataset = SpeechDataset(train_dir, config)
    test_dataset = SpeechDataset(test_dir, config)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.DNN_BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.DNN_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    return train_loader, test_loader, train_dataset.label_to_idx, train_dataset.idx_to_label

def collate_fn(batch):
    """Custom collate function to handle variable length sequences."""
    # Sort batch by length (descending order)
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)
    
    # Get sequences and labels
    seqs, labels = zip(*batch)
    
    # Get sequence lengths
    lengths = [seq.shape[0] for seq in seqs]
    max_length = max(lengths)
    
    # Pad sequences
    padded_seqs = []
    for seq in seqs:
        padded_seq = torch.cat([
            seq, 
            torch.zeros(max_length - seq.shape[0], seq.shape[1])
        ]) if seq.shape[0] < max_length else seq
        padded_seqs.append(padded_seq)
    
    # Stack tensors
    padded_seqs = torch.stack(padded_seqs)
    labels = torch.stack(labels)
    lengths = torch.LongTensor(lengths)
    
    return padded_seqs, labels, lengths
