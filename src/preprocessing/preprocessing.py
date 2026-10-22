import os
import numpy as np
from python_speech_features import mfcc, delta
import scipy.io.wavfile as wav

class PreprocessExtractor:
    def __init__(self, path="./audio/"):
        self.path = path
        self.winstep = 0.01
        self.x = []
        self.y = []
    
    def silence_trimming(self, audio, threshold=0.01):
        """Menghilangkan bagian audio yang diam dari threshold tertentu"""
        
        threshold_value = threshold * np.max(np.abs(audio))
        non_silent_indices = np.where(np.abs(audio) > threshold_value)[0]
        print(f"Non-silent indices: {non_silent_indices}")
    
        if len(non_silent_indices) > 0:
            return audio[non_silent_indices[0]:non_silent_indices[-1]]
        return audio
    
    def mfcc_extractor(self, audio, sr):
        """Mengextrasi fitur MFCC dari sampel audio"""
        mfcc_features = mfcc(audio, sr)
        mfcc_delta = delta(mfcc_features, 2)
        mfcc_delta_delta = delta(mfcc_delta, 2)
        mfcc_stacked = np.hstack((mfcc_features, mfcc_delta, mfcc_delta_delta))
        print(f"MFCC shape: {mfcc_stacked.shape}")
        return mfcc_stacked
    
    def normalize_features(self, features):
        """Normalisasi fitur dengan mengurangi rata-rata dan membagi dengan standar deviasi"""
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)    
        return (features - mean) / (std + 1e-8)
    
    def get_duration(self, num_frames):
        """Menghitung durasi total dari sampel audio berdasarkan jumlah frame"""
        num_time = num_frames * self.winstep
        return num_time
    
    def get_time_axis(self, num_frames):
        """Mengembalikan sumbu waktu untuk jumlah frame tertentu"""
        return np.arange(num_frames) * self.winstep
    
    def preprocess_extract_pipeline(self):
        """Pipeline untuk preprocess dan ekstraksi fitur dari sampel audio"""
        for vowel in os.listdir(self.path):
            folder_path = os.path.join(self.path, vowel)
            
            if not os.path.isdir(folder_path):
                continue
            
            for file in os.listdir(folder_path):
                if file.endswith(".wav"):
                    file_path = os.path.join(folder_path, file)
                    
                    sr, audio = wav.read(file_path)
                    
                    audio = self.silence_trimming(audio)
                    mfcc_features = self.mfcc_extractor(audio, sr)
                    normalized_features = self.normalize_features(mfcc_features)
                    
                    self.x.append(normalized_features)
                    self.y.append(vowel)
                    
                    print(f"Processed file: {file_path}, MFCC shape: {normalized_features.shape}")
                    
    def get_dtw_samples(self):
        """Mengambil sampel untuk Dynamic Time Warping (DTW)"""
        return self.x, self.y
    
    def get_gaussian_samples(self):
        """Mengambil sampel untuk Gaussian Mixture Model (GMM)"""
        x_avg = [np.mean(sample, axis=0) for sample in self.x]
        return np.array(x_avg), np.array(self.y)

preprocess_extractor = PreprocessExtractor()
preprocess_extractor.preprocess_extract_pipeline()
print(f"Duration of first sample: {preprocess_extractor.get_duration(len(preprocess_extractor.x)):.2f} seconds")

timestamps = preprocess_extractor.get_time_axis(len(preprocess_extractor.x))
print(f"The 10th frame happened at exactly {timestamps[9]} seconds.")
