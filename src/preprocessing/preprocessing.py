import os
import numpy as np
from python_speech_features import mfcc, delta
import scipy.io.wavfile as wav

class PreprocessExtractor:
    def __init__(self, path="./data/"):
        self.path = path
        self.winstep = 0.01
        self.winlen = 0.025
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
        mfcc_features = mfcc(audio, sr, winlen = self.winlen, winstep=self.winstep)
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
        for vowel in sorted(os.listdir(self.path)):
            folder_path = os.path.join(self.path, vowel)
            
            if not os.path.isdir(folder_path):
                continue
            
            for file in sorted(os.listdir(folder_path)):
                if file.endswith(".wav"):
                    file_path = os.path.join(folder_path, file)
                    
                    sr, audio = wav.read(file_path)
                    
                    audio = self.silence_trimming(audio)

                    # Tanpa normalisasi per utterance: CMVN -> mean tiap utterance jadi 0 -> info vowel hilang
                    mfcc_features = self.mfcc_extractor(audio, sr)

                    self.x.append(mfcc_features)
                    self.y.append(vowel)

                    num_frames = mfcc_features.shape[0]
                    duration = self.get_duration(num_frames)
                    print(f"Processed file: {file_path}, MFCC shape: {mfcc_features.shape}, Duration: {duration:.2f} seconds")
    
    def get_dtw_samples(self):
        """Mengambil sampel untuk Dynamic Time Warping (DTW)"""
        return self.x, self.y
    
    def get_gaussian_samples(self):
        """Mengambil sampel untuk Gaussian Mixture Model (GMM)"""
        x_avg = [np.mean(sample, axis=0) for sample in self.x]
        return np.array(x_avg), np.array(self.y)

    def save_samples(self, file_name):
        """Menyimpan sampel yang telah diproses ke file npz"""
        x_array = np.array(self.x, dtype=object)
        y_array = np.array(self.y)
        
        output_dir = os.path.dirname(file_name)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        np.savez(file_name, x=x_array, y=y_array, winlen=self.winlen, winstep=self.winstep)
        print(f"Samples saved to {file_name}")

template_preprocess_extractor = PreprocessExtractor(path="./data/template/")
template_preprocess_extractor.preprocess_extract_pipeline()
template_preprocess_extractor.save_samples("data/preprocessed/template_preprocessed.npz")

closed_preprocess_extractor = PreprocessExtractor(path="./data/test/closed/")
closed_preprocess_extractor.preprocess_extract_pipeline()
closed_preprocess_extractor.save_samples("data/preprocessed/test_closed_preprocessed.npz")

# open_preprocess_extractor = PreprocessExtractor(path="./data/test/open/")
# open_preprocess_extractor.preprocess_extract_pipeline()
# open_preprocess_extractor.save_samples("data/preprocessed/test_open_preprocessed.npz")


first_sample_frames = template_preprocess_extractor.x[0].shape[0]
print(f"Duration of first sample: {template_preprocess_extractor.get_duration(first_sample_frames):.2f} seconds")

timestamps = template_preprocess_extractor.get_time_axis(first_sample_frames)
print(f"The 10th frame happened at exactly {timestamps[9]} seconds.")



data = np.load("./data/preprocessed/template_preprocessed.npz", allow_pickle=True)
X = data['x']
y = data['y']

X_a = X[y == 'a']
X_e = X[y == 'e']
X_i = X[y == 'i']
X_o = X[y == 'o']
X_u = X[y == 'u']

print(f"Total 'a' samples available for DTW templates: {len(X_a)}")
template_a = X_a[0]
print(template_a)