import os
import numpy as np
from python_speech_features import mfcc, delta
import scipy.io.wavfile as wav

# Path: src/preprocessing/ -> root project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
PREPROCESSED_DIR = os.path.join(DATA_DIR, "preprocessed")

# Dataset:
# nama -> (folder, take)
# None -> semua take
DATASETS = {
    "template": ("template", None),
    "test_closed": ("test/closed", None),
    "test_open": ("test/open", ["02"]), # kalau pakai semua, ubah ke None
}

class PreprocessExtractor:
    def __init__(self, path="./data/", takes=None):
        self.path = path
        self.takes = takes
        self.winstep = 0.01
        self.winlen = 0.025
        self.x = []
        self.y = []
        self.speakers = []
        self.files = []
    
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
    
    def parse_filename(self, file):
        """
        Nama file -> (speaker, take)
        contoh: G09_S02_A01.wav -> ("G09_S02", "01")
        """
        parts = os.path.splitext(file)[0].split("_")
        if len(parts) != 3:
            raise ValueError(f"Format nama file salah: {file} (harus G0X_S0Y_<V>0Z.wav)")

        group, speaker, vowel_take = parts
        return f"{group}_{speaker}", vowel_take[1:]

    def preprocess_extract_pipeline(self):
        """Pipeline untuk preprocess dan ekstraksi fitur dari sampel audio"""
        for vowel in sorted(os.listdir(self.path)):
            folder_path = os.path.join(self.path, vowel)
            
            if not os.path.isdir(folder_path):
                continue
            
            for file in sorted(os.listdir(folder_path)):
                if file.endswith(".wav"):
                    speaker, take = self.parse_filename(file)

                    # Filter take
                    if self.takes is not None and take not in self.takes:
                        continue

                    file_path = os.path.join(folder_path, file)
                    
                    sr, audio = wav.read(file_path)
                    
                    audio = self.silence_trimming(audio)

                    # Tanpa normalisasi per utterance: CMVN -> mean tiap utterance jadi 0 -> info vowel hilang
                    mfcc_features = self.mfcc_extractor(audio, sr)

                    self.x.append(mfcc_features)
                    self.y.append(vowel)
                    self.speakers.append(speaker)
                    self.files.append(file)

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
        x_array = np.empty(len(self.x), dtype=object)
        for i, features in enumerate(self.x):
            x_array[i] = features
        y_array = np.array(self.y)

        output_dir = os.path.dirname(file_name)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        np.savez(file_name, x=x_array, y=y_array, speaker=np.array(self.speakers), files=np.array(self.files),
                 winlen=self.winlen, winstep=self.winstep)
        print(f"Samples saved to {file_name}")


def preprocess_all():
    """Semua dataset di DATASETS: wav -> npz"""
    for name, (folder, takes) in DATASETS.items():
        folder_path = os.path.join(DATA_DIR, folder)
        if not os.path.isdir(folder_path):
            print(f"Skip {name}: {folder_path} belum ada")
            continue

        extractor = PreprocessExtractor(path=folder_path, takes=takes)
        extractor.preprocess_extract_pipeline()
        if len(extractor.x) == 0:
            print(f"Skip {name}: tidak ada file .wav")
            continue

        extractor.save_samples(os.path.join(PREPROCESSED_DIR, f"{name}_preprocessed.npz"))


if __name__ == "__main__":
    preprocess_all()

    # Cek hasil: template
    data = np.load(os.path.join(PREPROCESSED_DIR, "template_preprocessed.npz"), allow_pickle=True)
    X = data['x']
    y = data['y']
    speakers = [str(speaker) for speaker in data['speaker']]

    print(f"Template: {len(X)} sampel, speaker: {sorted(set(speakers))}")
    print(f"Total 'a' samples available for DTW templates: {len(X[y == 'a'])}")