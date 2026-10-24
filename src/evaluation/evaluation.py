import os

import matplotlib
matplotlib.use("Agg")  # simpan ke file, tanpa window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from seaborn import heatmap

from src.models.dtw import DTWRecognizer
from src.models.gaussian import GaussianRecognizer
from src.preprocessing.preprocessing import PREPROCESSED_DIR, ROOT_DIR

RESULTS_DIR = os.path.join(ROOT_DIR, "results")
VOWELS = ["a", "i", "u", "e", "o"]

# Skenario -> file npz hasil preprocessing
# notes
# closed_recording: data uji = data template (take 01)
# closed_speaker: speaker sama, rekaman beda (take 02)
# open_speaker: speaker dari kelompok lain
SCENARIOS = {
    "closed_recording": "template_preprocessed.npz",
    "closed_speaker": "test_closed_preprocessed.npz",
    "open_speaker": "test_open_preprocessed.npz",
}


def load_samples(file_name):
    """npz -> dict (x, y, speaker, files), None kalau belum ada"""
    path = os.path.join(PREPROCESSED_DIR, file_name)
    if not os.path.exists(path):
        return None

    data = np.load(path, allow_pickle=True)
    return {
        "x": list(data["x"]),
        "y": [str(label) for label in data["y"]],
        "speaker": [str(speaker) for speaker in data["speaker"]],
        "files": [str(file) for file in data["files"]],
    }


def accuracy(records, by):
    """Accuracy = jumlah prediksi benar / jumlah seluruh data uji x 100%, per kelompok `by`"""
    return 100 * records.groupby(by)["correct"].mean()


class Evaluator:
    """Fit DTW + Gaussian dari template -> prediksi semua skenario -> tabel akurasi"""

    def __init__(self, dtw_mode="DBA", dtw_max_iter=5, gaussian_eps=0.0):
        self.systems = {
            "DTW Template": DTWRecognizer(mode=dtw_mode, max_iter=dtw_max_iter),
            "Model Gaussian": GaussianRecognizer(eps=gaussian_eps),
        }
        self.records = pd.DataFrame()

    def fit(self, template):
        """Template (5 speaker x 1 rekaman per vowel) -> fit semua sistem"""
        for name, system in self.systems.items():
            print(f"Fit {name} dengan {len(template['x'])} sampel template")
            system.fit(template["x"], template["y"])

    def run(self):
        """Prediksi setiap skenario yang datanya ada -> self.records (1 baris per file per sistem)"""
        template = load_samples(SCENARIOS["closed_recording"])
        if template is None:
            raise FileNotFoundError("template_preprocessed.npz belum ada, jalankan preprocessing dulu")
        self.fit(template)

        rows = []
        for scenario, file_name in SCENARIOS.items():
            data = load_samples(file_name)
            if data is None:
                print(f"Skip {scenario}: {file_name} belum ada")
                continue

            for name, system in self.systems.items():
                for x, label, speaker, file in zip(data["x"], data["y"], data["speaker"], data["files"]):
                    pred, _ = system.predict(x)
                    rows.append({"system": name, "scenario": scenario, "file": file,
                                 "speaker": speaker, "true": label, "pred": pred})
                print(f"{name} - {scenario}: {len(data['x'])} data uji")

        self.records = pd.DataFrame(rows)
        self.records["correct"] = self.records["true"] == self.records["pred"]
        return self.records

    def summary_table(self):
        """Tabel 1: sistem x skenario + rata-rata (%)"""
        table = accuracy(self.records, ["system", "scenario"]).unstack()
        table = table[[s for s in SCENARIOS if s in table.columns]]
        table["average"] = table.mean(axis=1)
        return table.round(2)

    def per_vowel_table(self):
        """Tabel 2: akurasi per vowel untuk setiap sistem dan skenario (%)"""
        table = accuracy(self.records, ["system", "scenario", "true"]).unstack()
        return table[[v for v in VOWELS if v in table.columns]].round(2)

    def per_speaker_table(self):
        """Tabel 3: akurasi per speaker untuk setiap sistem dan skenario (%)"""
        table = accuracy(self.records, ["system", "scenario", "speaker"]).unstack()
        return table.round(2)

    def confusion_matrices(self):
        """(sistem, skenario) -> confusion matrix (baris: label asli, kolom: prediksi)"""
        matrices = {}
        for (name, scenario), df in self.records.groupby(["system", "scenario"]):
            # DTW bisa "unknown" kalau input < template / 2
            labels = VOWELS + sorted(set(df["pred"]) - set(VOWELS))
            matrix = pd.crosstab(df["true"], df["pred"]).reindex(index=VOWELS, columns=labels, fill_value=0)
            matrix.index.name, matrix.columns.name = "true", "pred"
            matrices[(name, scenario)] = matrix
        return matrices

    def save(self, output_dir=RESULTS_DIR):
        """Simpan semua tabel (csv) dan confusion matrix (csv + png) ke results/"""
        os.makedirs(output_dir, exist_ok=True)

        self.records.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)
        self.summary_table().to_csv(os.path.join(output_dir, "accuracy_summary.csv"))
        self.per_vowel_table().to_csv(os.path.join(output_dir, "accuracy_per_vowel.csv"))
        self.per_speaker_table().to_csv(os.path.join(output_dir, "accuracy_per_speaker.csv"))

        for (name, scenario), matrix in self.confusion_matrices().items():
            file_name = f"confusion_{name.lower().replace(' ', '_')}_{scenario}"
            matrix.to_csv(os.path.join(output_dir, f"{file_name}.csv"))

            plt.figure(figsize=(5, 4))
            heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
            plt.title(f"{name} - {scenario}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{file_name}.png"), dpi=150)
            plt.close()

        print(f"Hasil evaluasi disimpan di {output_dir}")

    def report(self):
        """Cetak semua tabel ke terminal"""
        with pd.option_context("display.width", 200, "display.max_columns", None):
            print("\nTabel 1: Akurasi per skenario (%)")
            print(self.summary_table())
            print("\nTabel 2: Akurasi per vowel (%)")
            print(self.per_vowel_table())
            print("\nTabel 3: Akurasi per speaker (%)")
            print(self.per_speaker_table())
            for (name, scenario), matrix in self.confusion_matrices().items():
                print(f"\nConfusion matrix: {name} - {scenario}")
                print(matrix)


if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.run()
    evaluator.report()
    evaluator.save()
