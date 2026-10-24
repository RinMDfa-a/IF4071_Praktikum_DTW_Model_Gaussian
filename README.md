# IF4071 Praktikum DTW Model Gaussian

Pengenalan vokal (`a`, `i`, `u`, `e`, `o`) dengan DTW dan Gaussian Model dari fitur MFCC.

## Requirement

- Python `>= 3.14`
- [uv](https://github.com/astral-sh/uv)

## Setup

```bash
uv sync
```

## Run

Dari root project:

```bash
uv run python main.py
```

Pipeline yang dijalankan:

1. Preprocess semua wav di `data/` menjadi MFCC (39 dim) di `data/preprocessed/`
2. Fit DTW (DBA) dan Gaussian dari template
3. Evaluasi skenario closed/open
4. Simpan hasil ke `results/`

## Struktur data

```text
data/
  template/       # take 01 (template / closed recording)
  test/closed/    # take 02 (closed speaker)
  test/open/      # speaker kelompok lain (open speaker)
  preprocessed/   # hasil MFCC (.npz)
```

Format nama file: `G0X_S0Y_<Vokal>0Z.wav`  
Contoh: `G09_S02_A01.wav`

## Output

Hasil evaluasi ada di folder `results/`:

- `accuracy_summary.csv`
- `accuracy_per_vowel.csv`
- `accuracy_per_speaker.csv`
- `predictions.csv`
- confusion matrix (`.csv` dan `.png`)
