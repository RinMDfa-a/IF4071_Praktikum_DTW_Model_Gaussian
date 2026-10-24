from src.evaluation.evaluation import Evaluator
from src.preprocessing.preprocessing import preprocess_all


def main():
    # wav -> MFCC 39 dimensi -> data/preprocessed/*.npz
    preprocess_all()

    # Fit DTW + Gaussian dari template -> prediksi -> akurasi
    evaluator = Evaluator(dtw_mode="DBA", dtw_max_iter=5)
    evaluator.run()
    evaluator.report()
    evaluator.save()


if __name__ == "__main__":
    main()
