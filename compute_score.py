from sklearn.metrics import roc_auc_score
import pandas as pd

PUBLIC_LABELS_PATH = "./data/private_info/public.csv"
PRIVATE_LABELS_PATH = "./data/private_info/private.csv"
SUBM_PATH = "./data/submission.csv"
N_PUBLIC = 324726  # number or rows in public dataset

def compute_metric(submission, public_labels, private_labels):
    public_score = roc_auc_score(public_labels.values, submission[:N_PUBLIC])
    private_score = roc_auc_score(private_labels.values, submission[N_PUBLIC:])
    return public_score, private_score

if __name__ == "__main__":
    submission = pd.read_csv(SUBM_PATH, sep="\t")
    public_labels = pd.read_csv(PUBLIC_LABELS_PATH, sep="\t")
    private_labels = pd.read_csv(PRIVATE_LABELS_PATH, sep="\t")

    metric = compute_metric(submission, public_labels, private_labels)
    print(f"ROC AUC public: {metric[0]}")
