import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

def compute_far_frr(scores, labels, threshold):
    preds = (scores >= threshold).astype(int)
    far = ((preds == 1) & (labels == 0)).sum() / max(1, (labels == 0).sum())
    frr = ((preds == 0) & (labels == 1)).sum() / max(1, (labels == 1).sum())
    return far, frr

def main(data_csv, model_pkl, scaler_pkl):
    df = pd.read_csv(data_csv)
    y = df['user_id']
    X = df.drop(columns=['user_id'])


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Load model and scaler
    model = joblib.load(model_pkl)
    scaler = joblib.load(scaler_pkl)

    # Scale test set
    X_test_scaled = scaler.transform(X_test)

    # Predict probabilities
    proba = model.predict_proba(X_test_scaled)
    max_scores = proba.max(axis=1)

    # Predicted user IDs
    preds = model.predict(X_test_scaled)

    # Binary labels: 1 = correct user, 0 = impostor
    labels = (preds == y_test).astype(int).values

    # Check label variety
    if len(np.unique(labels)) < 2:
        print("AUC cannot be computed: only one class present in labels.")
        return

    # Find best EER
    thresholds = np.linspace(0, 1, 101)
    best_eer, best_t = 1.0, 0.5
    for t in thresholds:
        far, frr = compute_far_frr(max_scores, labels, t)
        eer = (far + frr) / 2
        if abs(far - frr) < abs(best_eer * 2 - 0):
            best_eer, best_t = eer, t

    # Compute AUC
    auc = roc_auc_score(labels, max_scores)

    print('AUC:', round(auc, 4))
    print('Approx EER:', round(best_eer, 4), 'at threshold', round(best_t, 2))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--scaler', required=True)
    args = ap.parse_args()
    main(args.data, args.model, args.scaler)
