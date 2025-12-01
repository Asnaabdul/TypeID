import argparse, os, joblib 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score 
from model import make_baseline_model
import model

def main(data_csv, out_pkl, out_scaler):
    df = pd.read_csv(data_csv)
    y = df['user_id']
    X = df.drop(columns=['user_id'])
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = make_baseline_model()
    model.fit(Xs, y)
    # Persist
    os.makedirs(os.path.dirname(out_pkl), exist_ok=True)
    joblib.dump(model, out_pkl)
    joblib.dump(scaler, out_scaler)
    print('Saved model to', out_pkl)
    print('Saved scaler to', out_scaler)
    # Quick split accuracy
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, random_state=42, stratify=y)
    model2 = make_baseline_model()
    model2.fit(Xtr, ytr)
    acc = accuracy_score(yte, model2.predict(Xte))
    print('Holdout accuracy:', round(acc,4))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--out', default='ml/models/model.pkl')
    p.add_argument('--scaler', default='ml/models/scaler.pkl')
    args = p.parse_args()
    main(args.data, args.out, args.scaler)
    