from sklearn.ensemble import RandomForestClassifier

def make_baseline_model():
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        n_jobs=-1,
        random_state=42
    )
