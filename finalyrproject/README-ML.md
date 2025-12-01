# TypeID – ML Starter Kit

This starter kit scaffolds the full ML workflow for **behavioral biometrics passwordless login**.

## Quick Start
1. Create a virtualenv and install requirements:
   ```
   pip install -r ml/requirements-ml.txt
   pip install -r backend/requirements-backend.txt
   ```
2. Generate synthetic data (for a dry run):
   ```
   python ml/src/simulate.py
   ```
3. Train a baseline model:
   ```
   python ml/src/train.py --data ml/data/samples.csv --out ml/models/model.pkl
   ```
4. Evaluate:
   ```
   python ml/src/evaluate.py --data ml/data/samples.csv --model ml/models/model.pkl
   ```
5. Run the backend API:
   ```
   python backend/app.py
   ```

## Workflow Phases
- Data capture (frontend/js/capture.js)
- Feature extraction (ml/src/features.py)
- Training (ml/src/train.py)
- Evaluation & thresholds (ml/src/evaluate.py)
- Inference (ml/src/inference.py)
- Integration (backend/app.py)
