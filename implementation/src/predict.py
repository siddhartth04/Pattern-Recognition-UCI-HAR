"""Predict activities from a numeric CSV in the original 561-feature order."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .models import predict

NAMES = {1:'WALKING', 2:'WALKING_UPSTAIRS', 3:'WALKING_DOWNSTAIRS',
         4:'SITTING', 5:'STANDING', 6:'LAYING'}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True, help='Numeric CSV, no header; NaN allowed')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    bundle = joblib.load(args.model)
    X = np.loadtxt(args.input, delimiter=',', ndmin=2)
    pred = predict(bundle, X)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'activity_id': pred, 'activity': [NAMES[v] for v in pred]}).to_csv(args.output, index=False)
    print(f'Wrote {len(pred)} predictions to {args.output}')

if __name__ == '__main__':
    main()
