import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
                
import pandas as pd
import joblib
from src.models.model import train_model
from src.models.evaluation import evaluate_model

def main():
    #data = pd.read_csv("data/bank.csv", delimiter=";",header='infer')
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/bank.csv'), delimiter=";",header='infer')

    y = data['y']
    X = data.drop(['y'], axis=1)

    model = train_model(X, y)
    metrics = evaluate_model(model, X, y)
    print(metrics)
    joblib.dump(model, 'models/trained_model.pkl')

if __name__ == '__main__':
    main()