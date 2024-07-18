import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import joblib
import pandas as pd
from src.models.evaluation import evaluate_model
from sklearn.metrics import accuracy_score

def load_model(model_path):
    return joblib.load(model_path)

def make_predictions(model, X):
    return model.predict(X)

def main():
    model_path = 'models/trained_model.pkl'
    
    model = load_model(model_path)
    
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/bank_new.csv'), delimiter=";",header='infer')
    X = data.drop('y', axis=1) 
    
    y_pred = make_predictions(model, X)
    print(y_pred)

if __name__ == '__main__':
    main()