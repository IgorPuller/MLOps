import pandas as pd
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

def preprocessor(X):
    data_new = pd.get_dummies(X, columns=['job','marital',
                                         'education', 'contact',
                                         'poutcome'])

    boolean_cols = ['default', 'housing', 'loan']
    data_new[boolean_cols] = data_new[boolean_cols].applymap(lambda x: True if x == 1 else 0)
    data_new['month'] = data_new['month'].map({'jan': 1, 'feb': 3, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 
                                            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}) 
    return data_new   

def train_model(X, y):

    y.replace(('yes', 'no'), (1, 0), inplace=True)
    
    preprocessor_pipeline = FunctionTransformer(preprocessor)
    smote = SMOTE(random_state=42)
    model = XGBClassifier(random_state=42)

    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor_pipeline),
        ('smote', smote),
        ('model', model)
    ])
    
    pipeline.fit(X, y)
    
    return pipeline
