import numpy as np
import pandas as pd
import os
import pickle
import sys
sys.path.append('/home/victoria/NeuroConn')
sys.path.append('/home/victoria/surfdist')
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, PolynomialFeatures
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.decomposition import PCA
from sklearn_pandas import DataFrameMapper
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn import svm
import seaborn as sns
from utils import *

data = pd.read_csv("SIFI_full.csv")
data = data[(data["FlashType"] == 1) | (data["FlashType"] == 2)]
data = data[data["NrBeeps"] == 2]
data = data.drop(["correct", "rt","correct_abs", "rt_abs", "rt_Cod","rt_rel", "bl_correct","bl_rt", "CFF", "NrBeeps"], axis=1, index=None)

grad_path = '/home/victoria/server/data/COST/COST_mri/derivatives/rest/derivatives/gradients'
gradient_data = get_gradient_data(grad_path)
subjects = np.array(list(gradient_data.keys()))
gradient_data = np.array(list(gradient_data.values()))

grad_labels = []
vrtx_ind = np.arange(0, 64984)
grads = np.arange(1, 101)
for grad in grads:
    labels = [f"{ind}_grad_{grad}" for ind in vrtx_ind]
    grad_labels.extend(labels)

gradient_data = pd.DataFrame(gradient_data, columns=grad_labels)
gradient_data["participant_id"] = subjects
gradient_data["participant_id"] = gradient_data["participant_id"].astype(str)
data["participant_id"] = data["participant_id"].astype(str)

gradient_data = gradient_data.merge(data, on="participant_id")
correct_rel = gradient_data["correct_rel"]
gradient_data = gradient_data.drop(["participant_id", "correct_rel"], axis=1, index = None)

cat_features = ['FlashType', 'sex']
grad_features = gradient_data.columns.tolist()[:-3]
age = ['age']

whiten = Pipeline(
    steps = [("whiten", StandardScaler())]
)

cat_encoder = Pipeline(
    steps = [("cat_encoder", OneHotEncoder(handle_unknown="ignore"))]
)

pca = Pipeline(
    steps = [("whiten", StandardScaler()),
             ('pca', PCA(n_components = 100))]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("cat_encoder", cat_encoder, cat_features),
        ("whiten", whiten, age),
        ("pca", pca, grad_features)
    ]
)

svr = svm.SVR(kernel = "linear", epsilon = 0.2)
svr_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('svr', svr)
])

X_train, X_test, y_train, y_test = train_test_split(gradient_data, correct_rel, test_size=0.25, random_state=42)

del gradient_data
print("5-fold cross-validation...")

cv = KFold(n_splits = 5, shuffle = True, random_state = 42)
cv_results = cross_validate(svr_pipe, X_train, y_train, cv=cv, scoring=['r2', 'neg_mean_squared_error'], return_estimator =True, n_jobs = -1)
print("Mean R2 across folds: ", np.mean(cv_results['test_r2']))
print("Mean MSE across folds: ", np.mean(cv_results['test_neg_mean_squared_error']))

coefs = pd.DataFrame(
    [
        est[-1].coef_[0] * est[:-1].transform(X_train.iloc[train_idx]).std(axis=0)
        for est, (train_idx, _) in zip(cv_results["estimator"], cv.split(X_train, y_train))
    ]
)
coefs.to_csv("coefs_grads_pca.csv")