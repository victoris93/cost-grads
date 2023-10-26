import numpy as np
import pandas as pd
import os
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, KFold
import seaborn as sns
from utils import *


data = pd.read_csv('trial_data_distances.csv')
data = data[(data["FlashType"] == 1) | (data["FlashType"] == 2)]
correct = data["correct"]

data = data.drop(["participant_id","correct", "answ", "rt", "Dist to V1, RH", "Dist to A1, RH", "dist_ratio_R", "dist_ratio_L", "Condition"], axis=1, index=None) # "Dist to V1, RH", "Dist to A1, RH", "Dist to A1, LH", "dist_ratio_R", "dist_ratio_L"

num_features = ["Jitter", "Dist to A1, LH", "Dist to V1, LH", "age"]
cat_features = ["sex", "FlashType", "NrBeeps"]
ord_features = ["Trial"]

whiten = Pipeline(
    steps = [("whiten", StandardScaler())]
)

cat_encoder = Pipeline(
    steps = [("cat_encoder", OneHotEncoder(handle_unknown="ignore"))]
)

ord_encoder = Pipeline(
    steps = [("cat_encoder", OrdinalEncoder())]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("whiten", whiten, num_features),
        ("ord_encoder", ord_encoder, ord_features),
        ("cat_encoder", cat_encoder, cat_features)
    ]
)

from sklearn import svm
svm = svm.SVC(kernel = "linear")
svm_pipe = Pipeline([('preprocessor', preprocessor),
                ('svm',svm)])

X_train, X_test, y_train, y_test = train_test_split(data, correct, test_size=0.25, random_state=42)
cat_feature_names = ["sex_0", "sex_1", "FlashType_1", "FlashType_2", "NrBeeps_0", "NrBeeps_1", "NrBeeps_2"]
feature_names = ord_features + cat_feature_names + num_features

print("5-fold cross-validation...")
cv = KFold(n_splits = 10, shuffle = True, random_state = 42)
cv_results = cross_validate(svm_pipe, X_train, y_train, cv=cv, scoring=['accuracy', 'f1'], return_estimator =True, n_jobs = -1)

print("Mean CV accuracy: ", np.mean(cv_results['test_accuracy']))
print("Mean CV F1 ", np.mean(cv_results['test_f1']))

coefs = pd.DataFrame(
    [
        est[-1].coef_[0] * est[:-1].transform(X_train.iloc[train_idx]).std(axis=0)
        for est, (train_idx, _) in zip(cv_results["estimator"], cv.split(X_train, y_train))
    ],
    columns = feature_names
)

coefs.to_csv("cv_coefs.csv")

y_train_pred = trained_svm.predict(X_train)
print("Test accuracy: ", trained_svm.score(y_train, y_test))

from sklearn.inspection import permutation_importance

print("Starting permutation feature importance...")
perm_acc = permutation_importance(trained_svm, X_test, y_test,n_repeats=1000, random_state=42, n_jobs = -1)

sorted_importances_idx = perm_acc.importances_mean.argsort()
importances = pd.DataFrame(
    perm_acc.importances[sorted_importances_idx].T,
    columns=X_train.columns[sorted_importances_idx],
)

importances.to_csv("logreg_perm_importances.csv")
print("Permutation importances saved.")