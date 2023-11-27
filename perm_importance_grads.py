import numpy as np
import pandas as pd
import os
import sys
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.metrics import check_scoring

def permutation_importance_pca(model, X, y, metric= 'neg_mean_absolute_error', n_repeats=10000):
    scorer = check_scoring(model, scoring=metric)
    baseline_score = scorer(model, X, y)
    n_features = X.shape[1]

    importances = np.zeros((n_features, n_repeats))

    for i in range(n_features):
        X_permuted = X.copy()
        for n in range(n_repeats):
            X_permuted[:, i] = shuffle(X_permuted[:, i], random_state=n)
            score = scorer(model, X_permuted, y)
            importances[i, n] = baseline_score - score
            
    importances_mean = np.mean(importances, axis=1)
    importances_std = np.std(importances, axis=1)

    return importances_mean, importances_std, importances

data = pd.read_csv("t_data.csv")
data = data.rename(columns={"0": 'participant_id', "1": "FlashType", "2": "correct_rel", "3": "sex", "4": "age"})
gradient_data = np.load("../gradient_data.npy")[:, :649840]
subjects = np.load("../subjects.npy")

grad_labels = []
vrtx_ind = np.arange(0, 64984) # 64984
grads = np.arange(1, 11)
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
             ('pca', PCA(n_components = 0.2))]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("cat_encoder", cat_encoder, cat_features),
        ("whiten", whiten, age),
        ("pca", pca, grad_features)
    ]
)

from sklearn import svm
svr = svm.SVR(kernel = "linear", epsilon = 0.2)
# svr_pipe = Pipeline([
#     ('preprocessor', preprocessor),
#     ('svr', svr)
# ])


X_train, X_test, y_train, y_test = train_test_split(gradient_data, correct_rel, test_size=0.25, random_state=42)

X_train_transformed = preprocessor.fit_transform(X_train)

print(f"N_components for connectivity: {preprocessor.named_transformers_['pca_conn'].named_steps['pca'].n_components_}")
print(f"N_components for gradient: {preprocessor.named_transformers_['pca_grad'].named_steps['pca'].n_components_}")
print(f"N_components for centroid_disp: {preprocessor.named_transformers_['pca_centroid_disp'].named_steps['pca'].n_components_}")
print(f"N_components for cortex_disp: {preprocessor.named_transformers_['pca_cortex_disp'].named_steps['pca'].n_components_}")

X_test_transformed = preprocessor.transform(X_test)
lr_trained = svr.fit(X_train_transformed, y_train)

print("Computing permutation feature importance...")
mean_importances_pca, std_importances_pca, importances_pca = permutation_importance_pca(svr, X_test_transformed, y_test)


np.save("results/mean_importance_pca.npy", mean_importances_pca)
np.save("results/std_importances_pca.npy", std_importances_pca)
np.save("results/importances_pca.npy", importances_pca)
