import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.datasets import TUDataset


##
#   READ DATASET
##
dataset = TUDataset(
    "storage/datasets",
    name="AIDS"
)
dataset._data.y = dataset.y.unsqueeze(1).float()
print(dataset)


##
#   EXTRACT FEATURES
##
X = []
X_count = []
y = dataset.y.reshape(-1).numpy()
for data in dataset:
    pooled_features = data.x.sum(0).reshape(1, -1).numpy()
    X.append(pooled_features)
    X_count.append([[1., data.x.shape[0]]])

X = np.concatenate(X, axis=0)
X_count = np.concatenate(X_count, axis=0)
print(X.shape, y.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    stratify=dataset.y,
    train_size=0.8,
    random_state=42
)


##
#   TRAIN LINEAR CLASSIFIER
##
model = LogisticRegression(max_iter=7000)
model.fit(X_train, y_train)

print(model.coef_)
print(model.intercept_)

y_pred = model.predict(X_train)
accuracy = f1_score(y_train, y_pred)
print("Train F1: ", accuracy)

y_pred = model.predict(X_test)
accuracy = f1_score(y_test, y_pred)
print("Test F1: ", accuracy)

print("Coeff: ", -model.intercept_ / model.coef_[0][0], "\n\n")


##
#   MANUAL CLASSIFICATION ON NODE COUNTS
##

X_train, X_test, y_train, y_test = train_test_split(
    X_count,
    y,
    stratify=dataset.y,
    train_size=0.8,
    random_state=42
)

for cutoff in range(50):
    y_pred = np.array(X_test[:, -1] < cutoff, dtype=float)
    accuracy = round(f1_score(y_test, y_pred), 2)
    print(f"Manual Test F1 (>= {cutoff}): ", accuracy)