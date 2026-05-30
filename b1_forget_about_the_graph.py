import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='data/PubMed', name='PubMed')
data = dataset[0]

X = data.x.numpy()
y = data.y.numpy()
train_mask = data.train_mask.numpy()
test_mask  = data.test_mask.numpy()

X_train, y_train = X[train_mask], y[train_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]

# Scaling helps SVM and LR; TF-IDF is sparse so skip centering
scaler = StandardScaler(with_mean=False).fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s  = scaler.transform(X_test)

classifiers = {
    'Logistic Regression': (LogisticRegression(max_iter=2000, C=1.0, random_state=42), True),
    'Linear SVM':          (SVC(kernel='linear', C=1.0, random_state=42),              True),
    'RBF SVM':             (SVC(kernel='rbf',    C=1.0, random_state=42),              True),
    'Random Forest':       (RandomForestClassifier(n_estimators=300, random_state=42), False),
}

rows = []
for name, (clf, use_scaled) in classifiers.items():
    Xtr, Xte = (X_train_s, X_test_s) if use_scaled else (X_train, X_test)
    clf.fit(Xtr, y_train)
    acc = accuracy_score(y_test, clf.predict(Xte))
    rows.append({'Model': name, 'Test Accuracy': round(acc, 4)})

df = pd.DataFrame(rows)
df.to_csv('grl-lab-outputs/b1-out.csv')
