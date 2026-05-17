import csv
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

X, y = [], []
with open("dataset.csv") as f:
    for row in csv.reader(f):
        if row:
            y.append(row[0])
            X.append([float(v) for v in row[1:]])

X = np.array(X)

# Normalize exactly like the app does
X_norm = X.copy()
for i in range(len(X)):
    wrist_x = X[i][0]
    wrist_y = X[i][1]
    for j in range(21):
        X_norm[i][j*2]   -= wrist_x
        X_norm[i][j*2+1] -= wrist_y

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_norm, y)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Done! Model retrained with normalized data.")