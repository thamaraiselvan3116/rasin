import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# sample dataset (replace with real raisin dataset later)
data = pd.DataFrame({
    "Area": [500, 600, 550, 700],
    "MajorAxisLength": [10, 12, 11, 13],
    "MinorAxisLength": [8, 9, 7, 10],
    "Eccentricity": [0.5, 0.6, 0.55, 0.65],
    "ConvexArea": [520, 620, 560, 720],
    "Extent": [0.7, 0.8, 0.75, 0.85],
    "Perimeter": [30, 35, 32, 38],
    "Class": [0, 1, 0, 1]
})

X = data.drop("Class", axis=1)
y = data["Class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

pickle.dump(model, open("randomforest.pkl", "wb"))

print("✅ RandomForest model saved")