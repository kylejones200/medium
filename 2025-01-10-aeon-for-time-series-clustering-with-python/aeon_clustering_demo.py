import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from aeon.datasets import make_example_3_class_dataset
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
from sklearn.model_selection import train_test_split, shuffle=False
from sklearn.metrics import accuracy_score
import logging

# Generate synthetic dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

X, y = make_example_3_class_dataset(n_instances=50, n_timepoints=30, random_state=42)

# Plot a few series
plot_series(X[0], X[1], X[2], labels=["Sine", "Cosine", "Sine2x"])
plt.title("Sample Series from Each Class")
plt.show()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Classifier
clf = KNeighborsTimeSeriesClassifier(n_neighbors=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Accuracy
logging.info(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
logging.info(f"Data shape: {X.shape}")
logging.info(f"Number of classes: {len(set(y))}")
logging.info(f"Class distribution: {np.bincount(y)}")
