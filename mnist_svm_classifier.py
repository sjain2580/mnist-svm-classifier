# Step 1: Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Step 2: Load and preprocess the dataset
# Load the MNIST dataset from OpenML.
# The 'data_home' parameter specifies where to cache the data.
print("Loading MNIST dataset. This may take a moment...")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

# The data is in a DataFrame, so we convert it to numpy arrays for easier handling
X = X.astype(np.float64)

# Preprocessing: Scale the pixel values
# The original pixel values are integers from 0 to 255.
# We normalize them to be between 0 and 1, which helps the SVM algorithm perform better.
X /= 255.0

print("\nFeatures (X) shape:", X.shape)
print("Target (y) shape:", y.shape)

# Step 3: Split the data into training and testing sets
# For faster training, we will use a smaller subset of the full dataset.
# Using the full 70,000 samples would take a very long time to train with SVM.
# We'll use 10,000 samples for this demonstration.
sample_size = 10000
X_subset, _, y_subset, _ = train_test_split(X, y, train_size=sample_size, random_state=42, stratify=y)

print(f"\nUsing a subset of {sample_size} samples for training and testing.")
X_train, X_test, y_train, y_test = train_test_split(X_subset, y_subset, test_size=0.2, random_state=42, stratify=y_subset)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# Step 4: Model Improvement with Hyperparameter Tuning
# We'll use a Pipeline to chain a scaler and the SVM classifier.
# Scaling is a crucial preprocessing step for SVM, and the pipeline
# ensures it's applied consistently.
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC(random_state=42))
])

# Define the grid of hyperparameters to search through.
# The RBF (Radial Basis Function) kernel is a good choice for image data.
# We will tune the 'C' (regularization parameter) and 'gamma' (kernel coefficient)
# parameters to find the best combination.
param_grid = {
    'classifier__C': [1, 10],
    'classifier__gamma': [0.001, 0.01]
}

# Use GridSearchCV to find the best combination of parameters.
print("\nPerforming GridSearchCV for hyperparameter tuning. This may take a few moments...")
grid_search = GridSearchCV(pipeline, param_grid, n_jobs=-1, verbose=2, cv=3)
grid_search.fit(X_train, y_train)

# The best estimator is the model with the best parameters.
best_model = grid_search.best_estimator_

print("\n--- Hyperparameter Tuning Results ---")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
print("Best parameters found:")
print(grid_search.best_params_)
print("-------------------------------------")

# Step 5: Evaluate the best model
# Make predictions on the test set using the best model.
y_pred = best_model.predict(X_test)

# Calculate the accuracy of the model.
accuracy = accuracy_score(y_test, y_pred)
print("\n--- Final Model Evaluation ---")
print(f"Accuracy: {accuracy:.4f}")

# The classification report provides a more detailed breakdown of performance
# per class, including precision, recall, and f1-score.
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("------------------------")

# Step 6: Make a prediction on a new data sample and visualize it
# Let's pick a random sample from the test set.
sample_index = np.random.randint(0, len(X_test))
sample_image = X_test[sample_index].reshape(28, 28)
true_label = y_test[sample_index]
predicted_label = best_model.predict([X_test[sample_index]])[0]

print("\n--- Prediction for a new sample ---")
print(f"True label: {true_label}")
print(f"Predicted label: {predicted_label}")
print("-----------------------------------")

# Visualize the image
plt.imshow(sample_image, cmap=plt.cm.gray_r, interpolation="nearest")
plt.title(f"Predicted: {predicted_label}, True: {true_label}")
plt.axis('off')
plt.savefig('predicted_vs_true.png')
plt.show()
