# MNIST Digit Classifier using Support Vector Machine

## Overview

This project demonstrates the use of a Support Vector Machine (SVM) to classify handwritten digits from the MNIST dataset. The model is trained to recognize and classify digits from 0 to 9. The project includes hyperparameter tuning to ensure optimal model performance.

## Features

- Data Loading: Automatically downloads and loads the MNIST dataset from scikit-learn's library.Data Preprocessing: Scales and normalizes the pixel data to improve model accuracy.
- Hyperparameter Tuning: Uses GridSearchCV to find the best combination of SVM parameters (C and gamma) for the dataset.Model Training: Trains an SVM classifier using the best-found parameters.
- Performance Evaluation: Generates a detailed classification report and calculates accuracy on a test set.Visualization: Visualizes a random test sample along with the model's prediction.

## Technologies Used

- Python: The core programming language.
- NumPy: For numerical operations and array handling.
- scikit-learn: The primary machine learning library for model building, preprocessing, and evaluation.
- Matplotlib: For creating visualizations.

## Data Analysis & Processing

The MNIST dataset, consisting of 70,000 images, is loaded directly. To manage training time, a subset of 10,000 samples is used for this demonstration. The pixel values (0-255) are normalized to a range between 0 and 1, a critical step for improving SVM performance. The data is then split into training and testing sets.

## Model Used

The model is a Support Vector Classifier (SVC) from the scikit-learn library. It uses a Radial Basis Function (RBF) kernel, which is highly effective for handling the non-linear relationships present in image data.

## Model Training

The training process involves a Pipeline that first scales the data and then trains the SVM. A GridSearchCV is applied to this pipeline to perform a cross-validated search for the optimal C and gamma hyperparameters. This ensures the final model is robust and not overfitted to a specific data split. The model is trained on 80% of the data subset and evaluated on the remaining 20%.

## How to Run the Project

1. Clone the repository:

```bash
git clone <>
cd <repository_name>
```

2. Create and activate a virtual environment (optional but recommended):python -m venv venv

- On Windows:
  
```bash
.\venv\Scripts\activate
```

- On macOS/Linux:

```bash
source venv/bin/activate
```

3. Install the required libraries:

```bash
pip install -r requirements.txt
```

4. Run the Script:

```bash
python predictive_maintenance.py
```

## Contributors

**<https://github.com/sjain2580>**
Feel free to fork this repository, submit issues, or pull requests to improve the project. Suggestions for model enhancement or additional visualizations are welcome!

## Connect with Me

Feel free to reach out if you have any questions or just want to connect!
**[![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sjain04/)**
**[![GitHub](https://img.shields.io/badge/-GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/sjain2580)**
**[![Email](https://img.shields.io/badge/-Email-D14836?style=flat-square&logo=gmail&logoColor=white)](mailto:sjain040395@gmail.com)**

---
