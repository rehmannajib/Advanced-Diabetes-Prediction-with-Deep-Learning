# PIMA Diabetes Prediction with Deep Learning

This project uses the **PIMA Indian Diabetes Dataset** to build a deep learning model for predicting the onset of diabetes. The model leverages neural networks implemented using **Keras** and **TensorFlow**, and is evaluated based on several performance metrics like accuracy, AUC score, and confusion matrix.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Summary](#model-summary)
- [Results](#results)

## Project Overview

The aim of this project is to build a **binary classification model** to predict whether or not a patient is likely to develop diabetes based on certain diagnostic measurements. The model uses features like **glucose levels**, **BMI**, and **age** to make predictions. We use a **deep learning** approach with **multiple hidden layers** and **dropout regularization** to prevent overfitting and enhance model performance.

## Dataset

The dataset used is the **PIMA Indian Diabetes Dataset** from the UCI Machine Learning Repository. It consists of 768 samples with 8 features related to diagnostic measurements and one target column (Outcome).

- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration after 2 hours in an oral glucose tolerance test
- **BloodPressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Triceps skinfold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg/(height in m)^2)
- **DiabetesPedigreeFunction**: Diabetes pedigree function
- **Age**: Age in years
- **Outcome**: Class variable (0 if non-diabetic, 1 if diabetic)

You can download the dataset from [here](https://www.kaggle.com/uciml/pima-indians-diabetes-database) or use the link included in the code for fetching the data.

## Dependencies

To run the code, you need the following Python libraries:

- **pandas**
- **numpy**
- **tensorflow**
- **scikit-learn**
- **matplotlib**
- **seaborn**

Install them using:

```bash
pip install pandas numpy tensorflow scikit-learn matplotlib seaborn
```


##  Project Structure
The project is structured as follows:

```bash
├── pima_diabetes_prediction.py   # Main Python script
├── README.md                     # Project documentation
```

## Usage
To run the project, follow these steps:

1 - **Clone the repository:**
```bash
git clone https://github.com/your-username/pima-diabetes-prediction.git
cd pima-diabetes-prediction
```
2 - **Run the Python script:**

```bash
python pima_diabetes_prediction.py
```

## Model Summary
The model architecture consists of the following layers:

- **Input Layer:** 8 input features (after scaling)
- **First Hidden Layer:** 128 neurons with ReLU activation
- **Second Hidden Layer:** 64 neurons with ReLU activation + Dropout (0.3)
- **Third Hidden Layer:** 32 neurons with ReLU activation + Dropout (0.3)
- **Output Layer:** 1 neuron with sigmoid activation for binary classification
The model uses **Adam optimizer**, **binary crossentropy** as the loss function, and **accuracy** and **AUC** as metrics.

## Results
After training the model, we evaluate it on the test set using several metrics:

- **Accuracy**: The model achieves an accuracy of approximately 75% on the test set.
- **AUC Score**: The AUC score of the model is around 0.83, indicating good performance in distinguishing between diabetic and non-diabetic patients.
- **Confusion Matrix**: The confusion matrix gives insight into the number of true positives, true negatives, false positives, and false negatives.

**Sample Results:**
- **Confusion Matrix:**
```bash
[[128  19]
 [ 32  52]]
```
- **Classification Report:**

```bash
              precision    recall  f1-score   support

           0       0.80      0.87      0.83       147
           1       0.73      0.62      0.67        84

    accuracy                           0.77       231
   macro avg       0.76      0.74      0.75       231
weighted avg       0.77      0.77      0.77       231
```


