# End-to-End Housing Price Prediction

## Project Overview

This project implements an end-to-end machine learning workflow to predict housing prices using the California housing dataset. It demonstrates the complete ML lifecycle including data preprocessing, model comparison, training, inference, and validation.

Multiple regression models are evaluated, and the best-performing model is selected for final prediction using a reusable preprocessing pipeline.

---

## Key Features

* End-to-end machine learning pipeline using scikit-learn
* Data preprocessing with imputation, scaling, and encoding
* Model comparison using:

  * Linear Regression
  * Decision Tree Regressor
  * Random Forest Regressor
* Cross-validation using RMSE for evaluation
* Separation of training, inference, and validation data
* CSV-based batch prediction workflow
* Comparison of predicted values with actual house prices

---

## Project Structure

```
end-to-end-housing-price-prediction/
│
├── data/
│   ├── housing.csv              # Original dataset
│   ├── input.csv                # Inference input (no target column)
│   ├── actual_values.csv        # Actual house prices for comparison
│   └── output.csv               # Predicted house prices
│
├── src/
│   ├── main.py                  # Training and inference pipeline
│   └── model_comparison.py      # Model comparison and evaluation
│
├── artifacts/                   # Generated locally (not tracked in Git)
│   ├── model.pkl
│   └── pipeline.pkl
│
├── README.md
└── requirements.txt
```

---

## Workflow

1. **Model Comparison**
   Multiple regression models are trained and evaluated using cross-validation to identify the best-performing model.

2. **Model Training**
   A Random Forest Regressor is trained using a full preprocessing pipeline.

3. **Inference**
   New input data (without target values) is passed through the trained pipeline to generate predictions.

4. **Validation**
   Predicted house prices are compared with actual values to assess performance.

---

## Technologies Used

* Python
* NumPy
* Pandas
* scikit-learn
* Joblib

---

## How to Run the Project

### Step 1: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Run model comparison (optional)

```bash
cd src
python model_comparison.py
```

### Step 3: Train model and generate predictions

```bash
python main.py
```

Predicted results will be saved to:

```
data/output.csv
```

---

## Notes

* Trained model and pipeline files (`.pkl`) are **not included in version control** due to size constraints.
* These artifacts are automatically generated locally when running `main.py`.

---

## Results

The Random Forest Regressor achieved better performance compared to Linear Regression and Decision Tree models and was selected for final deployment.

---

## Author

Suman Kumar Nayak