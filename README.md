# Real Estate Price Prediction

This project focuses on predicting real estate prices in Bengaluru, India, using various machine learning models and techniques. The dataset includes property details such as location, size, total square footage, number of bathrooms, price, and more.

---

## Features of the Project
- Data Cleaning and Preprocessing
- Feature Engineering
- Outlier Detection and Removal
- Model Training and Evaluation
- Price Prediction using a trained Linear Regression Model
- Model Export for Deployment

---

## Prerequisites

- Python 3.x
- Pandas
- Numpy
- Matplotlib
- Scikit-learn
- Pickle

Install the required libraries using pip:
```bash
pip install pandas numpy matplotlib scikit-learn
```

---

## Project Workflow

### 1. Data Preprocessing
- Handle missing values by dropping rows with nulls.
- Extract numerical features from categorical ones like `size`.
- Normalize location names.
- Create new features such as `price_per_sqft`.

### 2. Outlier Removal
- Filter properties based on logical conditions (e.g., minimum square footage per bedroom).
- Remove outliers using statistical methods.

### 3. Encoding Categorical Data
- Apply One-Hot Encoding to location features.

### 4. Train-Test Split
- Split the data into training and testing datasets for evaluation.

### 5. Model Training
- Use Linear Regression as the primary model.
- Perform hyperparameter tuning using GridSearchCV for Lasso Regression and Decision Tree Regressor.
- Evaluate models using K-Fold Cross Validation.

### 6. Deployment
- Export the trained model using Pickle.
- Create a JSON file for feature columns to use in deployment.

---

## Code Files

- **real_estate_prediction.ipynb**: Contains the complete code and workflow.
- **model_export.py**: Code to export the trained model.
- **columns.json**: JSON file containing the feature columns.

---

## Example Predictions

### Input:
- Location: `Whitefield`
- Square Feet: `1500`
- Number of Bathrooms: `2`
- Number of Bedrooms: `3`

### Output:
Predicted Price: **₹92.51 Lakhs**

### Usage:
```python
def predict_price(location, sqft, bath, bhk):
    loc_index = np.where(X.columns == location)[0][0]
    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    return lr_clf.predict([x])[0]
```

---

## Results
- The best-performing model achieved an R² score of **0.83** on the test set.
- Lasso and Decision Tree models were also tested but did not outperform Linear Regression.

---

## Exported Files

- **Trained Model**: `real_estate_prediction_model.pkl`
- **Feature Columns**: `columns.json`

---

## How to Run

1. Clone the repository:
```bash
git clone https://github.com/bknaveen/real-estate-price-prediction.git
cd real-estate-price-prediction
```

2. Train the model or use the pre-trained model provided in the repo.

3. Run the prediction script:
```bash
python predict.py
```

---

## Contributors
- **Nabin BK** ([GitHub](https://github.com/bknaveen))
