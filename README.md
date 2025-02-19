# Heart Disease Prediction Model

Overview:
This project aims to predict the presence of heart disease using machine learning techniques. A dataset containing various patient health indicators is used to train and evaluate a predictive model.

Dataset:
- The dataset is stored in `dataset.csv`.
- An explanation of the dataset columns and attributes is provided in `Dataset-explanation.csv`.

Steps in the Pipeline:
1. **Data Loading**: The dataset and explanation file are loaded from the specified directory.
2. **Data Exploration**: The dataset's first few rows, structure, and missing values are examined.
3. **Data Preprocessing**:
   - Missing values, if present, are handled by replacing them with column-wise mean values.
   - Features (independent variables) are separated from the target (dependent variable).
4. **Train-Test Split**:
   - The dataset is split into training (80%) and testing (20%) sets.
5. **Feature Scaling**:
   - StandardScaler is used to normalize feature values for better model performance.
6. **Model Training**:
   - A RandomForestClassifier is trained on the preprocessed data.
7. **Model Evaluation**:
   - Predictions are made on the test set.
   - Accuracy and a classification report are generated to assess performance.
8. **Data Saving**:
   - The preprocessed dataset is saved as `processed_data.csv`.

Requirements:
- Python 3.x
- Required Libraries: `pandas`, `sklearn`

Execution:
Run the script using Python:
```bash
python script.py
```

Output:
- The script prints dataset insights, missing value handling status, and model performance metrics.
- The processed dataset is saved for future use.

Note:
Ensure the dataset path is correctly specified before running the script.

