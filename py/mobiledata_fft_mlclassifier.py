# -*- coding: utf-8 -*-
"""
Propagation Modeling with Multi-Output Neural Networks

This script trains and evaluates neural network models to predict propagation characteristics
from a large dataset of FFT features. It uses scikit-learn's MLPClassifier for binary classification
of LOS, Obstructed, and Waveguided, and MultiOutputRegressor with MLPRegressor for regression
of RicianK, DelaySpread, MeanDelay, MaxDelay, and PathGain.

Key Features:
- Input: 52 FFT columns (real and imaginary parts) from training and test CSV files
  ('E:\\meas\\tn1951data\\dataMobile\\ffts\\airedffts_training.csv' and
  'E:\\meas\\tn1951data\\dataMobile\\ffts\\airedffts_testing.csv').
- Excludes columns: ID, Site, TxPos, TxHeight, Freq, Polarization, Range_m, Range_l, CoordX, CoordY.
- Targets:
  - Classification: LOS, Obstructed, Waveguided (binary, mapped to 0/1).
  - Regression: RicianK, DelaySpread, MeanDelay, MaxDelay, PathGain (log10-transformed).
- Preprocessing:
  - Handles NaN and infinities (replaced with mean for numerical columns, mode for classification).
  - Clips numerical values to float32 range (-3.4e38 to 3.4e38).
  - Converts PathGain to log10 domain, replacing -inf with -174.
  - Scales FFT features using StandardScaler (optional, controlled by SCALE_REG_IN).
- Training:
  - Uses chunked processing (chunksize=100,000) for the 4+ GB training dataset.
  - Trains MLPClassifier incrementally with partial_fit, skipping chunks with single classes.
  - Trains MultiOutputRegressor(MLPRegressor) for regression targets.
- Parameters:
  - TRAIN_MODELS: Set to True to train and save models, False to load existing models.
  - SCALE_REG_IN: Set to True to scale FFT features, False to use raw values.
- Outputs:
  - Saved models: mlp_classifier_<target>.pkl, mlp_regressor_multi.pkl, scaler.pkl.
  - Evaluation metrics: Accuracy, classification report, RMSE, R^2 for test data.
  - Visualizations: Confusion matrices, permutation importance for classification, scatter plot for PathGain.
- Dependencies: pandas, numpy, scikit-learn, matplotlib, seaborn, joblib.

Author: Rick Candell
Date: August 2025
"""

# Input file
training_file = 'E:\\meas\\tn1951data\\dataMobile\\ffts\\airedffts_training.csv'
test_file = 'E:\\meas\\tn1951data\\dataMobile\\ffts\\airedffts_testing.csv'

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score, confusion_matrix
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# paremters
TRAIN_MODELS = False  # Set to True to train models, False to load existing models
SCALE_REG_IN = True  # Set to True if you want to scale regression targets

# validate parameters
TRAIN_MODELS = True  if SCALE_REG_IN else TRAIN_MODELS  # If scaling is enabled, always train models    

# Define target columns
target_cols = ['LOS', 'Obstructed', 'Waveguided', 'RicianK', 'DelaySpread', 'MeanDelay', 'MaxDelay', 'PathGain']
clf_target_cols = ['LOS', 'Obstructed', 'Waveguided']
reg_target_cols = ['RicianK', 'DelaySpread', 'MeanDelay', 'MaxDelay', 'PathGain']

# Identify FFT columns, explicitly excluding ID and Site
sample_df = pd.read_csv(training_file, nrows=10)
fft_cols = [col for col in sample_df.columns if col not in ['ID', 'Site', 'TxPos', 'TxHeight', 'Freq', 
                                                           'Polarization', 'Range_m', 'Range_l', 'CoordX', 
                                                           'CoordY', 'LOS', 'Obstructed', 'Waveguided', 
                                                           'RicianK', 'DelaySpread', 'MeanDelay', 'MaxDelay', 
                                                           'PathGain']]
print("FFT Columns:", fft_cols)

# Verify FFT column count
if len(fft_cols) != 52:
    raise ValueError(f"Expected 52 FFT columns, found {len(fft_cols)}")

# Define columns to load
columns = fft_cols + target_cols
numerical_cols = fft_cols  # Only FFT columns are numerical features

# Define dtypes
dtypes = {col: 'float32' for col in fft_cols + reg_target_cols}
for col in clf_target_cols:
    dtypes[col] = 'category'

# Initialize scaler
scaler = StandardScaler() if SCALE_REG_IN else None

# Initialize models
myseed = np.random.randint(65536)
classifiers = {
    'LOS': MLPClassifier(hidden_layer_sizes=(1000, 50), activation='relu', solver='adam', 
                         max_iter=1, random_state=myseed, warm_start=True, early_stopping=False),
    'Obstructed': MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', 
                               max_iter=1, random_state=myseed, warm_start=True, early_stopping=False),
    'Waveguided': MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', 
                               max_iter=1, random_state=myseed, warm_start=True, early_stopping=False)
}
reg = MLPRegressor(hidden_layer_sizes=(1000, 500), activation='relu', solver='adam', 
                   max_iter=1, random_state=myseed, warm_start=True, early_stopping=False)
multi_reg = MultiOutputRegressor(reg)

# Load or train models
match  TRAIN_MODELS:
    case True:
        print("Training models...")

        # Fit scaler on a sample
        if SCALE_REG_IN:
            sample_df = pd.read_csv(training_file, usecols=fft_cols, 
                                    dtype={col: 'float32' for col in fft_cols}, nrows=100000)
            scaler.fit(sample_df[fft_cols])

        # Process chunks for training
        chunksize = 100e3 # 100,000 rows per chunk
        chunks = pd.read_csv(training_file, usecols=columns, 
                            dtype=dtypes, chunksize=chunksize)

        for i, chunk in enumerate(chunks):
            
            # drop any rows with -inf for path gain      
            # convert PathGain to log10 domain
            pgx = 10*np.log10(chunk['PathGain'])
            pgx = pgx.replace([-np.inf], min(pgx[pgx!=-np.inf]))
            #pgx = pgx[~pgx['PathGain'].isin([-np.inf])]
            chunk['PathGain'] = pgx.copy()
            
            # Preprocess classification targets
            for col in clf_target_cols:
                print(f"Chunk {i+1} - {col} Unique Values Before Mapping:", chunk[col].unique())
                mode = chunk[col].mode()[0] if not chunk[col].isna().all() else 0
                chunk[col] = chunk[col].replace([np.inf, -np.inf], mode).fillna(mode)
                chunk[col] = chunk[col].astype(str).str.lower().map({
                    '0': 0, '1': 1, 'true': 1, 'false': 0, 'yes': 1, 'no': 0, 'nan': mode, '-1': 0
                }).infer_objects(copy=False).fillna(mode).astype(int)
                print(f"Chunk {i+1} - {col} Unique Values After Mapping:", chunk[col].unique())
            
            # Check regression targets for NaN and large values
            for col in reg_target_cols:
                nan_count = chunk[col].isna().sum()
                max_val = chunk[col][~chunk[col].isna()].max()
                min_val = chunk[col][~chunk[col].isna()].min()
                print(f"Chunk {i+1} - {col}: NaN Count={nan_count}, Max={max_val}, Min={min_val}")
            
            # Impute and clip numerical columns
            for col in numerical_cols + reg_target_cols:
                col_mean = chunk[col][~chunk[col].isna()].mean()
                chunk[col] = chunk[col].fillna(col_mean)
                chunk[col] = chunk[col].clip(lower=-3.4e38, upper=3.4e38)
            
            # Scale features, keeping DataFrame structure
            #numerical_cols_scaling = [col in numerical_cols if col != 'PathGain']
            if SCALE_REG_IN:
                X = pd.DataFrame(scaler.transform(chunk[numerical_cols]), columns=numerical_cols, index=chunk.index)
            else:
                X = pd.DataFrame(chunk[numerical_cols],  columns=numerical_cols, index=chunk.index)
            y_clf = {col: chunk[col] for col in clf_target_cols}
            y_reg = chunk[reg_target_cols]
            
            # Incrementally train classifiers, skipping chunks with single class
            for col, clf in classifiers.items():
                unique_classes = np.unique(y_clf[col])
                if len(unique_classes) < 2:
                    print(f"Skipping partial_fit for {col} in chunk {i+1}: only {unique_classes} found")
                    continue
                clf.partial_fit(X, y_clf[col], classes=[0, 1])
            
            # Train regression model
            multi_reg.partial_fit(X, y_reg)

        # Save models
        for col, clf in classifiers.items():
            joblib.dump(clf, f'mlp_classifier_{col.lower()}.pkl')
        joblib.dump(multi_reg, 'mlp_regressor_multi.pkl')
        joblib.dump(scaler, 'scaler.pkl') if SCALE_REG_IN else None

        print("Models trained and saved successfully.")

    case False:
        print("Loading existing models...")

        # Load classifiers
        classifiers = {
            'LOS': joblib.load('mlp_classifier_los.pkl'),
            'Obstructed': joblib.load('mlp_classifier_obstructed.pkl'),
            'Waveguided': joblib.load('mlp_classifier_waveguided.pkl')
        }
        multi_reg = joblib.load('mlp_regressor_multi.pkl')
        scaler = joblib.load('scaler.pkl') if SCALE_REG_IN else None
            

# Evaluate on a test chunk
test_df = pd.read_csv(test_file, usecols=columns, dtype=dtypes)

''' # convert PathGain to log10 domain
pgx = 10*np.log10(test_df['PathGain'])
pgx = pgx.replace([-np.inf], -174)
test_df['PathGain'] = pgx.copy() '''

# drop any rows with -inf for path gain      
# convert PathGain to log10 domain
pgx = 10*np.log10(test_df['PathGain'])
pgx = pgx.replace([-np.inf], min(pgx[pgx!=-np.inf]))
#pgx = pgx[~pgx['PathGain'].isin([-np.inf])]
test_df['PathGain'] = pgx.copy()

for col in clf_target_cols:
    print(f"Test Chunk - {col} Unique Values Before Mapping:", test_df[col].unique())
    mode = test_df[col].mode()[0] if not test_df[col].isna().all() else 0
    test_df[col] = test_df[col].replace([np.inf, -np.inf], mode).fillna(mode)
    test_df[col] = test_df[col].astype(str).str.lower().map({
        '0': 0, '1': 1, 'true': 1, 'false': 0, 'yes': 1, 'no': 0, 'nan': mode, '-1': 0
    }).fillna(mode).astype(int)
    print(f"Test Chunk - {col} Unique Values After Mapping:", test_df[col].unique())

# Check and preprocess regression targets in test chunk
for col in reg_target_cols:
    nan_count = test_df[col].isna().sum()
    max_val = test_df[col][~test_df[col].isna()].max()
    min_val = test_df[col][~test_df[col].isna()].min()
    print(f"Test Chunk - {col}: NaN Count={nan_count}, Max={max_val}, Min={min_val}")
    test_df[col] = test_df[col].fillna(test_df[col][~test_df[col].isna()].mean())
    test_df[col] = test_df[col].clip(lower=-3.4e38, upper=3.4e38)

# Scale test features, keeping DataFrame structure
if SCALE_REG_IN:
    X_test = pd.DataFrame(scaler.transform(test_df[numerical_cols]), columns=numerical_cols, index=test_df.index)
else:
    X_test = pd.DataFrame(test_df[numerical_cols], columns=numerical_cols, index=test_df.index)
y_test_clf = {col: test_df[col] for col in clf_target_cols}
y_test_reg = test_df[reg_target_cols]

# Predict and evaluate classification targets
for col, clf in classifiers.items():
    y_pred_clf = clf.predict(X_test)
    print(f'\n{col} Accuracy: {accuracy_score(y_test_clf[col], y_pred_clf):.4f}')
    print(f"{col} Classification Report:")
    print(classification_report(y_test_clf[col], y_pred_clf))

# Predict and evaluate regression targets
y_pred_reg = multi_reg.predict(X_test)
for i, target in enumerate(reg_target_cols):
    rmse = np.sqrt(mean_squared_error(y_test_reg[target], y_pred_reg[:, i]))
    r2 = r2_score(y_test_reg[target], y_pred_reg[:, i])
    print(f'\n{target} - Root Mean Squared Error: {rmse:.4f}, R^2 Score: {r2:.4f}')

# Visualize confusion matrices for classification targets
for col, clf in classifiers.items():
    cm = confusion_matrix(y_test_clf[col], clf.predict(X_test))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel(f'Predicted {col}')
    plt.ylabel(f'Actual {col}')
    plt.title(f'{col} Confusion Matrix')
    plt.show()

# Visualize permutation importance for classification targets
'''
for col, clf in classifiers.items():
    perm_importance = permutation_importance(clf, X_test, y_test_clf[col], n_repeats=5, random_state=myseed)
    importances = pd.DataFrame({'Feature': fft_cols, 'Importance': perm_importance.importances_mean})
    plt.figure(figsize=(10, 12))
    sns.barplot(x='Importance', y='Feature', data=importances.sort_values('Importance', ascending=False))
    plt.title(f'Permutation Importance for {col} Prediction (FFT Columns)')
    plt.show()
'''

# Visualize regression predictions (e.g., PathGain)
plt.figure(figsize=(8, 6))
plt.scatter(y_test_reg['PathGain'], y_pred_reg[:, 4], alpha=0.5)
plt.xlabel('Actual PathGain')
plt.ylabel('Predicted PathGain')
plt.title('Actual vs Predicted PathGain')
plt.grid(True)
plt.show()

print("Model training and evaluation completed successfully.")