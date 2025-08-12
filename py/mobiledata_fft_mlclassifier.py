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

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score, confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import os

SAVE_RESULTS = False
SAVE_PLOTS = False
FIG_EXT = '.png'

# Validation function for classifiers
def validate_classifier(clf, clf_name, expected_features=52, expected_classes=[0, 1]):
    """
    Validate an MLPClassifier to ensure it was trained and is ready for production.
    
    Parameters:
    - clf: The loaded MLPClassifier object.
    - clf_name: Name of the classifier (e.g., 'LOS') for logging.
    - expected_features: Expected number of input features (default: 52).
    - expected_classes: Expected class labels (default: [0, 1]).
    
    Returns:
    - bool: True if valid, False otherwise.
    - str: Message describing the validation result or error.
    """
    try:
        check_is_fitted(clf, attributes=['coefs_', 'intercepts_', 'n_features_in_', 'classes_'])
        if clf.n_features_in_ != expected_features:
            return False, f"{clf_name}: Expected {expected_features} features, found {clf.n_features_in_}"
        if not np.array_equal(np.sort(clf.classes_), np.sort(expected_classes)):
            return False, f"{clf_name}: Expected classes {expected_classes}, found {clf.classes_}"
        for coef, intercept in zip(clf.coefs_, clf.intercepts_):
            if not np.all(np.isfinite(coef)) or not np.all(np.isfinite(intercept)):
                return False, f"{clf_name}: Non-finite values in weights or biases"
        return True, f"{clf_name}: Classifier is valid and ready for predictions"
    except NotFittedError:
        return False, f"{clf_name}: Classifier is not fitted (fit or partial_fit not called)"
    except Exception as e:
        return False, f"{clf_name}: Validation failed - {str(e)}"
    
def save_fig_to_file(myplot, root_path, name, ext):
    save_path = root_path + '\\figs\\' + name + ext
    myplot.savefig(save_path, dpi=300)
    
def process_file(root_path, file, TRAIN_MODELS=True, SCALE_REG_IN=True):
    
    training_file = root_path + '\\separated\\' + file + '_training.csv'
    test_file = root_path + '\\separated\\' + file + '_testing.csv'
    high_error_file = root_path + '\\results\\' + file + '_high_error_instances.csv'            
    pkl_path = root_path + '\\' + 'pkl'
        
    # paremters
    ERROR_THRESHOLD_PERCENTILE = 90  # Percentile for high-error instances (e.g., top 10%)
    TOP_N_ERRORS = 5  # Number of top high-error records to display per target
    
    # validate parameters
    #TRAIN_MODELS = True  if SCALE_REG_IN else TRAIN_MODELS  # If scaling is enabled, always train models    
    
    # Define target columns
    target_cols = ['LOS', 'Obstructed', 'Waveguided', 'RicianK', 'DelaySpread', 'MeanDelay', 'MaxDelay', 'PathGain']
    clf_target_cols = ['LOS', 'Obstructed', 'Waveguided']
    reg_target_cols = ['RicianK', 'DelaySpread', 'MeanDelay', 'MaxDelay', 'PathGain']
    metadata_cols = ['ID', 'Site', 'CoordX', 'CoordY']  # Metadata for context
    numerical_cols = fft_cols = [col for col in pd.read_csv(training_file, nrows=10).columns 
                                 if col not in ['ID', 'Site', 'TxPos', 'TxHeight', 'Freq', 'Polarization', 
                                                'Range_m', 'Range_l', 'CoordX', 'CoordY', 'LOS', 'Obstructed', 
                                                'Waveguided', 'RicianK', 'DelaySpread', 'MeanDelay', 'MaxDelay', 
                                                'PathGain']]
    
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
    
    # Define columns to load (include metadata for test data)
    columns = fft_cols + target_cols + metadata_cols
    
    # Define dtypes
    dtypes = {col: 'float32' for col in fft_cols + reg_target_cols}
    for col in clf_target_cols:
        dtypes[col] = 'category'
    for col in metadata_cols:
        dtypes[col] = 'object'  # Allow strings for ID, Site, etc.
    
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
                       max_iter=int(1), random_state=myseed, warm_start=True, early_stopping=False)
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
                joblib.dump(clf, pkl_path + '\\' + file + '_' + f'mlp_classifier_{col.lower()}.pkl')
            joblib.dump(multi_reg, pkl_path + '\\' + file + '_' + 'mlp_regressor_multi.pkl')
            joblib.dump(scaler, pkl_path + '\\' + file + '_' + 'scaler.pkl') if SCALE_REG_IN else None
    
            print("Models trained and saved successfully.")
    
        case False:
            print("Loading existing models...")
            # Load classifiers
            classifiers = {
                'LOS': joblib.load(pkl_path + '\\' + file + '_' + 'mlp_classifier_los.pkl'),
                'Obstructed': joblib.load(pkl_path + '\\' + file + '_' + 'mlp_classifier_obstructed.pkl'),
                'Waveguided': joblib.load(pkl_path + '\\' + file + '_' + 'mlp_classifier_waveguided.pkl')
            }
            multi_reg = joblib.load(pkl_path + '\\' + file + '_' + 'mlp_regressor_multi.pkl')
            scaler = joblib.load(pkl_path + '\\' + file + '_' + 'scaler.pkl') if SCALE_REG_IN else None
                
    
    # Evaluate on a test chunk
    test_df = pd.read_csv(test_file, usecols=columns, dtype=dtypes)
    
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
        if validate_classifier(clf, col)[0]:
            y_pred_clf = clf.predict(X_test)
            print(f'\n{col} Accuracy: {accuracy_score(y_test_clf[col], y_pred_clf):.4f}')
            print(f"{col} Classification Report:")
            print(classification_report(y_test_clf[col], y_pred_clf))
        else:
            print(f"Skipping the {col} classifier. Not fitted.")
    
    # Predict and evaluate regression targets
    y_pred_reg = multi_reg.predict(X_test)
    for i, target in enumerate(reg_target_cols):
        rmse = np.sqrt(mean_squared_error(y_test_reg[target], y_pred_reg[:, i]))
        r2 = r2_score(y_test_reg[target], y_pred_reg[:, i])
        print(f'\n{target} - Root Mean Squared Error: {rmse:.4f}, R^2 Score: {r2:.4f}')
        
    # Analyze large prediction errors for regression targets
    error_dfs = []
    for i, target in enumerate(reg_target_cols):
        # Calculate absolute errors
        abs_errors = np.abs(y_test_reg[target] - y_pred_reg[:, i])
        # Define threshold for high errors (top 10% or custom threshold)
        error_threshold = np.percentile(abs_errors, ERROR_THRESHOLD_PERCENTILE)
        high_error_mask = abs_errors > error_threshold
        # Create DataFrame for high-error instances
        high_error_df = pd.DataFrame({
            'Index': test_df[high_error_mask].index,
            'Actual': y_test_reg[target][high_error_mask],
            'Predicted': y_pred_reg[:, i][high_error_mask],
            'Absolute_Error': abs_errors[high_error_mask],
        })
        # Add metadata, FFT features, and classification targets
        high_error_df = pd.concat([
            test_df.loc[high_error_mask, metadata_cols + numerical_cols + clf_target_cols],
            high_error_df
        ], axis=1)
        high_error_df['Target'] = target
        # Sort by absolute error (descending)
        high_error_df = high_error_df.sort_values('Absolute_Error', ascending=False)
        error_dfs.append(high_error_df)
    
        # Print summary and top N high-error records
        print(f"\n{target} - High-Error Instances (>{ERROR_THRESHOLD_PERCENTILE}th percentile, threshold={error_threshold:.4f}):")
        print(f"Number of high-error instances: {len(high_error_df)}")
        print(f"Mean Absolute Error (high-error): {high_error_df['Absolute_Error'].mean():.4f}")
        print(f"Max Absolute Error: {high_error_df['Absolute_Error'].max():.4f}")
        print(f"Top {TOP_N_ERRORS} High-Error Records for {target}:")
        print(high_error_df[['Index', 'ID', 'Site', 'CoordX', 'CoordY', 'LOS', 'Obstructed', 'Waveguided', 
                             'Actual', 'Predicted', 'Absolute_Error']].head(TOP_N_ERRORS).to_string(index=False))
    
        # Plot error distribution
        plt.figure(figsize=(8, 6))
        sns.histplot(abs_errors, bins=50, kde=True)
        plt.axvline(error_threshold, color='red', linestyle='--', label=f'{ERROR_THRESHOLD_PERCENTILE}th Percentile')
        plt.xlabel(f'Absolute Error for {target}')
        plt.ylabel('Frequency')
        plt.title(f'Error Distribution for {target}')
        plt.legend()
        plt.show() if not SAVE_PLOTS else None
        save_fig_to_file(plt, root_path, file + f'__errordist_{target}', FIG_EXT) if SAVE_PLOTS else None
        plt.close()
    
        # Plot actual vs predicted, highlighting top N high-error instances
        top_n_mask = high_error_df.index[:TOP_N_ERRORS]
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test_reg[target][~high_error_mask], y_pred_reg[:, i][~high_error_mask], 
                    alpha=0.5, label='Low Error', color='blue')
        plt.scatter(y_test_reg[target][high_error_mask], y_pred_reg[:, i][high_error_mask], 
                    alpha=0.5, label='High Error', color='red')
        plt.scatter(y_test_reg[target][top_n_mask], y_pred_reg[:, i][top_n_mask], 
                    alpha=0.8, label=f'Top {TOP_N_ERRORS} Errors', color='black', marker='x', s=100)
        plt.plot([y_test_reg[target].min(), y_test_reg[target].max()], 
                 [y_test_reg[target].min(), y_test_reg[target].max()], 'k--')
        plt.xlabel(f'Actual {target}')
        plt.ylabel(f'Predicted {target}')
        plt.title(f'Actual vs Predicted {target} (Top {TOP_N_ERRORS} Errors in Black)')
        plt.legend()
        plt.grid(True)
        plt.show() if not SAVE_PLOTS else None
        save_fig_to_file(plt, root_path, file + f'__errors_{target}', FIG_EXT) if SAVE_PLOTS else None
        plt.close()        
    
    # Combine high-error instances and save to CSV
    if error_dfs:
        high_error_all = pd.concat(error_dfs, axis=0)
        high_error_all.to_csv(high_error_file, index=False) if SAVE_RESULTS else None
        print(f"Saved {len(high_error_all)} high-error instances to 'high_error_instances.csv'")
        
    
    # Visualize confusion matrices for classification targets
    for col, clf in classifiers.items():
        if validate_classifier(clf, col)[0]:
            cm = confusion_matrix(y_test_clf[col], clf.predict(X_test))
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel(f'Predicted {col}')
            plt.ylabel(f'Actual {col}')
            plt.title(f'{col} Confusion Matrix')
            plt.show() if not SAVE_PLOTS else None
            save_fig_to_file(plt, root_path, file + f'__confusion_{target}', FIG_EXT) if SAVE_PLOTS else None
            plt.close()
        else:
            print(f"Skipping the {col} classifier for confusion matrix. Not fitted.")
    
    # Visualize permutation importance for classification targets
    '''
    for col, clf in classifiers.items():
        perm_importance = permutation_importance(clf, X_test, y_test_clf[col], n_repeats=5, random_state=myseed)
        importances = pd.DataFrame({'Feature': fft_cols, 'Importance': perm_importance.importances_mean})
        plt.figure(figsize=(10, 12))
        sns.barplot(x='Importance', y='Feature', data=importances.sort_values('Importance', ascending=False))
        plt.title(f'Permutation Importance for {col} Prediction (FFT Columns)')
        plt.show() if not SAVE_PLOTS else None
    
    
    # Visualize regression predictions (e.g., PathGain)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_reg['PathGain'], y_pred_reg[:, 4], alpha=0.5)
    plt.xlabel('Actual PathGain')
    plt.ylabel('Predicted PathGain')
    plt.title('Actual vs Predicted PathGain')
    plt.grid(True)
    plt.show() if not SAVE_PLOTS else None
    save_fig_to_file(plt, root_path, file + f'__actuals_{target}', FIG_EXT) if SAVE_PLOTS
    
    '''
    
    
    print("Model training and evaluation completed successfully.")
    
def list_files(directory, ext, findstr):
    the_files = []
    # List entries in the specified directory only
    for entry in os.listdir(directory):
        full_path = os.path.join(directory, entry)
        # Check if it's a file and ends with .mat
        if os.path.isfile(full_path) and entry.endswith(ext) and (findstr in entry):
            # Get the file name without path and extension
            root_name = os.path.splitext(os.path.basename(full_path))[0]
            the_files.append(root_name)
    return the_files

    
    
if __name__ == "__main__":    
    
    # root directory
    root_path = 'E:\\meas\\tn1951data\\dataMobile\\ffts'
    file_ext = 'pp.csv'
    
    the_files = list_files(root_path, file_ext, 'GBurg')
    for file in the_files:
        print(f'Processing the file {file}')  
        process_file(root_path, file, TRAIN_MODELS=True)
        
    