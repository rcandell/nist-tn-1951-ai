# -*- coding: utf-8 -*-
"""
Propagation Modeling with Multi-Output Neural Networks and Convergence Monitoring

This script trains and evaluates neural network models to predict propagation characteristics
from a large dataset of FFT features. It uses scikit-learn's MLPClassifier for binary classification
of LOS, Obstructed, and Waveguided, and MultiOutputRegressor with MLPRegressor for regression
of RicianK, DelaySpread, MeanDelay, MaxDelay, and PathGain. It includes classifier validation,
error analysis, and convergence monitoring for MLPRegressor.

Key Features:
- Input: 52 FFT columns from training and test CSV files
  ('E:\\meas\\tn1951data\\dataMobile\\ffts\\airedffts_training.csv' and
  'E:\\meas\\tn1951data\\dataMobile\\ffts\\airedffts_testing.csv').
- Excludes columns: ID, Site, TxPos, TxHeight, Freq, Polarization, Range_m, Range_l, CoordX, CoordY.
- Targets:
  - Classification: LOS, Obstructed, Waveguided (binary, mapped to 0/1).
  - Regression: RicianK, DelaySpread, MeanDelay, MaxDelay, PathGain (log10-transformed).
- Preprocessing:
  - Handles NaN and infinities (mean for numerical columns, mode for classification).
  - Clips numerical values to float32 range (-3.4e38 to 3.4e38).
  - Converts PathGain to log10 domain, replacing -inf with minimum finite value.
  - Scales FFT features using StandardScaler (optional, controlled by SCALE_REG_IN).
- Training:
  - Uses chunked processing (chunksize=100,000) for the 4+ GB dataset.
  - Trains MLPClassifier incrementally with partial_fit, skipping single-class chunks.
  - Trains MultiOutputRegressor(MLPRegressor) incrementally with partial_fit.
  - max_iter=1000 controls epochs per partial_fit call, with convergence monitoring to stop early.
- Convergence Monitoring:
  - Tracks training loss and validation RMSE per MLPRegressor.
  - Stops training if loss changes are below TOL (1e-4) for N_ITER_NO_CHANGE (10) chunks.
  - Supports multiple passes over the dataset (MAX_EPOCHS=5).
- Validation:
  - Validates classifiers for training status (coefs_, intercepts_, n_features_in_, classes_).
- Error Analysis:
  - Identifies high-error regression instances (top 10% or above threshold).
  - Saves high-error records to CSV with metadata, FFT features, and errors.
- Parameters:
  - TRAIN_MODELS: True to train and save models, False to load existing models.
  - SCALE_REG_IN: True to scale FFT features, False to use raw values.
  - MAX_ITER: Epochs per partial_fit call (set to 1000).
  - MAX_EPOCHS: Maximum passes over the dataset (default: 5).
  - TOL: Tolerance for loss convergence (default: 1e-4).
  - N_ITER_NO_CHANGE: Chunks with no loss improvement to stop (default: 10).
- Outputs:
  - Saved models: mlp_classifier_<target>.pkl, mlp_regressor_multi.pkl, scaler.pkl.
  - Evaluation metrics: Accuracy, classification report, RMSE, R^2 for test data.
  - Visualizations: Error distributions, actual vs predicted, confusion matrices.
  - Convergence log: 'convergence_log.txt' with loss, RMSE, and iteration counts.
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
import logging

# Configure logging
logging.basicConfig(filename='convergence_log.txt', level=logging.INFO, 
                    format='%(asctime)s - %(message)s')

SAVE_RESULTS = True
SAVE_PLOTS = True
FIG_EXT = '.png'

# Validation function for classifiers
def validate_classifier(clf, clf_name, expected_features=52, expected_classes=[0, 1]):
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
    pkl_path = root_path + '\\pkl'

    # Parameters
    FULL_SET = True # if we are training the full set of meas, set to True
    ERROR_THRESHOLD_PERCENTILE = 90
    TOP_N_ERRORS = 5
    MAX_ITER = 500  # Epochs per partial_fit call
    MAX_EPOCHS = 5 if FULL_SET else 10  # Maximum passes over the dataset
    TOL = 1e-4  # Tolerance for loss convergence
    N_ITER_NO_CHANGE = 10  # Chunks with no loss improvement to stop
    CHUNKSIZE = 100000 if FULL_SET else 10000  # Rows per chunk
    REG_LEARRATE_INIT = 0.001 if FULL_SET else 0.001

    # Define target columns
    target_cols = ['LOS', 'Obstructed', 'Waveguided', 'RicianK', 'DelaySpread', 'MeanDelay', 'MaxDelay', 'PathGain']
    clf_target_cols = ['LOS', 'Obstructed', 'Waveguided']
    reg_target_cols = ['RicianK', 'DelaySpread', 'MeanDelay', 'MaxDelay', 'PathGain']
    metadata_cols = ['ID', 'Site', 'CoordX', 'CoordY']
    numerical_cols = fft_cols = [col for col in pd.read_csv(training_file, nrows=10).columns 
                                 if col not in ['ID', 'Site', 'TxPos', 'TxHeight', 'Freq', 'Polarization', 
                                                'Range_m', 'Range_l', 'CoordX', 'CoordY', 'LOS', 'Obstructed', 
                                                'Waveguided', 'RicianK', 'DelaySpread', 'MeanDelay', 'MaxDelay', 
                                                'PathGain']]

    # Verify FFT column count
    if len(fft_cols) != 52:
        raise ValueError(f"Expected 52 FFT columns, found {len(fft_cols)}")

    # Define columns to load
    columns = fft_cols + target_cols + metadata_cols

    # Define dtypes
    dtypes = {col: 'float32' for col in fft_cols + reg_target_cols}
    for col in clf_target_cols:
        dtypes[col] = 'category'
    for col in metadata_cols:
        dtypes[col] = 'object'

    # Initialize scaler
    scaler = StandardScaler() if SCALE_REG_IN else None

    # Initialize models
    myseed = np.random.randint(65536)
    classifiers = {
        'LOS': MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', 
                             max_iter=MAX_ITER, random_state=myseed, warm_start=True, early_stopping=False),
        'Obstructed': MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', 
                                   max_iter=MAX_ITER, random_state=myseed, warm_start=True, early_stopping=False),
        'Waveguided': MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', 
                                   max_iter=MAX_ITER, random_state=myseed, warm_start=True, early_stopping=False)
    }
    reg = MLPRegressor(hidden_layer_sizes=(500, 100), activation='relu', solver='adam', 
                       learning_rate_init=REG_LEARRATE_INIT, learning_rate='adaptive', 
                       max_iter=MAX_ITER, random_state=myseed, warm_start=True, early_stopping=False, 
                       tol=TOL, n_iter_no_change=N_ITER_NO_CHANGE, alpha=0.01)
    multi_reg = MultiOutputRegressor(reg)

    # Load or train models
    match TRAIN_MODELS:
        case True:
            print("Training models...")
            logging.info(f"Starting model training for {file} with max_iter={MAX_ITER}, max_epochs={MAX_EPOCHS}")

            # Fit scaler on a sample
            if SCALE_REG_IN:
                sample_df = pd.read_csv(training_file, usecols=fft_cols, 
                                        dtype={col: 'float32' for col in fft_cols}, nrows=100000)
                scaler.fit(sample_df[fft_cols])

            # Create validation set (100,000 rows)
            val_df = pd.read_csv(training_file, usecols=columns, dtype=dtypes, nrows=100000)
            val_df = val_df[~val_df[numerical_cols].isin([np.inf, -np.inf]).any(axis=1)]
            np.seterr(divide='ignore') 
            pgx = 10 * np.log10(val_df['PathGain'])
            pgx = pgx.replace([-np.inf], min(pgx[pgx != -np.inf]))
            val_df['PathGain'] = pgx.copy()
            for col in clf_target_cols:
                mode = val_df[col].mode()[0] if not val_df[col].isna().all() else 0
                val_df[col] = val_df[col].replace([np.inf, -np.inf], mode).fillna(mode)
                val_df[col] = val_df[col].astype(str).str.lower().map({
                    '0': 0, '1': 1, 'true': 1, 'false': 0, 'yes': 1, 'no': 0, 'nan': mode, '-1': 0
                }).infer_objects(copy=False).fillna(mode).astype(int)
            for col in numerical_cols + reg_target_cols:
                col_mean = val_df[col][~val_df[col].isna()].mean()
                val_df[col] = val_df[col].fillna(col_mean)
                val_df[col] = val_df[col].clip(lower=-3.4e38, upper=3.4e38)
            X_val = scaler.transform(val_df[numerical_cols]) if SCALE_REG_IN else val_df[numerical_cols]
            y_val_reg = val_df[reg_target_cols]

            # Initialize convergence tracking
            loss_history = [[] for _ in reg_target_cols]
            val_rmse_history = [[] for _ in reg_target_cols]
            no_change_counts = [0] * len(reg_target_cols)
            converged = [False] * len(reg_target_cols)
            total_iterations = 0

            # Training loop with multiple epochs
            for epoch in range(MAX_EPOCHS):
                if all(converged):
                    print(f"All regressors converged at epoch {epoch}")
                    logging.info(f"All regressors converged at epoch {epoch}")
                    break

                chunks = pd.read_csv(training_file, usecols=columns, dtype=dtypes, chunksize=CHUNKSIZE)
                chunk_count = 0
                for i, chunk in enumerate(chunks):
                    chunk_count += 1
                    initial_rows = chunk.shape[0]
                    chunk = chunk[~chunk[numerical_cols].isin([np.inf, -np.inf]).any(axis=1)]
                    dropped_rows = initial_rows - chunk.shape[0]
                    print(f"Epoch {epoch+1}, Chunk {i+1}: Dropped {dropped_rows} rows with infinities")

                    np.seterr(divide='ignore') 
                    pgx = 10 * np.log10(chunk['PathGain'])
                    pgx = pgx.replace([-np.inf], min(pgx[pgx != -np.inf]))
                    chunk['PathGain'] = pgx.copy()
                    for col in clf_target_cols:
                        mode = chunk[col].mode()[0] if not chunk[col].isna().all() else 0
                        chunk[col] = chunk[col].replace([np.inf, -np.inf], mode).fillna(mode)
                        chunk[col] = chunk[col].astype(str).str.lower().map({
                            '0': 0, '1': 1, 'true': 1, 'false': 0, 'yes': 1, 'no': 0, 'nan': mode, '-1': 0
                        }).infer_objects(copy=False).fillna(mode).astype(int)
                    for col in numerical_cols + reg_target_cols:
                        col_mean = chunk[col][~chunk[col].isna()].mean()
                        chunk[col] = chunk[col].fillna(col_mean)
                        chunk[col] = chunk[col].clip(lower=-3.4e38, upper=3.4e38)

                    X = scaler.transform(chunk[numerical_cols]) if SCALE_REG_IN else chunk[numerical_cols]
                    y_clf = {col: chunk[col] for col in clf_target_cols}
                    y_reg = chunk[reg_target_cols]

                    print("Training classifiers on this chunk...")
                    for col, clf in classifiers.items():
                        unique_classes = np.unique(y_clf[col])
                        if len(unique_classes) < 2:
                            print(f"Skipping partial_fit for {col} in epoch {epoch+1}, chunk {i+1}: only {unique_classes} found")
                            continue
                        clf.partial_fit(X, y_clf[col], classes=[0, 1])

                    print("Training regressors on this chunk...")
                    multi_reg.partial_fit(X, y_reg)
                    total_iterations += MAX_ITER
                    y_pred_val = multi_reg.predict(X_val)
                    chunk_losses = []
                    chunk_rmses = []
                    for j, target in enumerate(reg_target_cols):
                        if converged[j]:
                            continue
                        loss = multi_reg.estimators_[j].loss_
                        chunk_losses.append(loss)
                        loss_history[j].append(loss)
                        rmse = np.sqrt(mean_squared_error(y_val_reg[target], y_pred_val[:, j]))
                        chunk_rmses.append(rmse)
                        val_rmse_history[j].append(rmse)
                        if len(loss_history[j]) > 1:
                            loss_diff = abs(loss_history[j][-2] - loss_history[j][-1])
                            if loss_diff < TOL:
                                no_change_counts[j] += 1
                            else:
                                no_change_counts[j] = 0
                            if no_change_counts[j] >= N_ITER_NO_CHANGE:
                                converged[j] = True
                                print(f"{target} converged at epoch {epoch+1}, chunk {i+1}")
                                logging.info(f"{target} converged at epoch {epoch+1}, chunk {i+1}")

                    print(f"Epoch {epoch+1}, Chunk {i+1} - Iterations: {MAX_ITER}, Losses: {[f'{l:.6f}' for l in chunk_losses]}, Validation RMSEs: {[f'{r:.6f}' for r in chunk_rmses]}")
                    logging.info(f"Epoch {epoch+1}, Chunk {i+1} - Iterations: {MAX_ITER}, Total Iterations: {total_iterations}, Losses: {chunk_losses}, Validation RMSEs: {chunk_rmses}")

                print(f"Epoch {epoch+1} completed with {chunk_count} chunks, Total Iterations: {total_iterations}")
                logging.info(f"Epoch {epoch+1} completed with {chunk_count} chunks, Total Iterations: {total_iterations}")

            # Plot convergence curves
            for j, target in enumerate(reg_target_cols):
                plt.figure(figsize=(10, 6))
                plt.plot(loss_history[j], label='Training Loss')
                plt.plot(val_rmse_history[j], label='Validation RMSE')
                plt.xlabel('Chunk')
                plt.ylabel('Metric')
                plt.title(f'Convergence for {target} (max_iter={MAX_ITER})')
                plt.legend()
                plt.grid(True)
                plt.show() if not SAVE_PLOTS else None
                save_fig_to_file(plt, root_path, file + f'__convergence_{target}', FIG_EXT) if SAVE_PLOTS else None
                plt.close()

            # Save models
            for col, clf in classifiers.items():
                joblib.dump(clf, pkl_path + '\\' + file + '_' + f'mlp_classifier_{col.lower()}.pkl')
            joblib.dump(multi_reg, pkl_path + '\\' + file + '_' + 'mlp_regressor_multi.pkl')
            joblib.dump(scaler, pkl_path + '\\' + file + '_' + 'scaler.pkl') if SCALE_REG_IN else None

            print(f"Models trained and saved successfully for {file}. Total Iterations: {total_iterations}")
            logging.info(f"Models trained and saved successfully for {file}. Total Iterations: {total_iterations}")

        case False:
            print(f"Loading existing models for {file}...")
            classifiers = {
                'LOS': joblib.load(pkl_path + '\\' + file + '_' + 'mlp_classifier_los.pkl'),
                'Obstructed': joblib.load(pkl_path + '\\' + file + '_' + 'mlp_classifier_obstructed.pkl'),
                'Waveguided': joblib.load(pkl_path + '\\' + file + '_' + 'mlp_classifier_waveguided.pkl')
            }
            for col, clf in classifiers.items():
                is_valid, message = validate_classifier(clf, col)
                print(message)
                if not is_valid:
                    print(f"Warning: {message}")
            multi_reg = joblib.load(pkl_path + '\\' + file + '_' + 'mlp_regressor_multi.pkl')
            scaler = joblib.load(pkl_path + '\\' + file + '_' + 'scaler.pkl') if SCALE_REG_IN else None

    # Evaluate on test data
    test_df = pd.read_csv(test_file, usecols=columns, dtype=dtypes)
    initial_rows = test_df.shape[0]
    test_df = test_df[~test_df[numerical_cols].isin([np.inf, -np.inf]).any(axis=1)]
    dropped_rows = initial_rows - test_df.shape[0]
    print(f"Test Data for {file}: Dropped {dropped_rows} rows with infinities")

    pgx = 10 * np.log10(test_df['PathGain'])
    pgx = pgx.replace([-np.inf], min(pgx[pgx != -np.inf]))
    test_df['PathGain'] = pgx.copy()
    for col in clf_target_cols:
        mode = test_df[col].mode()[0] if not test_df[col].isna().all() else 0
        test_df[col] = test_df[col].replace([np.inf, -np.inf], mode).fillna(mode)
        test_df[col] = test_df[col].astype(str).str.lower().map({
            '0': 0, '1': 1, 'true': 1, 'false': 0, 'yes': 1, 'no': 0, 'nan': mode, '-1': 0
        }).fillna(mode).astype(int)
    for col in numerical_cols + reg_target_cols:
        col_mean = test_df[col][~test_df[col].isna()].mean()
        test_df[col] = test_df[col].fillna(col_mean)
        test_df[col] = test_df[col].clip(lower=-3.4e38, upper=3.4e38)

    X_test = scaler.transform(test_df[numerical_cols]) if SCALE_REG_IN else test_df[numerical_cols]
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
        abs_errors = np.abs(y_test_reg[target] - y_pred_reg[:, i])
        error_threshold = np.percentile(abs_errors, ERROR_THRESHOLD_PERCENTILE)
        high_error_mask = abs_errors > error_threshold
        high_error_df = pd.DataFrame({
            'Index': test_df[high_error_mask].index,
            'Actual': y_test_reg[target][high_error_mask],
            'Predicted': y_pred_reg[:, i][high_error_mask],
            'Absolute_Error': abs_errors[high_error_mask],
        })
        high_error_df = pd.concat([
            test_df.loc[high_error_mask, metadata_cols + numerical_cols + clf_target_cols],
            high_error_df
        ], axis=1)
        high_error_df['Target'] = target
        high_error_df = high_error_df.sort_values('Absolute_Error', ascending=False)
        error_dfs.append(high_error_df)

        print(f"\n{target} - High-Error Instances (>{ERROR_THRESHOLD_PERCENTILE}th percentile, threshold={error_threshold:.4f}):")
        print(f"Number of high-error instances: {len(high_error_df)}")
        print(f"Mean Absolute Error (high-error): {high_error_df['Absolute_Error'].mean():.4f}")
        print(f"Max Absolute Error: {high_error_df['Absolute_Error'].max():.4f}")
        print(f"Top {TOP_N_ERRORS} High-Error Records for {target}:")
        print(high_error_df[['Index', 'ID', 'Site', 'CoordX', 'CoordY', 'LOS', 'Obstructed', 'Waveguided', 
                             'Actual', 'Predicted', 'Absolute_Error']].head(TOP_N_ERRORS).to_string(index=False))

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
        print(f"Saved {len(high_error_all)} high-error instances to '{high_error_file}'")

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
            save_fig_to_file(plt, root_path, file + f'__confusion_{col}', FIG_EXT) if SAVE_PLOTS else None
            plt.close()
        else:
            print(f"Skipping the {col} classifier for confusion matrix. Not fitted.")

    print(f"Model training and evaluation completed successfully for {file}.")

def list_files(directory, ext, findstr=None):
    the_files = []
    for entry in os.listdir(directory):
        full_path = os.path.join(directory, entry)
        if os.path.isfile(full_path) and entry.endswith(ext) and (True if (findstr is None) else findstr in entry):
            root_name = os.path.splitext(os.path.basename(full_path))[0]
            the_files.append(root_name)
    return the_files

if __name__ == "__main__":
    DOALL = True
    FINDSTR = None
    root_path = ('E:\\meas\\tn1951data\\dataMobile\\ffts\\all' if DOALL else 'E:\\meas\\tn1951data\\dataMobile\\ffts')
    file_ext = '.csv'
    the_files = list_files(root_path, file_ext) if DOALL else list_files(root_path, file_ext, findstr=FINDSTR)
    for file in the_files:
        print(f'Processing the file {file}')
        process_file(root_path, file, TRAIN_MODELS=True)
