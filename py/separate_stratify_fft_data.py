# -*- coding: utf-8 -*-
"""
Dataset Split for Propagation Modeling with Site Stratification

This script splits a large CSV dataset containing FFT features and propagation characteristics
into training and test sets for use in machine learning tasks, such as predicting LOS, Obstructed,
Waveguided, RicianK, DelaySpread, MeanDelay, MaxDelay, and PathGain. It uses stratified random
sampling to ensure balanced representation of sites in both sets, reducing bias toward any single
site during model training.

Key Features:
- Input: CSV file at 'e:/meas/tn1951data/dataMobile/ffts/airedffts.csv' (4+ GB, 70 columns
  including 52 FFT columns, Site, and target columns).
- Process:
  - Reads the dataset in chunks to handle large file size.
  - Splits the dataset into 80% training and 20% test sets using stratified random sampling
    based on the 'Site' column to maintain site proportions (random_state=42 for reproducibility).
  - Verifies and logs site distribution in training and test sets.
- Outputs:
  - Training set: 'e:/meas/tn1951data/dataMobile/ffts/separated/airedffts_training.csv'
  - Test set: 'e:/meas/tn1951data/dataMobile/ffts/separated/airedffts_testing.csv'
  - Prints shape and site distribution of training and test sets for verification.
- Dependencies: pandas, scikit-learn.

Author: Rick Candell
Date: August 2025
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import os

def separate_files(root_path, root_file):
    csv_file = root_path + '\\' + root_file + '.csv'
    train_file = root_path + '\\separated\\' + root_file + '_training.csv'
    test_file = root_path + '\\separated\\' + root_file + '_testing.csv'

    # Read the CSV file in chunks to handle large size
    chunksize = 100000
    chunks = pd.read_csv(csv_file, chunksize=chunksize)
    df_list = []
    
    # Process chunks
    for chunk in chunks:
        # Ensure 'Site' column is treated as a string
        chunk['Site'] = chunk['Site'].astype(str)
        df_list.append(chunk)
    
    # Concatenate chunks into a single DataFrame
    df = pd.concat(df_list, ignore_index=True)
    print(f"Loaded dataset: {df.shape}")

    # Log site distribution before splitting
    site_counts = df['Site'].value_counts()
    print("\nSite distribution in full dataset:")
    for site, count in site_counts.items():
        print(f"Site {site}: {count} rows ({count/len(df)*100:.2f}%)")

    # Perform stratified split based on Site
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=df['Site']
    )

    # Log site distribution in training and test sets
    print("\nSite distribution in training set:")
    train_site_counts = train_df['Site'].value_counts()
    for site, count in train_site_counts.items():
        print(f"Site {site}: {count} rows ({count/len(train_df)*100:.2f}%)")
    
    print("\nSite distribution in test set:")
    test_site_counts = test_df['Site'].value_counts()
    for site, count in test_site_counts.items():
        print(f"Site {site}: {count} rows ({count/len(test_df)*100:.2f}%)")

    # Save the split files
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    print(f"\nTraining set saved: {train_df.shape}, Test set saved: {test_df.shape}")

def list_files(directory, ext, findstr=None):
    the_files = []
    for entry in os.listdir(directory):
        full_path = os.path.join(directory, entry)
        if os.path.isfile(full_path) and entry.endswith(ext) and (True if findstr is None else findstr in entry):
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
        separate_files(root_path, file)
