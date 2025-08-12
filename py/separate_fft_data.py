"""
Dataset Split for Propagation Modeling

This script splits a large CSV dataset containing FFT features and propagation characteristics
into training and test sets for use in machine learning tasks, such as predicting LOS, Obstructed,
Waveguided, RicianK, DelaySpread, MeanDelay, MaxDelay, and PathGain.

Key Features:
- Input: CSV file at 'e:/meas/tn1951data/dataMobile/ffts/airedffts.csv' (4+ GB, 70 columns
  including 52 FFT columns and target columns).
- Process: Splits the dataset into 80% training and 20% test sets using stratified random sampling
  with shuffling (random_state=42 for reproducibility).
- Outputs:
  - Training set: 'e:/meas/tn1951data/dataMobile/ffts/airedffts_training.csv'
  - Test set: 'e:/meas/tn1951data/dataMobile/ffts/airedffts_testing.csv'
- Prints the shape of training and test sets for verification.
- Dependencies: pandas, scikit-learn.

Author: Rick Candell
Date: August 2025
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import os

def separate_files(root_path, root_file):
    
    csv_file = root_path + '\\' + root_file + '.csv'
    train_file = root_path + '\\separated\\' + root_file[0:len(root_file)] + '_training.csv'
    test_file = root_path + '\\separated\\' + root_file[0:len(root_file)] + '_testing.csv'
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Randomly split into 80% train and 20% test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
    
    # Save the split files
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    print(f"Training set: {train_df.shape}, Test set: {test_df.shape}")

def list_files(directory, ext):
    the_files = []
    # List entries in the specified directory only
    for entry in os.listdir(directory):
        full_path = os.path.join(directory, entry)
        # Check if it's a file and ends with .mat
        if os.path.isfile(full_path) and entry.endswith(ext):
            # Get the file name without path and extension
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