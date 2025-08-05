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

# CSV data file paths
csv_file = 'e:/meas/tn1951data/dataMobile/ffts/airedffts.csv'
train_file = 'e:/meas/tn1951data/dataMobile/ffts/airedffts_training.csv'
test_file = 'e:/meas/tn1951data/dataMobile/ffts/airedffts_testing.csv'

# Read the CSV file
df = pd.read_csv(csv_file)

# Randomly split into 80% train and 20% test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

# Save the split files
train_df.to_csv(train_file, index=False)
test_df.to_csv(test_file, index=False)

print(f"Training set: {train_df.shape}, Test set: {test_df.shape}")