from Dataset import EEGDataset
from DiffModels.EEGESN import EchoStateNetwork
from DiffModels.Working import EEGModel
import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_model(model, filename):
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")
def load_model(filename):
    esn = EchoStateNetwork(input_size=8, reservoir_size=100,
                           output_size=1)  # 13 different possibilities but only one output maybe change in future for different percentages?
    return esn.load_state_dict(torch.load(filename))

def split_data(raw_test, train_size=0.7, valid_size=0.2, test_size=0.1):
    #assert train_size + valid_size + test_size == 1, "Split sizes must sum to 1"
    buffer_cut_off = 250 * 4  # Buffer to cut off from both ends
    total_size = len(raw_test)  # Total number of rows in the dataset

    # Calculate indices for training, validation, and test split
    train_cutoff = int((total_size - 2 * buffer_cut_off) * train_size)
    valid_cutoff = train_cutoff + int((total_size - 2 * buffer_cut_off) * valid_size)

    # Apply buffer cut-off and then slice for training, validation, and test sets
    raw_test_without_buffer = raw_test.iloc[buffer_cut_off:-buffer_cut_off]  # Remove buffer rows from start and end
    raw_train_set = raw_test_without_buffer.iloc[:train_cutoff]
    raw_valid_set = raw_test_without_buffer.iloc[train_cutoff:train_cutoff + valid_cutoff]
    raw_test_set = raw_test_without_buffer.iloc[train_cutoff + valid_cutoff:]
    return raw_train_set, raw_valid_set, raw_test_set


import pandas as pd
from Dataset import EEGDataset
import glob
def load_and_process_files(folder_path, segment_size=500, overlap=250, trim_amount=0):
    segments = []
    train_dataset = EEGDataset()
    valid_dataset = EEGDataset()
    test_dataset = EEGDataset()

    file_paths = glob.glob(folder_path + '/*.csv')
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        train, valid, test = split_data(df, train_size=0.7, valid_size=0.2, test_size=0.1)
        train_dataset.add_test(train)
        valid_dataset.add_test(valid)
        test_dataset.add_test(test)
        break
    return train_dataset, valid_dataset, test_dataset

def model_creation(dataSet):
    esn = EchoStateNetwork(input_size=64, reservoir_size=100, output_size=1)# 13 different possibilities but only one output maybe change in future for different percentages?
    train_model(esn, dataSet)
    save_model(esn,"esnModel")
    return esn


def train_model(model, dataSet, epochs=10, learning_rate=0.001):
    for index in range(len(dataSet)):
        train_loader = dataSet[index]
        model.train_model(train_loader, epochs, learning_rate)


def main():
    csv_file_path = '../cata'
    training,valid,test = load_and_process_files(csv_file_path)
    print("ASDASDSAD")

    model_creation(training)
    print("SO sick")


if __name__ == "__main__":
    main()
