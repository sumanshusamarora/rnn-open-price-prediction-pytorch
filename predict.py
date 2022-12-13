"""
Predict from model
"""

import argparse

import mlflow
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

model_uri = ""

model = mlflow.pytorch.load_model(model_uri=model_uri)

gpu_available = torch.cuda.is_available()

trainer = pl.Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1 if torch.cuda.is_available() else None)


class TimeSeriesDataset(Dataset):
    """Time Series dataset."""

    def __init__(self, X_ts, X_cat, y):
        """
        """
        self.X_ts = torch.from_numpy(X_ts).float()
        self.X_cat = torch.from_numpy(X_cat).long()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X_ts)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {
            'X_ts': self.X_ts[idx],
            'X_cat': self.X_cat[idx],
            'y': self.y[idx]
        }

        return sample

torch_dataset = TimeSeriesDataset(X_ts=final_ts_array, X_cat=final_cat_array, y=final_y_array)


def create_input_data(data_test, company, time_window):
    """
    This is currently not generic and assumes that data is already preprocessed
    with standard scaler etc..
    """

    data_test_this_company = data_test[data_test.company == company].sort_values(by="Date (AUS)").drop(
        columns=["Date (AUS)", "Days Listed", "company", "ANN Freq.", "Posts", "Views", "Comments"]).reset_index(
        drop=True)
    X_ts = data_test_this_company[cols_to_keep].loc[:time_window - 1].values
    X_cat = np.full((X_ts.shape[0], 1), ordinal_encoder.encoder_dict_["FileName"][company])
    y_true = data_test_this_company["Open"].loc[time_window:].values
    return X_ts, X_cat, y_true


def predict_current(X_ts, X_cat, model):
    """
    Predict values
    """

    test_dataset = TimeSeriesDataset(X_ts=X_ts, X_cat=X_cat, y=np.zeros((X_ts.shape[0])))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    y_pred = trainer.predict(model, test_loader)
    return y_pred