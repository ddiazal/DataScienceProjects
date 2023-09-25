import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["font.size"] = 10
plt.style.use("bmh")

print(os.sys.version)


PATH = "/Users/danieldiazalmeida/Downloads/DataScienceDatasets"
data = "aapl_raw_data.csv"

df = pd.read_csv(
    os.path.join(PATH, data)
)
print(df.shape)
print(df.info())
print('')
print(df.head())
print('')
print(df.tail())

date_index = pd.date_range(start=df["date"].iloc[0],
                           end=df["date"].iloc[-1],
                           freq='D'
                          )
df_ = pd.DataFrame(index=date_index)
df["date"] = pd.to_datetime(df.date)
df.index = df["date"]
df = df_.join(df)
df = df.drop('date', axis=1)
df.index.name = "date"
print(df.head())

df["open"].plot()
plt.show()

df = df.interpolate(method='index')

sns.displot(df, x="open", kind='hist')
plt.show()

print(df.describe().T)
print(df.isnull().sum())

# -----------------------
# Data splitting
train_idx = int(.8 * df.shape[0])
train_data = df["open"].iloc[:train_idx]
test_data = df["open"].iloc[train_idx:]

def normalizer(data)->tuple:
    """_summary_

    Args:
        data (_type_): _description_

    Returns:
        tuple: _description_
    """
    mean = data.mean()
    std = data.std()

    return (data - mean) / std, mean, std


# normalized train data
train_data_n, train_mean, train_std = normalizer(train_data)
test_data_n = (test_data - train_mean) / train_std


def get_seq(data:pd.DataFrame, seq_length:int=1)->tuple:
    """_summary_

    Args:
        data (pd.DataFrame): _description_
        seq_length (int, optional): _description_. Defaults to 1.

    Returns:
        tuple: _description_
    """
    x = []
    y = []
    for i in range(len(data) - seq_length):
        x.append(data[i:(i+seq_length)])
        y.append(data[(i+seq_length)])

    return np.array(x), np.array(y)


X_train, y_train = get_seq(data=train_data_n)
x_test, y_test = get_seq(data=test_data_n)
print(X_train[:10])
print(y_train[:10])

# ---------------
# Modeling
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader

from torchmetrics import MeanSquaredError
print(torch.__version__)

# training set
train_set = TensorDataset(
    # sequence zero padding
    pad_sequence([torch.tensor(X_train, dtype=torch.float32)]),
    torch.tensor(y_train, dtype=torch.float32)
)
# test set
test_set = TensorDataset(
    # sequence zero padding
    pad_sequence([torch.tensor(x_test, dtype=torch.float32)]),
    torch.tensor(y_test, dtype=torch.float32)
)


g = torch.Generator().manual_seed(42)
train_loader = DataLoader(train_set,
                          batch_size=32, 
                          shuffle=True, generator=g)

test_loader = DataLoader(test_set,
                          batch_size=32, 
                          shuffle=False)

# LSTM model class
class MyRModel(nn.Module):
    def __init__(self):
        super(MyRModel, self).__init__()
        self.hidden_size = 32

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.fcl = nn.Linear(in_features=self.hidden_size,
                             out_features=1
                            )

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        c0 = torch.zeros(1, x.size(0), self.hidden_size)

        x, _ = self.lstm(x, (h0, c0))
        out = self.fcl(x[:, -1, :])
        return out
    

# GRU model class
class MyGRUModel(nn.Module):
    def __init__(self):
        super(MyGRUModel, self).__init__()
        self.hidden_size = 16

        self.gru = nn.GRU(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.fcl = nn.Linear(in_features=self.hidden_size,
                             out_features=1
                            )

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size)

        x, _ = self.gru(x, h0)
        out = self.fcl(x[:, -1, :])
        return out


model = MyGRUModel()
criterion = nn.MSELoss()
optim = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 501

# training loop
for epoch in range(EPOCHS):
    model.train()
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.view(-1, 1, 1)
        yhat = model(x_batch).squeeze()
        loss = criterion(yhat, y_batch)

        optim.zero_grad()
        loss.backward()
        optim.step()

    if epoch%10==0:
        print(f"Epoch: {epoch}"
              f"\nLoss: {loss.item()}"
             )
        
# Define MSE metric
mse = MeanSquaredError()

model.eval()
with torch.no_grad():
    for seqs, labels in test_loader:
        seqs = seqs.view(-1, 1, 1)
        # Pass seqs to net and squeeze the result
        outputs = model(seqs).squeeze()
        mse(outputs, labels)

# Compute final metric value
test_mse = mse.compute()
print(f"Test MSE: {test_mse}")

plt.plot(labels.detach().numpy()*train_std + train_mean)
plt.plot(range(len(labels))[-1], outputs.detach().numpy()[-1]*train_std + train_mean, marker='o')
plt.plot(range(len(labels))[-1], df["open"].iloc[-1], marker='o')
plt.show()