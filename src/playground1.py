import pandas as pd
import json
from torch.utils.data import Dataset, DataLoader
import torch

def load_data(file_name):
    with open(file_name, 'r') as f:
        data = json.load(f)
    return data

file_name = "../data/dev.json"
data = load_data(file_name)

# Printing the first 5 entries
for i, (k, v) in enumerate(data.items()):
    if i == 5:
        break
    print(f"ID: {k}, Data: {v}")

# Convert the dictionary to a pandas DataFrame
df = pd.DataFrame.from_dict(data, orient='index')

print(df.head())



class SentimentDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        return {
            "sentence": row.sentence,
            "term": row.term,
            "polarity": 1 if row.polarity == "positive" else 0 if row.polarity == "neutral" else -1,
            "id": row.id
        }

# Create a SentimentDataset instance
sentiment_dataset = SentimentDataset(df)

# Access an element
print(sentiment_dataset[10])

print("---------------------")

# Create a DataLoader
dataloader = DataLoader(sentiment_dataset, batch_size=32, shuffle=True)

# Iterate over the DataLoader
for batch in dataloader:
    print(batch)
    print("####")
    #break