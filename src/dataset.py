import pandas as pd
import json
from torch.utils.data import Dataset, DataLoader
import torch

def load_data(file_name):
    with open(file_name, 'r') as f:
        data = json.load(f)
    return data

class SentimentDataset(Dataset):
    def __init__(self, file_name):
        data = load_data(file_name)
        df = pd.DataFrame.from_dict(data, orient='index')
        self.dataframe = df

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


# Create a DataLoader
def main():
    file_name = "../data/dev.json"
    sentiment_dataset = SentimentDataset(file_name)
    dataloader = DataLoader(sentiment_dataset, batch_size=32, shuffle=True)
    # Iterate over the DataLoader
    for batch in dataloader:
        print(batch)
        #print("####")
        break

if __name__ == "__main__":
    main()