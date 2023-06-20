from cgi import print_environ_usage
from cmath import exp
import pandas as pd
import json
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np

SENTIMENT_TO_LABEL_MAP = {"positive": 2, "neutral": 1, "negative": 1}

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


# Use the tokenizer to encode the text dat
def encode_data(tokenizer, sentences, labels, max_length):
    input_ids = []
    attention_masks = []

    for sentence in sentences:
        encoding = tokenizer.encode_plus(sentence, 
                                         truncation=True, 
                                         max_length=max_length, 
                                         padding='max_length',
                                         return_tensors='pt')

        input_ids.append(encoding['input_ids'])
        attention_masks.append(encoding['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    return input_ids, attention_masks, labels


def get_dataset(file_name, tokenizer, mode="integrate"):
    # load data
    data = load_data(file_name)

    # Convert the dictionary to a pandas DataFrame
    df = pd.DataFrame.from_dict(data, orient='index')

    # Encode our concatenated data
    terms = df['term'].values
    reviews = df['sentence'].values

    if mode == "ignore":
        sentences = reviews
    elif mode == "integrate":
        sentences = "Pay attention to the term " + terms + ". " + reviews
    else:
        raise Exception("Unknown dataset mode.")

    # print(terms)
    # print("-------------------")
    # print(sentences)

    labels = df['polarity'].values

    labels = [SENTIMENT_TO_LABEL_MAP[p] for p in labels]
    input_ids, attention_masks, labels = encode_data(tokenizer, sentences, labels, max_length=128)

    # Combine the training inputs into a TensorDataset
    dataset = TensorDataset(input_ids, attention_masks, labels)
    return dataset

def get_dataloader(file_name, tokenizer, batch_size, mode="integrate"):
    dataset = get_dataset(file_name, tokenizer, mode)
    return DataLoader(dataset, shuffle=True, batch_size=batch_size)

def split_dataset_and_get_loader(dataset, batch_site):
    # Create a 90-10 train-validation split
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create the DataLoaders for our training and validation datasets
    train_dataloader = DataLoader(dataset, shuffle=True, batch_size=32)
    validation_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=32)
    return train_dataloader, validation_dataloader


# Create a DataLoader
def main():
    file_name = "../data/train.json"
    # Load the pre-trained model and the tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3) # since we have three sentiment classes

    sentiment_dataset = SentimentDataset(file_name)
    dataloader = DataLoader(sentiment_dataset, batch_size=32, shuffle=True)
    dataloader = get_dataloader(file_name, tokenizer, 32, mode="integrate")
    # Iterate over the DataLoader
    sum_id = 0
    sum_mask = 0
    for batch in dataloader:
        #print(batch)
        #break

        b_input_ids = batch[0].to(device)
        b_attention_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        #print(b_attention_mask[:, -1])
        #print(b_input_ids[:, -1])
        #print("####")
        sum_id += torch.sum(b_input_ids[:, -1])
        sum_mask += torch.sum(b_attention_mask[:, -1])
    print("#####")
    print(f"sum of ids = {sum_id}; sum of mask = {sum_mask}")
    print(len(sentiment_dataset))

if __name__ == "__main__":
    main()