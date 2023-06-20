from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained model and the tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3) # since we have three sentiment classes


# Use the tokenizer to encode the text data
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

# Encode our concatenated data
sentences = df['sentence'].values
labels = df['polarity'].values
input_ids, attention_masks, labels = encode_data(tokenizer, sentences, labels, max_length=128)




from torch.utils.data import TensorDataset, random_split, DataLoader

# Combine the training inputs into a TensorDataset
dataset = TensorDataset(input_ids, attention_masks, labels)

# Create a 90-10 train-validation split
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create the DataLoaders for our training and validation datasets
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=32)
validation_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=32)


from transformers import AdamW

# Specify loss function
loss_fn = torch.nn.CrossEntropyLoss()

# Specify optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

# Specify number of training epochs
epochs = 4


# Helper function for formatting elapsed times
import time
import datetime

def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))

# Store quantities such as training and validation loss, 
# validation accuracy, and timings.
training_stats = []

# Measure the total training time for the whole run.
total_t0 = time.time()

# For each epoch...
for epoch_i in range(0, epochs):

    # Training

    print(f'Epoch {epoch_i + 1}/{epochs}')
    print('-' * 10)

    t0 = time.time()
    total_train_loss = 0
    model.train()

    #loop = tqdm(train_dataloader)

    for step, batch in enumerate(train_dataloader):
        b_input_ids = batch[0].to(device)
        b_attention_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()        

        outputs = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_attention_mask, 
                        labels=b_labels)

        loss = outputs.loss
        total_train_loss += loss.item()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

    avg_train_loss = total_train_loss / len(train_dataloader)

    training_time = format_time(time.time() - t0)

    print(f"  Average training loss: {avg_train_loss:.2f}")
    print(f"  Training epoch took: {training_time}")


    # Validation

    t0 = time.time()

    model.eval()

    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    for batch in validation_dataloader:
        b_input_ids = batch[0].to(device)
        b_attention_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        with torch.no_grad():        
            outputs = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_attention_mask,
                            labels=b_labels)
            
        loss = outputs.loss
        total_eval_loss += loss.item()

        logits = outputs.logits

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences, and
        # accumulate it over all batches.
        total_eval_accuracy += flat_accuracy(logits, label_ids)
        

    # Report the final accuracy for this validation run.
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print(f"  Accuracy: {avg_val_accuracy:.2f}")

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    
    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)
    
    print(f"  Validation Loss: {avg_val_loss:.2f}")
    print(f"  Validation took: {validation_time}")

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

print(f"Training complete!")