import transformers as tf
import evaluate
from datasets import DatasetDict
import datasets as ds
import pandas as pd
import numpy as np



def main():
    polarities = ['positive', 'negative', 'neutral']
    my_features = ds.Features({'sentence': ds.Value('string'), 'polarity': ds.ClassLabel(names=polarities), 'term': ds.Value('string')})
    df_train = pd.read_json('data/train.json').transpose()[['sentence', 'polarity', 'term']]
    df_test = pd.read_json('data/test.json').transpose()[['sentence', 'polarity', 'term']]
    dataset_test = ds.Dataset.from_pandas(df_test, features=my_features, preserve_index=False)
    dataset_train = ds.Dataset.from_pandas(df_train, features=my_features, preserve_index=False)
    print(dataset_test[1])

    #split our test data in half for validation
    test_valid_dataset = dataset_test.train_test_split(test_size=0.5)

    dataset = DatasetDict({
        'train': dataset_train,
        'test': test_valid_dataset['test'],
        'valid': test_valid_dataset['train']})
    roberta_tokenizer = tf.AutoTokenizer.from_pretrained("roberta-base")

    def tokenize_function(examples):
        
        """ This function tokenizes the text in the examples dictionary.
            We pass it to the map function of the dataset so that we can batch the tokenization for efficiency by
            tokenizing batches in parallel.
        """
        return roberta_tokenizer(examples["sentence"], padding="max_length", truncation=True)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    training_args = tf.TrainingArguments(
        output_dir="my_model",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )

    clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return clf_metrics.compute(predictions=predictions, references=labels)
        print(tokenized_datasets)

    model = tf.AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
    
    trainer = tf.Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.evaluate()


if __name__== "__main__":
    main()