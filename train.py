import yaml
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

with open("configs.yml", "r") as f:
    config = yaml.safe_load(f.read())

df = pd.read_parquet("ifttt_prompts_cleaned.parquet")

label_mapping = {label: idx for idx, label in enumerate(df["label"].unique())}
df["label"] = df["label"].map(label_mapping)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

tokenizer = BertTokenizer.from_pretrained("gaunernst/bert-tiny-uncased")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")

    return {"accuracy": accuracy, "f1": f1}


def tokenize_function(examples):
    return tokenizer(
        examples["prompt"],
        padding="max_length",
        truncation=True,
        max_length=config["train"]["max_length"],
    )


train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset.set_format(
    type="torch", columns=["input_ids", "attention_mask", "label"]
)
test_dataset.set_format(
    type="torch", columns=["input_ids", "attention_mask", "label"]
)

model = BertForSequenceClassification.from_pretrained(
    "gaunernst/bert-tiny-uncased", num_labels=len(label_mapping)
)


training_args = TrainingArguments(
    output_dir=config["train"]["output"],
    eval_strategy=config["train"]["eval_strategy"],
    save_strategy=config["train"]["save_strategy"],
    learning_rate=config["train"]["learning_rate"],
    per_device_train_batch_size=config["train"]["batch_size"],
    per_device_eval_batch_size=config["train"]["batch_size"],
    num_train_epochs=config["train"]["epochs"],
    weight_decay=config["train"]["weight_decay"],
    dataloader_num_workers=config["train"]["num_workers"],
    load_best_model_at_end=True,
    greater_is_better=True,
    metric_for_best_model="f1",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

if __name__ == "__main__":
    trainer.train()
    trainer.save_model()
    results = trainer.evaluate()

    print(results)
