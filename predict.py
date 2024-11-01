import os
import torch
import argparse
import pandas as pd
from transformers import BertForSequenceClassification, BertTokenizer

model = BertForSequenceClassification.from_pretrained("./results/checkpoint")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

label2id = {
    "keyword search": 0,
    "applet description": 1,
    "generic problem description": 2,
}
id2label = {v: k for k, v in label2id.items()}


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        prog="Prompt Classifier", usage="%(prog)s [options]"
    )

    argparser.add_argument(
        "-i", "--input", default="input/mock.csv", help="Input csv filename"
    )
    argparser.add_argument(
        "-o", "--output", default="output", help="Output file directory"
    )

    try:
        args = argparser.parse_args()
        filename = args.input

        if not filename.endswith(".csv"):
            raise NameError("Not supported file format")

        df = pd.read_csv(filename)
        prompts = df["prompt"].to_list()

        inputs = tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=128,
        )

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predictions = logits.argmax(dim=-1).cpu().numpy()

        outputs = []

        for prompt, pred in zip(prompts, predictions):
            outputs.append({"prompt": prompt, "kind": id2label[pred]})

        df = pd.DataFrame(outputs)
        df.to_csv(
            os.path.join(
                args.output, os.path.basename(filename).split(".")[0] + "_out.csv"
            ),
            index=False,
        )

    except Exception as e:

        print("Error while inferencing\n{}".format(e))
