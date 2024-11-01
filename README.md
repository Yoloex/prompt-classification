# IFTTT Prompt Classification

An IFTTT prompt classifier based on BERT

## Introduction

IFTTT is a platform that allows users to create customized integrations called Applets using `triggers` and `actions`. One of IFTTT's pages shows a text input field where a user could try to discover something. User prompts fall into the following categories:

- keyword search, results would be services we have and applets created by other users, examples are:
  
  - `blink doorbell`
  - `turn off tv`
  - `alexa`

- applet description - usually contains mention of an event and action, could be in `if ... then ...` form, examples are:

  - `if given an idea, then write a script`
  - `at 11pm set ring mode to home`
  - `When ring allarm is on cam imou is active`
  
- generic problem description, examples are:

  - `Help me be more productive`
  - `How to grow my YouTube channel? I need 100,000 subscribers.`

BERT-based classifier will classify users' prompts into these 3 categories.

## Getting Started

- Dataset Generation
  
  `utils/generate_dataset.py` will use OpenAI's `GPT-4o` model to generate 10000 samples for each class using prompt engieering.

  `utils/generate_dataset_test.py` will show generating samples with `max_tokens` and will confirm possible maximum sample size which can be generated in a single inference.

  `utils/analyze_dataset.ipynb` will assess generated samples and clean data.

- Train
  
  `train.py` will fine-tune `BERT-base-uncased` model on generated dataset.

  `predict.py` will accept a `.csv` file having a single column prompt containing prompts entered by users and produces another `.csv` file where prompts get classified.

## Run

- Environment Setup
  
  ```bat
  conda create -n ifttt python==3.10
  conda activate ifttt
  pip install -r requirements.txt
  ```

- Dataset Generation
  
  - Set `OPENAI_API_KEY` as your own.
  - Run `utils/generate_dataset.py`.
  - Generated dataset will be saved in `data`
  - Explore and clean dataset using `utils/analyze_dataset.ipynb`

- Train

  - Run `train.py`
  - Checkpoint will be saved in `results`.
  - Run `predict.py`

    ```bat
    python predict.py -i input/mock.csv -o output
    ```
