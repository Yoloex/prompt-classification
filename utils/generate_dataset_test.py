import os
import openai
import pandas as pd
import yaml
from dotenv import load_dotenv

with open("configs.yml", "r") as f:
    config = yaml.safe_load(f.read())

BATCH_SIZE = config["dataset"]["batch_size"]

dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)

# Define categories and prompts for generating samples
categories = ["keyword search", "applet description", "generic problem description"]
prompts = {
    "keyword search": f"""Generate {BATCH_SIZE} examples of search keywords for automation services. 
    For example, 'blink doorbell', 'turn off tv', 'alexa' and etc.
    Only generate prompt without number or bullet points.
    Print results only with one '\n', like this, 'first result\nsecond result'
    """,
    "applet description": f"""Generate {BATCH_SIZE} 'If...then' or 'When...' or other examples for possible automations which can be triggered on IoT, mobile or by APIs. 
    For example, 'if given an idea, then write a script', 'at 11pm set ring mode to home', 'When ring allarm is on cam imou is active' and etc. 
    Only generate prompt without number or bullet points.
    Print results only with one '\n', like this, 'first result\nsecond result'  
    """,
    "generic problem description": f"""Generate {BATCH_SIZE} general problem descriptions where a user wants help with automation.
    For example, 'Help me be more productive', 'How to grow my YouTube channel? I need 100,000 subscribers.' and etc.
    Only generate prompt without number or bullet points.
    Print results only with one '\n', like this, 'first result\nsecond result'  
    """,
}
# Empty list to store generated data
data = []


client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Generate samples in batches
for category, prompt_text in prompts.items():
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are an annotation assistant generating prompt examples for user prompts classification.",
            },
            {
                "role": "user",
                "content": prompt_text,
            },
        ],
        max_tokens=1024,
        n=1,
        stop=None,
    )
    examples = response.choices[0].message.content.split("\n")
    for example in examples:
        data.append({"prompt": example, "label": category})


# Save generated data to CSV
df = pd.DataFrame(data)
df.to_csv("data/ifttt_prompts.csv", index=False)
