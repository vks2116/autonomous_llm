# README

## Get Started
You can get started with Autonomous LLM by cloning this GitHub project. This code has been tested to run on Google Colab environment, and macOS for location inference.

For inference, you will need to install dependencies before getting started.

For model fine-tuning and training, this project leveraged NVIDIA A100. You will need to obtain a similar or better GPU due to memory requirements and the need for large RAM size.

Refer to the attached files to set up your environment and prepare for the implementation. The provided scripts detail the basic initialization steps and outline the resources necessary to begin.


## Install Dependencies

The installation process includes all required Python libraries and frameworks. Use the following commands to install dependencies:
```
!pip install openai==0.28
!pip install -U openai==0.28
!pip install langchain_openai google-search-results
!pip install flask
!pip install transformers==4.28.1
!pip install huggingface-hub==0.14.1
!pip install torch==2.0.1
!pip install tqdm==4.65.0
!pip install prompt_toolkit==3.0.38
!pip install sentencepiece==0.1.99
!pip install accelerate==0.19.0
!pip install einops==0.7.0
!pip install bitsandbytes peft flask
!pip install serpapi google-search-results openai langchain streamlit
!pip install pyngrok
```

Refer to the below for additional setup instructions.

## Model Training Setup

* Pre-training data gathering:

You will need to collect all the prompts and the suggested APIs to be contacted. The prompts and APIs have to be saved in JSON format. The training files need to be split into training and validation.Sample training and validation files that were used in the project are available in the /training folder.

The data used in training have to be created in a predifined format, and more details can be obtained on the training JSON file format here - https://gorilla.cs.berkeley.edu/blogs/5_how_to_gorilla.html#train-your-own-gorilla

* Importing required libraries:

The training requires the following libraries. Use the following commands to install them:
```
!pip install transformers==4.28.1
!pip install huggingface-hub==0.14.1
!pip install torch==2.0.1
!pip install tqdm==4.65.0
!pip install prompt_toolkit==3.0.38
!pip install sentencepiece==0.1.99
!pip install accelerate==0.19.0
!pip install einops==0.7.0
!pip install bitsandbytes peft flask
```

* Configuring fine-tuning parameters for optimization:

Below are the parameters we have used in our project. You can update these training parameters and train:
```
# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="steps",
    eval_steps=100,
    logging_dir=LOG_DIR,
    logging_steps=10,
    per_device_train_batch_size=1,
    learning_rate=5e-5,
    num_train_epochs=1,
    save_strategy="epoch",
    save_total_limit=2,
    weight_decay=0.01,
    report_to="none",
)
```

## Model Training:

Before training the model, make sure that you have the right GPU procured. We have used A100. Training with 500 prompts takes approximately 10 minutes.

Furthermore, we have used 8-bit quantized data for training:
```
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    load_in_8bit=True,
    torch_dtype=torch.float16,
)
model.resize_token_embeddings(len(tokenizer))
```

## API Inferencing with Training Weights
In `LLM_Training.ipynb`, steps for using the trained model for inference include:
1. Loading the trained weights.
2. Configuring the inference pipeline.
3. Testing the model's output with sample inputs.

* Hosting API for enabling trained model reference:

We have provided grok code to enable referencing to the trained model. This can be leveraged in case you want to create a reference to a trained model for multiple users.

## API Inferencing with GoEx
The `AutonomousLLM.ipynb` file provides comprehensive details for integrating GoEx with the trained LLM. Steps include:
1. Setting up GoEx runtime.
2. Using the fine-tuned model for API responses.
3. Processing inputs and generating outputs autonomously.

Refere to https://github.com/ShishirPatil/gorilla/blob/main/openfunctions/README.md for additional details. 

## Contacting API
In `AutonomousLLM.ipynb`, API integration steps include:
1. Configuring API keys and endpoints.
2. Sending requests using secure methods.
3. Parsing API responses for downstream tasks.

## Identification of Action
This section in `AutonomousLLM.ipynb` details how the system identifies required actions from parsed data:
1. Using the model to classify or determine actions based on inputs.
2. Validating and logging identified actions.

In addtion to the basic system action identification, the gorilla execution function callign feature can also be used. This is an out of the box feature that can be used, and more details are available in the linke below.

* https://gorilla.cs.berkeley.edu/blogs/4_open_functions.html
* https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard
* https://gorilla.cs.berkeley.edu/leaderboard
* https://colab.research.google.com/drive/16M5J2H9F8YQora_W2PDnp120slZH-Mqd?usp=sharing
* https://github.com/ShishirPatil/gorilla/tree/main/openfunctions

## Identify System
The `AutonomousLLM.ipynb` file explains how to determine the target system for the identified action:
1. Mapping actions to system capabilities.
2. Using GoEx workflows to route actions appropriately.

The system identification code will help identify the system. It references the actions.csv file that is available in \ptompts folder. 

The key function is below.
```
def generate_prompt_for_action(csv_file):
    try:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(csv_file)

        # Check if the required columns exist in the DataFrame
        if 'action' not in df.columns or 'system' not in df.columns:
            raise ValueError("CSV file must contain 'action' and 'system' columns.")

        # Start building the prompt string
        prompt_for_action = ("Identify action to take and the systems to update. The action item and systems pairs are as follows:\n")

        # Iterate through the rows of the DataFrame and add action-system pairs to the prompt string
        for _, row in df.iterrows():
            prompt_for_action += f"{row['action']} : {row['system']}\n"

        return prompt_for_action.strip()

    except Exception as e:
        return str(e)
```

## Converting Action to System Request
Detailed in `AutonomousLLM.ipynb`:
1. Translating model outputs into system-understandable requests.
2. Formatting requests to meet API or system specifications.

This action is enabled with the prompt_for_system_update variable. This can be edited for further improving the fuctionality of the project. The prompt used in the project is provided below for reference. 
```
prompt_for_system_update = (
    "Convert user-provided description action into a system request. "
    "The description action and system request pairs are as follows:\n"
    "description action contains Amazon:\n"
    "{\n"
    "  'action': 'BUY',\n"
    "  'product': {\n"
    "    'id': 'SG12345',\n"
    "    'name': 'Samsung Galaxy S23',\n"
    "    'variant': '128GB, Phantom Black',\n"
    "    'quantity': 1\n"
    "  },\n"
    "  'buyer': {\n"
    "    'id': 'B10001',\n"
    "    'name': 'John Doe',\n"
    "    'contact': 'john.doe@example.com',\n"
    "    'phone': '+1234567890'\n"
    "  },\n"
    "  'shipping': {\n"
    "    'address': '123 Main Street, Apartment 45B, New York, NY, 10001, USA'\n"
    "  },\n"
    "  'payment': {\n"
    "    'method': 'CreditCard',\n"
    "    'card': {\n"
    "      'number': '4111111111111111',\n"
    "      'expiry': '12/26',\n"
    "      'cvv': '123'\n"
    "    },\n"
    "    'billingAddress': '123 Main Street, Apartment 45B, New York, NY, 10001, USA'\n"
    "  },\n"
    "  'order': {\n"
    "    'subtotal': 799.99,\n"
    "    'tax': 64.00,\n"
    "    'shippingFee': 15.00,\n"
    "    'total': 878.99\n"
    "  }\n"
    "}"
    "description action contains Robinhood:\n"
    "{\n"
    "  \"account\": \"https://api.robinhood.com/accounts/XXXXAAAA/\",\n"
    "  \"instrument\": \"https://api.robinhood.com/instruments/39ff611b-84e7-425b-bfb8-6fe2a983fcf3/\",\n"
    "  \"symbol\": \"AAPL\",\n"
    "  \"type\": \"market\",\n"
    "  \"time_in_force\": \"gfd\",\n"
    "  \"trigger\": \"immediate\",\n"
    "  \"price\": \"150.00\",\n"
    "  \"quantity\": 1,\n"
    "  \"side\": \"buy\"\n"
    "}"
    "description action contains Slack:\n"
    "{\n"
    "  \"channel\": \"C1234567890\",\n"
    "  \"text\": \"Hello, team! This is a message from a validated user.\",\n"
    "  \"as_user\": true\n"
    "}"
    "description action contains HubSpot:\n"
    "{\n"
    "  \"emails\": [\n"
    "    \"user1@example.com\",\n"
    "    \"user2@example.com\",\n"
    "    \"user3@example.com\"\n"
    "  ],\n"
    "  \"subject\": \"Important Announcement\",\n"
    "  \"content\": {\n"
    "    \"type\": \"html\",\n"
    "    \"value\": \"<p>Hello,</p><p>This is an important update for all users. Please review it at your earliest convenience.</p><p>Best regards,</p><p>Your Team</p>\"\n"
    "  },\n"
    "  \"from\": {\n"
    "    \"email\": \"yourteam@example.com\",\n"
    "    \"name\": \"Your Team\"\n"
    "  }\n"
    "}"
)
```


## Sending System Request to Application
The final integration step in `AutonomousLLM.ipynb` includes:
1. Sending the formatted request to the target application.
2. Handling system responses and ensuring proper logging.

We have created Application Stubs for this project. Sample code for reaching out to application stubs is provided below. This can also be leveraged to contact custom application with python code. 

```
import json

def clean_json(json_data):
    # If the input is a dictionary, convert it to a string
    if isinstance(json_data, dict):
        json_string = json.dumps(json_data)
    elif isinstance(json_data, str):
        json_string = json_data
    else:
        raise ValueError("Input must be a JSON object (dict) or string")

    # Remove newline characters
    return json_string.replace("\n", "")

from flask import Flask, render_template_string
from google.colab.output import eval_js

# Initialize Flask app
app = Flask(__name__)

# Predefined JSON data
data = clean_json(system_action_uc1_level3)

@app.route('/')
def home():
    return '<h1>Welcome to the Flask App</h1><p>Go to <a href="/json">JSON Data Page</a> to view the JSON.</p>'

@app.route('/json')
def display_json():
    from json import dumps
    #formatted_json = dumps(data, indent=4)
    formatted_json = json.dumps(data, indent=4, ensure_ascii=False)  # Indented JSON
    return render_template_string(json_template, json_data=formatted_json)

# Start Flask app and use Colab Proxy
if __name__ == "__main__":
    # Get the Colab Proxy URL for port 5000
    proxy_url = eval_js("google.colab.kernel.proxyPort(5000)")
    print(f" * Colab Proxy URL: {proxy_url}")
    app.run(port=5000)

```

## Logging Action
The system actions can be logged for future reference or for post facto validation. The code snip used in the project is below. The logs are created as JSON file, to enable easier reading and analysis by other systems.

```
import logging
import json

# Define log directory and file
log_directory = "/content/drive/My Drive/W6998-DL/Project2/logs"
log_filename = "autonomous_actions.log"
log_file_path = os.path.join(log_directory, log_filename)

# Create the log directory if it doesn't exist
os.makedirs(log_directory, exist_ok=True)

# Ensure the log file exists
if not os.path.isfile(log_file_path):
    # Create an empty log file
    with open(log_file_path, "w") as log_file:
        log_file.write("")  # Create an empty file

# Configure logging
logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True  # Ensure configuration applies even if logging was configured before
)

# Log the action
log_entry = {
    "action": system_action_to_take_uc4,
    "status": "success",
    "response": system_action_uc1_level4
}

logging.info("System action logged: %s", json.dumps(log_entry))

# Flush the logs to ensure they are written to the file
logging.shutdown()

# Verify the log file exists and display its location
if os.path.isfile(log_file_path):
    print(f"Log file saved at: {log_file_path}")

# Display log contents
with open(log_file_path, "r") as log_file:
    print(log_file.read())

```
