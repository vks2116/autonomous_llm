# README

## Get Started
Refer to the attached files to set up your environment and prepare for the implementation. The provided scripts detail the basic initialization steps and outline the resources necessary to begin.

## Install Dependencies
Install Dependencies

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
Model Training Setup

Pre-training data gathering:

You will need to collect all the prompts and the suggested APIs to be contacted.

The prompts and APIs have to be saved in JSON format. The training files need to be split into training and validation.

Sample training and validation files that were used in the project are available in the /training folder.

Importing required libraries:

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
Configuring fine-tuning parameters for optimization:

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

Model Training:

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

Hosting API for enabling trained model reference:

We have provided grok code to enable referencing to the trained model. This can be leveraged in case you want to create a reference to a trained model for multiple users.

## API Inferencing with GoEx
The `AutonomousLLM.ipynb` file provides comprehensive details for integrating GoEx with the trained LLM. Steps include:
1. Setting up GoEx runtime.
2. Using the fine-tuned model for API responses.
3. Processing inputs and generating outputs autonomously.

## Contacting API
In `AutonomousLLM.ipynb`, API integration steps include:
1. Configuring API keys and endpoints.
2. Sending requests using secure methods.
3. Parsing API responses for downstream tasks.

## Identification of Action
This section in `AutonomousLLM.ipynb` details how the system identifies required actions from parsed data:
1. Using the model to classify or determine actions based on inputs.
2. Validating and logging identified actions.

## Identify System
The `AutonomousLLM.ipynb` file explains how to determine the target system for the identified action:
1. Mapping actions to system capabilities.
2. Using GoEx workflows to route actions appropriately.

## Converting Action to System Request
Detailed in `AutonomousLLM.ipynb`:
1. Translating model outputs into system-understandable requests.
2. Formatting requests to meet API or system specifications.

## Sending System Request to Application
The final integration step in `AutonomousLLM.ipynb` includes:
1. Sending the formatted request to the target application.
2. Handling system responses and ensuring proper logging.

For further details, refer to the attached notebook files.

