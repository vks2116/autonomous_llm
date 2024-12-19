# README

## Get Started
Refer to the attached files to set up your environment and prepare for the implementation. The provided scripts detail the basic initialization steps and outline the resources necessary to begin.

## Install Dependencies
Install Dependencies

The installation process includes all required Python libraries and frameworks. Use the following commands to install dependencies:
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

Refer to the below for additional setup instructions.

## Model Training Setup
The `LLM_Training.ipynb` file provides detailed steps for setting up the training environment. This includes:
1. Importing required libraries.
2. Loading pre-trained models and datasets.
3. Configuring fine-tuning parameters for optimization.

## Model Training
The training process outlined in `LLM_Training.ipynb` involves:
1. Preparing datasets.
2. Implementing fine-tuning on pre-trained language models.
3. Logging training progress and saving checkpoints.
4. Evaluating model performance using test datasets.

## API Inferencing with Training Weights
In `LLM_Training.ipynb`, steps for using the trained model for inference include:
1. Loading the trained weights.
2. Configuring the inference pipeline.
3. Testing the model's output with sample inputs.

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

