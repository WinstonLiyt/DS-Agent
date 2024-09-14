import os
from functools import partial
import tiktoken
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import openai

# Azure OpenAI setup
chat_api_base = "https://msraopenaieastus.openai.azure.com/"
chat_api_version = "2024-05-01-preview"
credential = DefaultAzureCredential()
token_provider = get_bearer_token_provider(
    credential,
    "https://cognitiveservices.azure.com/.default",
)
chat_client = openai.AzureOpenAI(
    azure_ad_token_provider=token_provider,
    api_version=chat_api_version,
    azure_endpoint=chat_api_base,
)

# OpenAI API key setup
openai.api_key = "FILL IN YOUR KEY HERE."
os.environ["OPENAI_API_KEY"] = openai.api_key

def log_to_file(log_file, prompt, completion, model, max_tokens_to_sample):
    """ Log the prompt and completion to a file."""
    with open(log_file, "a") as f:
        f.write("\n===================prompt=====================\n")
        f.write(prompt)
        f.write(f"\n==================={model} response ({max_tokens_to_sample})=====================\n")
        f.write(completion)
        f.write("\n===================tokens=====================\n")
        f.write("\n\n")

def complete_text_openai(prompt, model="gpt-3.5-turbo", max_tokens_to_sample=1000, temperature=0.5, log_file=None, **kwargs):
    """ Call the OpenAI API to complete a prompt using the new API format."""
    raw_request = {
        "model": model,
        "messages": [prompt],
        "temperature": temperature,
        "max_tokens": max_tokens_to_sample,
        **kwargs
    }
    
    iteration = 0
    completion = None
    while iteration < 10:
        try:
            # Using the new completion API for gpt-3.5 and gpt-4 models
            # response = openai.Completion.create(**raw_request)  # Updated to correct function and parameters
            # completion = response.choices[0].text  # Accessing the 'text' field for completion
            response = chat_client.chat.completions.create(**raw_request)
            completion = response.choices[0].message.content
            break
        except Exception as e:
            iteration += 1
            print(f"===== Retry: {iteration} =====")
            print(f"Error occurs when calling API: {e}")
            continue
    
    if log_file is not None:
        log_to_file(log_file, prompt, completion, model, max_tokens_to_sample)
    return completion

def complete_text(prompt, log_file, model, **kwargs):
    """ Complete text using the specified model with appropriate API. """
    return complete_text_openai(prompt, log_file=log_file, model=model, **kwargs)

def complete_text_fast(prompt, **kwargs):
    return complete_text(prompt = prompt, model ="gpt-4o", temperature =0.01, **kwargs)

# response = complete_text_openai(
#     prompt={'role': 'user', 'content': 'What is the capital of France?'},
#     model="gpt-4o",
#     max_tokens_to_sample=200
# )

# print(response)

