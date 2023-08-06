# This is a just a PoC for the 7B model. It's not meant to be used for anything serious.
# Adding it to the repo for testing purposes.

# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from transformers import utils as tf_utils
#
# # Load tokenizer
# tokenizer = AutoTokenizer.from_pretrained("TheBloke/llama2_7b_chat_uncensored-GPTQ")
#
# # Load model
# tf_utils.logging.set_verbosity_error() # reduce logging
# model = AutoModelForCausalLM.from_pretrained("TheBloke/llama2_7b_chat_uncensored-GPTQ", torch_dtype="auto", low_cpu_mem_usage=True)
#
# # Show progress bar
# progress_bar = tf_utils.tqdm(unit="B", unit_scale=True, miniters=1, desc="Downloading model")
# model.save_pretrained('./model', progress_bar=progress_bar)
# # Chat loop
# while True:
#     # Get user input and tokenize
#     text = input("### HUMAN: ")
#     input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors='pt')
#
#     # Generate response
#     outputs = model.generate(input_ids, max_length=100, do_sample=True)
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#
#     # Print response
#     print("### RESPONSE: " + response)


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.autonotebook import tqdm

# This is used to redirect HuggingFace's tqdm to our custom tqdm
from transformers import logging
logging.get_verbosity = lambda: logging.NOTSET

print("Starting the script...")

# Wrap the tqdm around the HuggingFace's loading process
with tqdm(total=2, desc="Loading components", position=0) as progress_bar:

    print("Loading the tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("TheBloke/llama2_7b_chat_uncensored-GPTQ")
    progress_bar.update(1)

    print("Loading the model...")
    model = AutoModelForCausalLM.from_pretrained("TheBloke/llama2_7b_chat_uncensored-GPTQ", torch_dtype="auto", low_cpu_mem_usage=True)
    progress_bar.update(1)

print("Model loaded successfully!")

# Initialize conversational context
context = []

# Chat loop
while True:
    # Get user input
    text = input("### HUMAN: ")

    # Exit condition
    if text.lower() in ['exit', 'quit']:
        break

    # Update context
    context.append(text + tokenizer.eos_token)
    input_text = "".join(context)
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    # Ensure we don't exceed max token limit
    if len(input_ids[0]) > model.config.max_position_embeddings:
        print("Sorry, the conversation is too long. Let's start over.")
        context = []
        continue

    # Generate response
    outputs = model.generate(input_ids, max_length=len(input_ids[0]) + 50, do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Print response and update context
    print("### RESPONSE: " + response.split(tokenizer.eos_token)[-1])
    context.append(response + tokenizer.eos_token)
