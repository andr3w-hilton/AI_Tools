# This is a just a PoC for the 13B model. It's not meant to be used for anything serious.
# Adding it to the repo for testing purposes.

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("TheBloke/Wizard-Vicuna-13B-Uncensored-HF")
model = AutoModelForCausalLM.from_pretrained("TheBloke/Wizard-Vicuna-13B-Uncensored-HF")

# Chat loop
while True:
    # Get user input and tokenize
    text = input("You: ")
    input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors='pt')

    # Generate response
    outputs = model.generate(input_ids, max_length=100, do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Print response
    print("Claude: " + response)