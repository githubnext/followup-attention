from transformers import GPT2Tokenizer, GPTJModel
import torch

tokenizer = GPT2Tokenizer.from_pretrained("hf-internal-testing/tiny-random-gptj")
model = GPTJModel.from_pretrained("hf-internal-testing/tiny-random-gptj")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state