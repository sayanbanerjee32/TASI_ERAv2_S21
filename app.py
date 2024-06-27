import gradio as gr
import os
import tiktoken

import torch
import torch.nn as nn
import torch.nn.functional as F
from model_gpt2 import GPT, GPTConfig

from huggingface_hub import hf_hub_download
import joblib

REPO_ID = "sayanbanerjee32/nanogpt2_test"
data_file = "data/meta.pkl"
model_file = "saved_model/ckpt.pt"

# load model
model_spec = torch.load(
    hf_hub_download(repo_id=REPO_ID, filename=model_file),
    map_location=torch.device('cpu'))
model_args = model_spec['model_args']
model_weights = model_spec['model']
modelconf = GPTConfig(**model_args)
trained_model = GPT(modelconf)
trained_model.load_state_dict(model_weights)

# import the encoder and decoder from tiktoken
enc = tiktoken.get_encoding("gpt2")

def generate_text(seed_text, max_new_tokens, temperature, top_k = None):
    text = seed_text if seed_text is not None else " "
    text = text if text.endswith(" ") else seed_text + " "
    context = torch.tensor(enc.encode(text), dtype=torch.long).unsqueeze(0)
    temperature = temperature if temperature > 0 else 1e-5
    top_k = top_k if top_k is None or top_k > 0 else None
    return enc.decode(trained_model.generate(context, temperature = temperature, top_k = top_k, max_new_tokens=max_new_tokens)[0].tolist())



with gr.Blocks() as demo:
    gr.HTML("<h1 align = 'center'> Text Generator </h1>")
    gr.HTML("<h4 align = 'center'> Generate text in the style of William Shakespeare based on an intial text</h4>")
    
    content = gr.Textbox(label = "Enter seed text to generate content")
    with gr.Row():
        mtk = gr.Number(label = "Max tokens to generate", value = 100)
        tmp = gr.Slider(label = "Temparature (use higher value for higher creativity)", minimum = 0.0, maximum= 1.0,value = 0.7)
        tn = gr.Number(label="Select Top N in each step (Optional)", value = None)
    inputs = [
                content,
                mtk, tmp, tn
              ]
    generate_btn = gr.Button(value = 'Generate')
    outputs  = [gr.Textbox(label = "Generated text")]
    generate_btn.click(fn = generate_text, inputs= inputs, outputs = outputs)

# for collab
# demo.launch(debug=True) 

if __name__ == '__main__':
    demo.launch() 
