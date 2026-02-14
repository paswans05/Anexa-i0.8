import gradio as gr
import torch
import os
import time
from anexagpt import GPTLanguageModel, encode, decode, device, itos

from huggingface_hub import hf_hub_download

# Load Model
print("Loading Anexa...")
model = GPTLanguageModel()

model_path = 'model.safetensors'
if not os.path.exists(model_path):
    print(f"Downloading {model_path} from Hugging Face...")
    try:
        model_path = hf_hub_download(repo_id="aarnavrexwal05/Anexa-v0.8-Thinking", filename="model.safetensors")
    except Exception as e:
        print(f"Could not download model: {e}")

if os.path.exists(model_path):
    from safetensors.torch import load_file
    state_dict = load_file(model_path)
    model.load_state_dict(state_dict)

model.to(device)
model.eval()

# ANSI removal for web
def clean_ansi(text):
    import re
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def predict(message, history):
    # Prepare input
    # Context window handling omitted for simplicity in demo
    context_idxs = encode(message)
    context = torch.tensor(context_idxs, dtype=torch.long, device=device).unsqueeze(0)
    
    # Stream generation
    buffer = ""
    is_thinking = False
    
    # HTML for output
    output_html = ""
    
    for idx_next in model.generate_stream(context, max_new_tokens=200):
        token_val = idx_next.item()
        char = itos[token_val]
        buffer += char
        
        # Thinking logic for visualization
        if "<think>" in buffer and not is_thinking:
            is_thinking = True
            output_html += "<span style='color: grey;'>"
            
        if "</think>" in buffer and is_thinking:
            is_thinking = False
            output_html += "</span>"
            
        # Add char to output
        # Escape HTML if needed, but simple char addition for now
        if char == "\n":
             output_html += "<br>"
        else:
             output_html += char
             
        yield output_html

# Create Interface
demo = gr.ChatInterface(
    predict,
    title="Anexa v0.8 (Thinking Mode)",
    description="A 0.8M parameter model trained on TinyStories + Reasoning Data. It can 'think' before answering!",
    examples=["Hello", "3+3=?", "Once upon a time"]
)

if __name__ == "__main__":
    demo.launch()
