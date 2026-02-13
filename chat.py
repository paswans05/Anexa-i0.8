import torch
import os
import sys
import gc
from safetensors.torch import load_file
from anexagpt import GPTLanguageModel, encode, decode, device, itos

# ANSI colors
GRAY = "\033[90m"
RESET = "\033[0m"

def load_model():
    print(f"Loading Anexa on {device}...")
    model = GPTLanguageModel()
    
    if os.path.exists('model.safetensors'):
        print("Loading weights from model.safetensors...")
        state_dict = load_file('model.safetensors')
        model.load_state_dict(state_dict)
        del state_dict
        gc.collect()
    else:
        print("Warning: No model.safetensors found. Using random weights.")
        
    model.to(device)
    model.eval()
    return model

def chat():
    model = load_model()
    
    print("\n--- Anexa Chat (Thinking Mode) ---")
    print("Type 'quit' to exit.\n")
    
    while True:
        try:
            prompt = input("You: ")
        except EOFError:
            break
            
        if prompt.lower() in ['quit', 'exit']:
            break
        
        if not prompt.strip():
            continue
            
        print("Anexa: ", end="", flush=True)
        
        try:
            # Encode prompt
            context_idxs = encode(prompt)
            context = torch.tensor(context_idxs, dtype=torch.long, device=device).unsqueeze(0)
            
            # State for Thinking Mode
            buffer = ""
            is_thinking = False
            
            # Stream generation
            for idx_next in model.generate_stream(context, max_new_tokens=200):
                token_val = idx_next.item()
                char = itos[token_val]
                
                # Update buffer for tag detection
                buffer += char
                
                # Check for start of thinking
                if "<think>" in buffer and not is_thinking:
                    is_thinking = True
                    print(GRAY, end="", flush=True) 
                    
                # Check for end of thinking
                if "</think>" in buffer and is_thinking:
                    is_thinking = False
                    print(RESET, end="", flush=True)
                
                # Print character
                print(char, end="", flush=True)
                
                # Keep buffer small to save memory
                if len(buffer) > 20:
                    buffer = buffer[-20:]
                    
            print(RESET) # Reset color at the end
            print() # Newline
            
        except KeyError:
            print(f"{RESET}\nError: Your prompt contains characters not in the model's vocabulary.")
        except KeyboardInterrupt:
            print(f"{RESET}\nInterrupted.")
            break

if __name__ == "__main__":
    chat()
