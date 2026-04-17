import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from train import OUTPUT_PATH

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load the fine-tuned model and tokenizer from the saved directory
model_path = os.path.join(OUTPUT_PATH, 'data', 'DialoGPT-final')
print(f"Loading model from {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to(DEVICE)

print("Model loaded successfully! You can start chatting. Type 'quit' to exit.\n")

# Initialize the chat history
chat_history_ids = None

# 2. Chat loop
step = 0
while True:
    user_input = input(">> You: ")
    
    # Exit condition
    if user_input.lower() in ['quit', 'exit', 'stop']:
        print("Ending chat. Goodbye!")
        break

    # Encode the user input, append the eos_token, and convert to a PyTorch tensor
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt').to(model.device)

    # Append the new user input to the existing chat history (if this is not the first turn)
    if step > 0:
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
    else:
        bot_input_ids = new_user_input_ids

    # Generate a response using the model
    # We use sampling parameters (temperature, top_p, top_k) to make the responses more natural
    chat_history_ids = model.generate(
        bot_input_ids,
        # max_length=1000,                  # Maximum length of the entire conversation history
        max_new_tokens=100,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,           # Prevents the model from repeating the same phrases
        do_sample=True,                   # Enables random sampling instead of greedy decoding
        top_k=50,                         # Limits sampling to the top 50 probable tokens
        top_p=0.95,                       # Nucleus sampling
        temperature=0.75                  # Controls randomness (higher = more creative, lower = more focused)
    )

    # Extract only the newly generated tokens (the bot's response) from the chat history
    # The response starts right after the length of the input we fed to the model
    bot_response_ids = chat_history_ids[:, bot_input_ids.shape[-1]:]
    
    # Decode the response to a readable string
    response = tokenizer.decode(bot_response_ids[0], skip_special_tokens=True)
    
    print(f"DialoGPT: {response}")
    step += 1