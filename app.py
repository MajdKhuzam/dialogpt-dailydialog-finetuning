from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch, uuid
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODEL_PATH = os.path.abspath(os.path.join(__file__, '..', 'output', 'DialoGPT-final'))
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
model.eval()

sessions = {}  # session_id -> chat_history_ids tensor

class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.post("/chat")
def chat(req: ChatRequest):
    new_input = tokenizer.encode(req.message + tokenizer.eos_token, return_tensors="pt")
    history = sessions.get(req.session_id)
    bot_input = torch.cat([history, new_input], dim=-1) if history is not None else new_input
    output = model.generate(bot_input, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id,
                            do_sample=True, top_k=50, top_p=0.95, temperature=0.75, no_repeat_ngram_size=3)
    sessions[req.session_id] = output
    response = tokenizer.decode(output[:, bot_input.shape[-1]:][0], skip_special_tokens=True)
    return {"response": response}

@app.get("/new_session")
def new_session():
    return {"session_id": str(uuid.uuid4())}

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)