from fastapi import FastAPI, Request
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

app = FastAPI()

# Load model and tokenizer at startup
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=128)

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")
    if not prompt:
        return {"error": "No prompt provided"}
    result = generator(prompt, max_new_tokens=64, do_sample=True, temperature=0.7)
    return {"response": result[0]['generated_text'][len(prompt):].strip()}
