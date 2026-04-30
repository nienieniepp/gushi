from flask import Flask, render_template, request
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

MODEL_DIR = "../models/gpt2-poem"


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device


try:
    TOKENIZER, MODEL, DEVICE = load_model()
    MODEL_READY = True
except Exception:
    TOKENIZER, MODEL, DEVICE = None, None, "cpu"
    MODEL_READY = False


@app.route("/", methods=["GET", "POST"])
def index():
    generated = ""
    prompt = ""
    if request.method == "POST" and MODEL_READY:
        prompt = request.form.get("prompt", "春江花月夜")
        inputs = TOKENIZER(prompt, return_tensors="pt").to(DEVICE)
        outputs = MODEL.generate(
            **inputs,
            do_sample=True,
            max_new_tokens=64,
            temperature=0.9,
            top_p=0.95,
            pad_token_id=TOKENIZER.eos_token_id,
        )
        generated = TOKENIZER.decode(outputs[0], skip_special_tokens=True)
    return render_template("index.html", generated=generated, prompt=prompt, model_ready=MODEL_READY)


if __name__ == "__main__":
    app.run(debug=True)
