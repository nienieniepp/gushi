"""Flask Demo: 中文古诗生成（GPT-2 微调模型）"""

from pathlib import Path

import torch
from flask import Flask, render_template, request
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

# 模型目录：默认使用项目中的微调模型目录
MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "gpt2-poem-finetuned"


def load_model(model_dir: Path):
    """加载分词器和模型，供页面请求复用。"""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device


def generate_poem(prompt: str, max_new_tokens: int = 64) -> str:
    """根据用户输入生成古诗文本。"""
    inputs = TOKENIZER(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = MODEL.generate(
            **inputs,
            do_sample=True,
            max_new_tokens=max_new_tokens,
            temperature=0.9,
            top_p=0.95,
            repetition_penalty=1.15,
            pad_token_id=TOKENIZER.eos_token_id,
        )
    return TOKENIZER.decode(outputs[0], skip_special_tokens=True)


# 启动时预加载模型，便于演示时快速响应
try:
    TOKENIZER, MODEL, DEVICE = load_model(MODEL_DIR)
    MODEL_READY = True
    MODEL_ERROR = ""
except Exception as e:  # noqa: BLE001 - 在 demo 场景下显示错误信息方便排查
    TOKENIZER, MODEL, DEVICE = None, None, "cpu"
    MODEL_READY = False
    MODEL_ERROR = str(e)


@app.route("/", methods=["GET", "POST"])
def index():
    prompt = ""
    generated = ""

    if request.method == "POST" and MODEL_READY:
        prompt = request.form.get("prompt", "").strip() or "春风又绿江南岸"
        generated = generate_poem(prompt)

    return render_template(
        "index.html",
        model_ready=MODEL_READY,
        model_dir=str(MODEL_DIR),
        model_error=MODEL_ERROR,
        prompt=prompt,
        generated=generated,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
