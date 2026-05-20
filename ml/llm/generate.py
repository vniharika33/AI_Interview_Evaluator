import torch
from ml.llm.model import GPTModel, generate_text_simple
from ml.llm.data import SimpleTokenizerV2, build_vocab
# -----------------------------
# LOAD DATA + TOKENIZER
# -----------------------------
with open("ml/llm/data.txt", "r", encoding="utf-8") as f:
    text = f.read()

vocab = build_vocab(text)
tokenizer = SimpleTokenizerV2(vocab)

# -----------------------------
# MODEL CONFIG 
# -----------------------------
cfg = {
    "vocab_size": len(vocab),
    "context_length": 64,
    "emb_dim": 128,
    "n_heads": 4,
    "n_layers": 2,
    "drop_rate": 0.1,
    "qkv_bias": False
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# LOAD MODEL
# -----------------------------
model = GPTModel(cfg)
model.load_state_dict(torch.load("ml/llm/model.pth", map_location=device))
model.to(device)
model.eval()

# -----------------------------
# GENERATE FUNCTION
# -----------------------------
def generate_text(prompt, max_tokens=50):
    # encode input
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)

    # generate
    output_ids = generate_text_simple(
        model=model,
        idx=input_tensor,
        max_new_tokens=max_tokens,
        context_size=cfg["context_length"]
    )

    # decode output
    generated_text = tokenizer.decode(output_ids[0].tolist())
    return generated_text


# -----------------------------
# TEST RUN
# -----------------------------
if __name__ == "__main__":
    prompt = "Generate interview questions on OS"
    output = generate_text(prompt, max_tokens=50)

    print("\nPrompt:", prompt)
    print("\nGenerated:\n", output)