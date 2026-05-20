import torch
from torch.utils.data import DataLoader
from ml.llm.model import GPTModel
from ml.llm.data import SimpleTokenizerV2, build_vocab, GPTDatasetV1

# -----------------------------
# LOAD DATA
# -----------------------------
with open("ml/llm/data.txt", "r", encoding="utf-8") as f:
    text = f.read()

vocab = build_vocab(text)
tokenizer = SimpleTokenizerV2(vocab)

dataset = GPTDatasetV1(text, tokenizer, max_length=16, stride=8)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

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

model = GPTModel(cfg).to(device)

# -----------------------------
# LOSS FUNCTION
# -----------------------------
def calc_loss_batch(input_batch, target_batch):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        target_batch.view(-1)
    )
    return loss

# -----------------------------
# OPTIMIZER
# -----------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# -----------------------------
# TRAIN LOOP
# -----------------------------
epochs = 10

for epoch in range(epochs):
    total_loss = 0

    for input_batch, target_batch in dataloader:
        optimizer.zero_grad()

        loss = calc_loss_batch(input_batch, target_batch)
        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# -----------------------------
# SAVE MODEL
# -----------------------------
torch.save(model.state_dict(), "ml/llm/model.pth")
print("Model saved!")