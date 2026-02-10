"""
Lucky-Seed · DL Classifier (PyTorch)
──────────────────────────────────────────────────────────────
Architecture: Character-Embedding → BiLSTM → Self-Attention → FC

- 입력: 한국어 미션 텍스트 (문자 단위 토크나이징)
- 출력: 5-class softmax (건강/마음챙김/생산성/관계/자기성장)
- 해석: Attention weight → 어떤 문자 위치에 집중했는지 시각화
"""

import os
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import warnings
warnings.filterwarnings("ignore")

LABEL_NAMES = ["건강", "마음챙김", "생산성", "관계", "자기성장"]
MAX_SEQ_LEN = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Tokenizer ─────────────────────────────────────────────────────────────

class CharTokenizer:
    """문자 단위 토크나이저 (특수 토큰 포함)."""
    PAD, UNK, PAD_ID, UNK_ID = "<PAD>", "<UNK>", 0, 1

    def __init__(self):
        self.char2id: dict = {self.PAD: 0, self.UNK: 1}
        self.id2char: dict = {0: self.PAD, 1: self.UNK}
        self.vocab_size: int = 2

    def fit(self, texts: list[str]) -> "CharTokenizer":
        chars = set("".join(texts))
        for ch in sorted(chars):
            if ch not in self.char2id:
                idx = len(self.char2id)
                self.char2id[ch] = idx
                self.id2char[idx] = ch
        self.vocab_size = len(self.char2id)
        return self

    def encode(self, text: str, max_len: int = MAX_SEQ_LEN) -> list[int]:
        ids = [self.char2id.get(ch, self.UNK_ID) for ch in text]
        # padding / truncation
        if len(ids) >= max_len:
            return ids[:max_len]
        return ids + [self.PAD_ID] * (max_len - len(ids))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"char2id": self.char2id}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok = cls()
        tok.char2id = data["char2id"]
        tok.id2char = {int(v): k for k, v in tok.char2id.items()}
        tok.vocab_size = len(tok.char2id)
        return tok


# ── Dataset ───────────────────────────────────────────────────────────────

class MissionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer: CharTokenizer, max_len=MAX_SEQ_LEN):
        self.encodings = [tokenizer.encode(t, max_len) for t in texts]
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.encodings[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


# ── Self-Attention Module ──────────────────────────────────────────────────

class SelfAttention(nn.Module):
    """
    Scaled Dot-Product Self-Attention.
    반환: (context_vector, attention_weights)
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key   = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = math.sqrt(hidden_dim)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor | None = None):
        # hidden: (B, T, H)
        Q = self.query(hidden)
        K = self.key(hidden)
        V = self.value(hidden)

        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # (B, T, T)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)                # (B, T, T)

        # mean-pool over token dimension → (B, H)
        context = torch.bmm(attn_weights, V).mean(dim=1)
        return context, attn_weights


# ── BiLSTM + Attention Model ──────────────────────────────────────────────

class BiLSTMAttention(nn.Module):
    """
    아키텍처:
      Embedding → BiLSTM(2-layer) → Self-Attention → LayerNorm
      → FC(hidden→hidden//2) → GELU → Dropout → FC(→num_classes)
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 5,
        dropout: float = 0.3,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.embed_dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out_dim = hidden_dim * 2  # bidirectional
        self.attention = SelfAttention(lstm_out_dim)
        self.layer_norm = nn.LayerNorm(lstm_out_dim)

        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, input_ids: torch.Tensor):
        # PAD mask: True where PAD token
        mask = (input_ids == self.pad_id)  # (B, T)

        x = self.embed_dropout(self.embedding(input_ids))  # (B, T, E)
        lstm_out, _ = self.lstm(x)                          # (B, T, 2H)
        context, attn_weights = self.attention(lstm_out, mask)  # (B, 2H)
        context = self.layer_norm(context)
        logits = self.classifier(context)                   # (B, C)
        return logits, attn_weights


# ── Training Loop ─────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * y.size(0)
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(y.cpu().tolist())
    n = len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return total_loss / n, acc, f1, all_labels, all_preds


def train_dl_model(
    df,
    model_dir: str = "saved_models",
    tokenizer_path: str = "saved_models/tokenizer.json",
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    os.makedirs(model_dir, exist_ok=True)

    X = df["text"].tolist()
    y = df["label"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # 토크나이저
    tokenizer = CharTokenizer().fit(X_train)
    tokenizer.save(tokenizer_path)
    print(f"[Tokenizer] vocab_size={tokenizer.vocab_size}")

    train_ds = MissionDataset(X_train, y_train, tokenizer)
    test_ds  = MissionDataset(X_test,  y_test,  tokenizer)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size)

    # 모델
    model = BiLSTMAttention(
        vocab_size=tokenizer.vocab_size,
        embed_dim=64,
        hidden_dim=128,
        num_layers=2,
        num_classes=5,
        dropout=0.3,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Trainable params: {total_params:,}")

    # 클래스 불균형 처리
    from collections import Counter
    counts = Counter(y_train)
    weights = torch.tensor(
        [1.0 / counts[i] for i in range(5)], dtype=torch.float
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_f1, best_state = 0.0, None
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}

    print(f"\n── DL 학습 시작 (device={DEVICE}, epochs={epochs}) ──")
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc, val_f1, _, _ = eval_epoch(model, test_loader, criterion, DEVICE)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{epochs} │ "
                f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} │ "
                f"val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
            )

    # 최적 모델 로드 & 최종 평가
    model.load_state_dict(best_state)
    _, final_acc, final_f1, y_true, y_pred = eval_epoch(model, test_loader, criterion, DEVICE)
    print(f"\n[Best] Accuracy={final_acc:.4f}  F1={final_f1:.4f}")
    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES))

    # 저장
    model_path = os.path.join(model_dir, "bilstm_attention.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "vocab_size": tokenizer.vocab_size,
                "embed_dim": 64,
                "hidden_dim": 128,
                "num_layers": 2,
                "num_classes": 5,
                "dropout": 0.3,
            },
        },
        model_path,
    )
    print(f"[Saved] {model_path}")

    return {
        "accuracy": final_acc,
        "f1_weighted": final_f1,
        "history": history,
        "model": model,
        "tokenizer": tokenizer,
        "y_test": y_true,
        "y_pred": y_pred,
    }


# ── 추론 인터페이스 ────────────────────────────────────────────────────────

class DLPredictor:
    """학습된 DL 모델로 단일 텍스트 예측 + Attention 반환."""

    def __init__(
        self,
        model_path: str = "saved_models/bilstm_attention.pth",
        tokenizer_path: str = "saved_models/tokenizer.json",
    ):
        checkpoint = torch.load(model_path, map_location="cpu")
        cfg = checkpoint["config"]
        self.model = BiLSTMAttention(**cfg)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.tokenizer = CharTokenizer.load(tokenizer_path)
        self.label_names = LABEL_NAMES

    @torch.no_grad()
    def predict(self, text: str) -> dict:
        ids = self.tokenizer.encode(text, MAX_SEQ_LEN)
        x = torch.tensor([ids], dtype=torch.long)
        logits, attn = self.model(x)
        proba = F.softmax(logits, dim=-1)[0].cpu().numpy()
        pred = int(proba.argmax())

        # Attention: mean over (T×T) → (T,) for first sample
        # attn: (1, T, T)
        attn_scores = attn[0].mean(dim=0).cpu().numpy()  # (T,)
        chars = list(text[:MAX_SEQ_LEN])

        return {
            "predicted_label": pred,
            "predicted_category": self.label_names[pred],
            "probabilities": {
                cat: float(p) for cat, p in zip(self.label_names, proba)
            },
            "attention": {
                "chars": chars,
                "scores": attn_scores[: len(chars)].tolist(),
            },
        }


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("data/mission_dataset.csv")
    results = train_dl_model(df, epochs=30)
