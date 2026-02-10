"""
Lucky-Seed · Train Pipeline
─────────────────────────────────────
ML + DL 모델 통합 학습 스크립트.
실행: python train.py [--epochs 30] [--batch_size 32]
"""

import argparse
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import json
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pathlib import Path

from data.generate_dataset import save_dataset, LABEL_NAMES
from models.ml_classifier import train_and_evaluate
from models.dl_classifier import train_dl_model

# ── 그래프 저장 ───────────────────────────────────────────────────────────

def plot_confusion_matrix(cm, title, save_path, labels=LABEL_NAMES):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
    )
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  Saved → {save_path}")


def plot_training_history(history: dict, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="Train Loss", color="#4C72B0")
    axes[0].plot(history["val_loss"],   label="Val Loss",   color="#DD8452")
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(history["train_acc"], label="Train Acc", color="#4C72B0")
    axes[1].plot(history["val_acc"],   label="Val Acc",   color="#DD8452")
    axes[1].plot(history["val_f1"],    label="Val F1",    color="#55A868", linestyle="--")
    axes[1].set_title("Accuracy & F1")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Score")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.suptitle("BiLSTM+Attention Training History", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_model_comparison(ml_results: dict, dl_result: dict, save_path: str):
    models = list(ml_results.keys()) + ["BiLSTM+Attn"]
    accs   = [ml_results[m]["accuracy"]    for m in ml_results] + [dl_result["accuracy"]]
    f1s    = [ml_results[m]["f1_weighted"] for m in ml_results] + [dl_result["f1_weighted"]]

    x = np.arange(len(models))
    width = 0.35
    colors_acc = ["#4C72B0"] * len(ml_results) + ["#8172B2"]
    colors_f1  = ["#DD8452"] * len(ml_results) + ["#C44E52"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, accs, width, label="Accuracy", color=colors_acc, alpha=0.85)
    bars2 = ax.bar(x + width/2, f1s,  width, label="F1 (weighted)", color=colors_f1, alpha=0.85)

    ax.set_title("ML vs DL 성능 비교", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # 수치 레이블
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    # ML vs DL 경계선
    ax.axvline(x=len(ml_results) - 0.5, color="gray", linestyle="--", alpha=0.5)
    ax.text(len(ml_results) - 0.48, 0.51, "◀ ML  DL ▶", fontsize=9, color="gray")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  Saved → {save_path}")


# ── 메인 ─────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs("assets", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)

    # 1. 데이터셋 준비
    print("\n" + "="*55)
    print("  Step 1: 데이터셋 생성")
    print("="*55)
    save_dataset("data")
    df = pd.read_csv("data/mission_dataset.csv")
    print(f"  총 샘플: {len(df)}")

    # 2. ML 학습
    print("\n" + "="*55)
    print("  Step 2: ML 모델 학습 (scikit-learn)")
    print("="*55)
    ml_results, X_train, X_test, y_train, y_test = train_and_evaluate(df)

    # 혼동 행렬 저장
    for name, res in ml_results.items():
        plot_confusion_matrix(
            res["confusion_matrix"],
            f"{name} - Confusion Matrix",
            f"assets/cm_{name.lower()}.png",
        )

    # 3. DL 학습
    print("\n" + "="*55)
    print("  Step 3: DL 모델 학습 (PyTorch BiLSTM+Attention)")
    print("="*55)
    dl_result = train_dl_model(
        df,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    # DL 혼동 행렬
    cm_dl = confusion_matrix(dl_result["y_test"], dl_result["y_pred"])
    plot_confusion_matrix(cm_dl, "BiLSTM+Attention - Confusion Matrix", "assets/cm_dl.png")

    # DL 학습 곡선
    plot_training_history(dl_result["history"], "assets/dl_training_history.png")

    # 4. 모델 비교 차트
    print("\n" + "="*55)
    print("  Step 4: 모델 비교 시각화")
    print("="*55)
    plot_model_comparison(ml_results, dl_result, "assets/model_comparison.png")

    # 5. 결과 요약 JSON 저장
    summary = {
        "ml_models": {
            name: {
                "accuracy": round(res["accuracy"], 4),
                "f1_weighted": round(res["f1_weighted"], 4),
                "cv_mean": round(res["cv_mean"], 4),
                "cv_std": round(res["cv_std"], 4),
            }
            for name, res in ml_results.items()
        },
        "dl_model": {
            "BiLSTMAttention": {
                "accuracy": round(dl_result["accuracy"], 4),
                "f1_weighted": round(dl_result["f1_weighted"], 4),
            }
        },
    }
    with open("assets/model_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n[Summary]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n✅ 모든 학습 완료! assets/ 폴더에서 결과물을 확인하세요.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lucky-Seed 모델 학습")
    parser.add_argument("--epochs",     type=int, default=30,  help="DL 학습 에포크")
    parser.add_argument("--batch_size", type=int, default=32,  help="배치 크기")
    args = parser.parse_args()
    main(args)
