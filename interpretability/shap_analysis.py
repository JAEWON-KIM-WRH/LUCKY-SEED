"""
Lucky-Seed · Model Interpretability
─────────────────────────────────────────────────────────────────────
1. SHAP: ML(LogisticRegression) 모델 → 어떤 문자 패턴이 분류를 결정하는가
2. Attention Visualization: DL(BiLSTM+Attention) → 어떤 위치에 집중했는가
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

LABEL_NAMES = ["건강", "마음챙김", "생산성", "관계", "자기성장"]
LABEL_COLORS = {
    "건강":    "#FF6B6B",
    "마음챙김": "#4ECDC4",
    "생산성":  "#45B7D1",
    "관계":    "#96CEB4",
    "자기성장": "#FFEAA7",
}


# ── SHAP 분석 ────────────────────────────────────────────────────────────

def run_shap_analysis(
    model_path: str = "saved_models/logisticregression_model.pkl",
    data_path:  str = "data/mission_dataset.csv",
    output_dir: str = "assets",
    n_samples:  int = 100,
) -> None:
    """SHAP 값 계산 및 시각화 저장."""
    try:
        import shap
    except ImportError:
        print("[SHAP] shap 라이브러리가 없습니다. pip install shap")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 모델 & 데이터 로드
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)

    df = pd.read_csv(data_path)
    sample_df = df.sample(n=min(n_samples, len(df)), random_state=42)

    # TF-IDF 변환
    tfidf = pipeline.named_steps["tfidf"]
    clf   = pipeline.named_steps["clf"]
    X_tfidf = tfidf.transform(sample_df["text"].tolist())

    # Linear 모델용 LinearExplainer
    explainer = shap.LinearExplainer(clf, X_tfidf, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_tfidf)  # list of arrays (one per class)

    feature_names = tfidf.get_feature_names_out()

    # ── 1. Summary Bar Plot (클래스별 상위 Feature) ──────────────────────
    fig, axes = plt.subplots(1, len(LABEL_NAMES), figsize=(20, 5))
    for i, (ax, label) in enumerate(zip(axes, LABEL_NAMES)):
        # 해당 클래스의 SHAP mean abs
        vals = np.array(shap_values[i] if isinstance(shap_values, list) else shap_values[:, :, i])
        mean_abs = np.abs(vals).mean(axis=0)

        top_idx = mean_abs.argsort()[-15:][::-1]
        top_features = feature_names[top_idx]
        top_values   = mean_abs[top_idx]

        color = LABEL_COLORS.get(label, "#888888")
        ax.barh(range(len(top_features)), top_values[::-1], color=color, alpha=0.85)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features[::-1], fontsize=8)
        ax.set_title(f"{label}", fontsize=11)
        ax.set_xlabel("Mean |SHAP|")

    plt.suptitle("SHAP Feature Importance (Top-15 per Category)", fontsize=13, y=1.02)
    plt.tight_layout()
    shap_bar_path = os.path.join(output_dir, "shap_summary_bar.png")
    plt.savefig(shap_bar_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] Saved → {shap_bar_path}")

    # ── 2. 개별 예측 설명 (워터폴 스타일) ───────────────────────────────
    _plot_waterfall_examples(
        shap_values, feature_names, sample_df, output_dir
    )


def _plot_waterfall_examples(shap_values, feature_names, sample_df, output_dir):
    """카테고리별 대표 예시 1개씩 워터폴 플롯."""
    fig, axes = plt.subplots(1, len(LABEL_NAMES), figsize=(22, 6))

    for class_idx, (ax, label) in enumerate(zip(axes, LABEL_NAMES)):
        class_samples = sample_df[sample_df["label"] == class_idx]
        if class_samples.empty:
            ax.set_visible(False)
            continue

        sample_row = class_samples.iloc[0]
        sv = shap_values[class_idx] if isinstance(shap_values, list) else shap_values[:, :, class_idx]
        sample_sv = sv[class_samples.index[0] - sample_df.index[0]]

        # 상위 10 SHAP features
        top_idx = np.abs(sample_sv).argsort()[-10:][::-1]
        top_features = feature_names[top_idx]
        top_vals     = sample_sv[top_idx]

        colors = ["#FF4B4B" if v > 0 else "#4B9CFF" for v in top_vals]
        ax.barh(range(len(top_features)), top_vals[::-1], color=colors[::-1], alpha=0.85)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features[::-1], fontsize=8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"[{label}]\n\"{sample_row['text'][:20]}...\"", fontsize=9)
        ax.set_xlabel("SHAP value")

    red_patch  = mpatches.Patch(color="#FF4B4B", label="분류 확률 ↑")
    blue_patch = mpatches.Patch(color="#4B9CFF", label="분류 확률 ↓")
    fig.legend(handles=[red_patch, blue_patch], loc="upper right", fontsize=9)
    plt.suptitle("SHAP Waterfall - 카테고리별 대표 예시", fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, "shap_waterfall.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] Saved → {path}")


# ── Attention 시각화 ─────────────────────────────────────────────────────

def visualize_attention(
    text: str,
    attn_scores: list[float],
    chars: list[str],
    predicted_category: str,
    save_path: str = None,
    ax: plt.Axes = None,
) -> plt.Figure | None:
    """
    Attention weight를 히트맵으로 시각화.
    ax가 주어지면 기존 axes에 그림. 아니면 새 Figure 반환.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(max(6, len(chars) * 0.35), 2))
    else:
        fig = ax.get_figure()

    scores = np.array(attn_scores)
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    color = LABEL_COLORS.get(predicted_category, "#888888")

    for i, (ch, score) in enumerate(zip(chars, scores)):
        rect = mpatches.FancyBboxPatch(
            (i, 0), 0.9, 0.9,
            boxstyle="round,pad=0.05",
            facecolor=color,
            alpha=float(score) * 0.8 + 0.15,
        )
        ax.add_patch(rect)
        ax.text(
            i + 0.45, 0.45, ch,
            ha="center", va="center", fontsize=10,
            color="black",
        )

    ax.set_xlim(-0.1, len(chars))
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title(
        f"Attention Visualization → 예측: [{predicted_category}]",
        fontsize=10,
    )

    if standalone:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=120, bbox_inches="tight")
            print(f"[Attention] Saved → {save_path}")
        return fig
    return None


def run_attention_examples(
    model_path: str = "saved_models/bilstm_attention.pth",
    tokenizer_path: str = "saved_models/tokenizer.json",
    output_dir: str = "assets",
) -> None:
    """DL 모델의 Attention 시각화 예시 저장."""
    try:
        from models.dl_classifier import DLPredictor
    except ImportError:
        print("[Attention] DL 모델 로드 실패")
        return

    os.makedirs(output_dir, exist_ok=True)

    predictor = DLPredictor(model_path, tokenizer_path)

    examples = [
        "오늘 30분 달리기 완료",
        "명상 5분 하기",
        "할일 목록 우선순위 정리",
        "친구에게 먼저 연락하기",
        "책 30페이지 읽기",
    ]

    fig, axes = plt.subplots(len(examples), 1, figsize=(14, len(examples) * 2.2))
    for i, (example, ax) in enumerate(zip(examples, axes)):
        result = predictor.predict(example)
        attn = result["attention"]
        visualize_attention(
            example,
            attn["scores"],
            attn["chars"],
            result["predicted_category"],
            ax=ax,
        )

    plt.suptitle("BiLSTM+Attention - 문자 단위 Attention 가중치", fontsize=12, y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, "attention_examples.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[Attention] Saved → {path}")


if __name__ == "__main__":
    run_shap_analysis()
    run_attention_examples()
