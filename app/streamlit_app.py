"""
Lucky-Seed · Streamlit Web Demo
────────────────────────────────────────────────────────────────────
탭 1: 🎰 갓생 가챠   — 미션 텍스트 입력 → 카테고리 예측 → 랜덤 카드+명언 뽑기
탭 2: 🔬 모델 비교   — ML vs DL 예측 비교 & Attention 시각화
탭 3: 📊 성능 리포트 — 학습 결과 시각화 (혼동 행렬, 비교 차트)
────────────────────────────────────────────────────────────────────
실행: streamlit run app/streamlit_app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
import random
import pickle
import base64
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── 페이지 설정 ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="🌱 Lucky-Seed | AI 갓생 가챠",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS 스타일 ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;700&display=swap');
  html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; }

  .hero-title {
    text-align: center; font-size: 2.8rem; font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0;
  }
  .hero-sub {
    text-align: center; color: #888; font-size: 1rem; margin-top: 0.3rem;
  }

  .card-container {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 20px; padding: 30px; margin: 10px 0;
    border: 2px solid rgba(255,255,255,0.1);
    box-shadow: 0 20px 60px rgba(0,0,0,0.4);
    text-align: center; color: white;
  }
  .card-common    { border-color: #95a5a6; }
  .card-uncommon  { border-color: #27ae60; }
  .card-rare      { border-color: #2980b9; }
  .card-epic      { border-color: #8e44ad; box-shadow: 0 0 30px rgba(142,68,173,0.5); }
  .card-legendary {
    border-color: #f39c12;
    box-shadow: 0 0 50px rgba(243,156,18,0.7), 0 0 100px rgba(243,156,18,0.3);
    animation: glow 2s ease-in-out infinite alternate;
  }
  @keyframes glow {
    from { box-shadow: 0 0 20px rgba(243,156,18,0.5); }
    to   { box-shadow: 0 0 60px rgba(243,156,18,0.9), 0 0 120px rgba(243,156,18,0.4); }
  }

  .grade-badge {
    display: inline-block; padding: 4px 12px; border-radius: 20px;
    font-size: 0.75rem; font-weight: 700; letter-spacing: 1px; margin-bottom: 12px;
  }
  .grade-Common    { background:#95a5a6; color:white; }
  .grade-Uncommon  { background:#27ae60; color:white; }
  .grade-Rare      { background:#2980b9; color:white; }
  .grade-Epic      { background:#8e44ad; color:white; }
  .grade-Legendary { background:linear-gradient(90deg,#f39c12,#e74c3c); color:white; }

  .quote-text {
    font-size: 1.15rem; font-style: italic; line-height: 1.7;
    margin: 16px 0 8px; color: #ecf0f1;
  }
  .quote-author { color: #bdc3c7; font-size: 0.85rem; }

  .mission-box {
    background: rgba(255,255,255,0.05); border-radius: 12px;
    padding: 16px 20px; margin-bottom: 10px;
    border-left: 4px solid #667eea;
    font-size: 1.1rem; color: #ecf0f1;
  }

  .metric-pill {
    background: rgba(255,255,255,0.1); border-radius: 30px;
    padding: 6px 16px; display: inline-block; margin: 4px;
    font-size: 0.85rem; color: #ecf0f1;
  }

  .prob-bar-wrap { margin: 4px 0; }
  .stProgress > div > div { background: linear-gradient(90deg, #667eea, #764ba2); }
</style>
""", unsafe_allow_html=True)

# ── 상수 & 데이터 ──────────────────────────────────────────────────────────

LABEL_NAMES = ["건강", "마음챙김", "생산성", "관계", "자기성장"]
LABEL_EMOJIS = {"건강":"💪","마음챙김":"🧘","생산성":"⚡","관계":"🤝","자기성장":"🌱"}
LABEL_COLORS = {
    "건강":"#FF6B6B","마음챙김":"#4ECDC4","생산성":"#45B7D1",
    "관계":"#96CEB4","자기성장":"#FFEAA7",
}

GRADE_WEIGHTS  = {"Common": 60, "Uncommon": 25, "Rare": 10, "Epic": 4, "Legendary": 1}
GRADE_EMOJIS   = {"Common":"⚪","Uncommon":"🟢","Rare":"🔵","Epic":"🟣","Legendary":"🌟"}
GRADE_MESSAGES = {
    "Common":    "기본기를 다지는 시작이에요.",
    "Uncommon":  "조금씩 달라지고 있어요!",
    "Rare":      "희귀한 각성이 일어나고 있어요!",
    "Epic":      "서사시적인 하루가 펼쳐집니다!",
    "Legendary": "🎉 전설적인 갓생 카드 등장! 오늘은 특별한 날이에요!",
}

QUOTES = {
    "건강": [
        ("건강한 몸은 영혼이 머무는 가장 좋은 집이다.", "버트런드 러셀"),
        ("당신의 몸은 당신이 사는 곳이다. 잘 돌보라.", "짐 론"),
        ("운동은 약국에서 살 수 없는 가장 좋은 약이다.", "레베카 클레어"),
        ("건강을 잃으면 모든 것을 잃는다.", "히포크라테스"),
        ("매일 조금씩, 꾸준히 하면 기적이 된다.", "작자 미상"),
    ],
    "마음챙김": [
        ("지금 이 순간만이 우리가 가진 전부다.", "에크하르트 톨레"),
        ("마음의 평화는 외부가 아닌 내면에서 온다.", "달라이 라마"),
        ("조용히 숨을 쉬면 폭풍도 고요해진다.", "작자 미상"),
        ("자기 자신을 돌보는 것은 이기심이 아니라 자기 보존이다.", "오드리 로드"),
        ("행복은 목적지가 아니라 여행하는 방식이다.", "마거릿 리 런벡"),
    ],
    "생산성": [
        ("중요한 일을 먼저 하라. 나머지는 스스로 해결된다.", "피터 드러커"),
        ("완료가 완벽보다 낫다.", "마크 저커버그"),
        ("집중이야말로 천재의 비결이다.", "아이작 뉴턴"),
        ("계획 없는 목표는 그냥 소원에 불과하다.", "앙투안 드 생텍쥐페리"),
        ("작은 진전도 진전이다.", "작자 미상"),
    ],
    "관계": [
        ("우리는 서로를 통해 성장한다.", "작자 미상"),
        ("진정한 우정은 두 영혼이 하나가 되는 것이다.", "아리스토텔레스"),
        ("사람들은 느낀 감정을 기억한다.", "마야 안젤루"),
        ("함께하면 우리는 더 많은 것을 이룰 수 있다.", "헬렌 켈러"),
        ("타인의 성공에 기뻐하는 것이 진정한 너그러움이다.", "작자 미상"),
    ],
    "자기성장": [
        ("배움에는 끝이 없다.", "공자"),
        ("어제의 나보다 더 나은 오늘의 내가 되어라.", "작자 미상"),
        ("가장 수익 높은 투자는 자기 자신에 대한 투자다.", "벤저민 프랭클린"),
        ("지식은 행동을 통해서만 힘이 된다.", "앤턴 체호프"),
        ("성장은 불편함 밖에 있다.", "작자 미상"),
    ],
}

DEFAULT_MISSIONS = {
    "건강": ["오늘 30분 달리기", "물 2리터 마시기", "하루 1만 보 걷기", "스트레칭 10분"],
    "마음챙김": ["명상 5분 하기", "감사일기 3줄 쓰기", "디지털 디톡스 1시간"],
    "생산성": ["할일 목록 우선순위 정리", "포모도로 4세트", "이메일 받은편지함 정리"],
    "관계": ["친구에게 먼저 연락하기", "가족과 저녁 식사", "감사 인사 전하기"],
    "자기성장": ["책 30페이지 읽기", "온라인 강의 1강", "새로운 기술 배우기"],
}


# ── 유틸 함수 ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_ml_model():
    try:
        with open("saved_models/best_ml_model.pkl", "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


@st.cache_resource
def load_dl_model():
    try:
        from models.dl_classifier import DLPredictor
        return DLPredictor(
            "saved_models/bilstm_attention.pth",
            "saved_models/tokenizer.json",
        )
    except Exception:
        return None


def weighted_gacha(seed: int | None = None) -> str:
    if seed is not None:
        random.seed(seed)
    grades  = list(GRADE_WEIGHTS.keys())
    weights = list(GRADE_WEIGHTS.values())
    return random.choices(grades, weights=weights, k=1)[0]


def draw_prob_bars(proba: dict):
    for cat, p in sorted(proba.items(), key=lambda x: -x[1]):
        emoji = LABEL_EMOJIS.get(cat, "")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.progress(p, text=f"{emoji} {cat}")
        with col2:
            st.write(f"**{p*100:.1f}%**")


def visualize_attention_streamlit(chars: list, scores: list, category: str) -> plt.Figure:
    from interpretability.shap_analysis import visualize_attention
    fig = visualize_attention(
        " ".join(chars), scores, chars, category
    )
    return fig


# ── HERO HEADER ───────────────────────────────────────────────────────────

st.markdown('<h1 class="hero-title">🌱 Lucky-Seed</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">AI 갓생 가챠 · ML + DL 미션 분류 데모</p>', unsafe_allow_html=True)
st.markdown("---")

# ── TABS ──────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["🎰 갓생 가챠", "🔬 모델 비교", "📊 성능 리포트"])


# ══════════════════════════════════════════════════════════════════════════
# TAB 1: 갓생 가챠
# ══════════════════════════════════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.subheader("📝 오늘의 미션 입력")

        # 퀵 버튼: text_input 위젯 렌더링 전에 먼저 session_state 값 설정
        st.caption("✨ 빠른 선택")
        quick_cols = st.columns(5)
        quick_labels = list(LABEL_EMOJIS.keys())
        for i, (qcol, cat) in enumerate(zip(quick_cols, quick_labels)):
            with qcol:
                if st.button(f"{LABEL_EMOJIS[cat]}", key=f"quick_{i}", help=cat, use_container_width=True):
                    st.session_state["mission_val"] = random.choice(DEFAULT_MISSIONS[cat])

        # text_input: key 없이 value로만 제어 (key 충돌 방지)
        user_mission = st.text_input(
            "미션을 입력하거나 위에서 선택하세요",
            placeholder="예: 오늘 30분 달리기, 명상 5분 하기 ...",
            value=st.session_state.get("mission_val", ""),
        )
        # 직접 타이핑 시 mission_val 동기화
        st.session_state["mission_val"] = user_mission

        st.markdown("---")
        spin_btn = st.button(
            "🎰 가챠 뽑기!", type="primary", use_container_width=True, key="spin_btn"
        )

        if "gacha_history" not in st.session_state:
            st.session_state["gacha_history"] = []

        if st.session_state["gacha_history"]:
            st.caption("📜 오늘의 뽑기 기록")
            for record in st.session_state["gacha_history"][-5:][::-1]:
                st.markdown(
                    f'<span class="metric-pill">'
                    f'{GRADE_EMOJIS[record["grade"]]} {record["grade"]} '
                    f'| {LABEL_EMOJIS[record["category"]]} {record["category"]}'
                    f'</span>',
                    unsafe_allow_html=True,
                )

    with col_right:
        st.subheader("🃏 결과 카드")

        if spin_btn:
            mission_text = user_mission.strip() if user_mission.strip() else \
                random.choice(DEFAULT_MISSIONS[random.choice(list(DEFAULT_MISSIONS))])

            with st.spinner("뽑는 중..."):
                time.sleep(0.6)

            # 시드: 현재 ms 기반
            seed = int(time.time() * 1000) % 999999
            grade = weighted_gacha(seed)

            # ML 예측으로 카테고리 추론
            ml_model = load_ml_model()
            if ml_model is not None:
                pred_label = ml_model.predict([mission_text])[0]
                category = LABEL_NAMES[pred_label]
            else:
                category = random.choice(LABEL_NAMES)

            # 명언 선택
            quote_text, quote_author = random.choice(QUOTES[category])

            # 기록 저장
            st.session_state["gacha_history"].append({
                "mission": mission_text, "grade": grade, "category": category
            })

            # 카드 렌더링
            grade_css = f"card-{grade.lower()}"
            st.markdown(f"""
            <div class="card-container {grade_css}">
              <div class="grade-badge grade-{grade}">{GRADE_EMOJIS[grade]} {grade.upper()}</div>
              <div style="font-size:3rem; margin:12px 0;">{LABEL_EMOJIS[category]}</div>
              <div style="font-size:1.3rem; font-weight:700; margin-bottom:8px;">{category}</div>
              <div class="mission-box">🎯 {mission_text}</div>
              <div class="quote-text">"{quote_text}"</div>
              <div class="quote-author">— {quote_author}</div>
              <hr style="border-color:rgba(255,255,255,0.2); margin:16px 0;">
              <div style="font-size:0.85rem; color:#aaa;">{GRADE_MESSAGES[grade]}</div>
              <div style="font-size:0.75rem; color:#666; margin-top:8px;">Seed: {seed}</div>
            </div>
            """, unsafe_allow_html=True)

            if grade in ("Epic", "Legendary"):
                st.balloons()
        else:
            st.info("왼쪽에서 미션을 입력하고 가챠를 뽑아보세요! 🎲")


# ══════════════════════════════════════════════════════════════════════════
# TAB 2: 모델 비교
# ══════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🔬 ML vs DL 실시간 예측 비교")

    text_input = st.text_input(
        "분류할 미션 텍스트를 입력하세요",
        value="오늘 30분 달리기",
        key="compare_input",
    )
    predict_btn = st.button("🚀 예측 실행", type="primary", key="predict_btn")

    if predict_btn and text_input.strip():
        ml_col, dl_col = st.columns(2)

        # ML 예측
        with ml_col:
            st.markdown("#### 📐 ML Model (LogReg + TF-IDF)")
            ml_model = load_ml_model()
            if ml_model is not None:
                pred = ml_model.predict([text_input])[0]
                cat  = LABEL_NAMES[pred]
                st.success(f"예측: {LABEL_EMOJIS[cat]} **{cat}**")

                if hasattr(ml_model.named_steps.get("clf"), "predict_proba"):
                    proba = ml_model.predict_proba([text_input])[0]
                    proba_dict = {k: float(v) for k, v in zip(LABEL_NAMES, proba)}
                    draw_prob_bars(proba_dict)
                else:
                    st.caption("(SVM은 확률 미지원)")

                st.caption("⚡ 빠른 추론 · 해석 가능 · 가벼움")
            else:
                st.warning("모델 미학습. `python train.py` 를 먼저 실행하세요.")

        # DL 예측
        with dl_col:
            st.markdown("#### 🧠 DL Model (BiLSTM + Attention)")
            dl_model = load_dl_model()
            if dl_model is not None:
                result = dl_model.predict(text_input)
                cat    = result["predicted_category"]
                st.success(f"예측: {LABEL_EMOJIS[cat]} **{cat}**")

                if result["probabilities"]:
                    draw_prob_bars(result["probabilities"])

                # Attention 시각화
                st.caption("🔍 Attention Weights:")
                attn = result["attention"]
                if attn["chars"]:
                    fig = visualize_attention_streamlit(
                        attn["chars"], attn["scores"], cat
                    )
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                st.caption("🎯 높은 정확도 · 문자 단위 이해 · 해석 가능")
            else:
                st.warning("DL 모델 미학습. `python train.py` 를 먼저 실행하세요.")

    elif not predict_btn:
        st.info("미션 텍스트를 입력하고 예측 실행 버튼을 눌러주세요.")

    # 모델 아키텍처 설명
    st.markdown("---")
    with st.expander("📖 모델 아키텍처 상세보기"):
        arch_col1, arch_col2 = st.columns(2)
        with arch_col1:
            st.markdown("""
**ML Pipeline (scikit-learn)**
```
Input Text
    ↓
TF-IDF Vectorizer
  · analyzer: char_wb (문자 n-gram)
  · ngram_range: (2, 4)
  · max_features: 10,000
  · sublinear_tf: True
    ↓
Logistic Regression
  · C: 5.0
  · multi_class: multinomial
  · solver: lbfgs
    ↓
Softmax → 5 Classes
```
            """)
        with arch_col2:
            st.markdown("""
**DL Model (PyTorch BiLSTM+Attention)**
```
Input Text
    ↓
Char Tokenizer (문자 단위)
    ↓
Embedding (vocab_size × 64)
    ↓
BiLSTM (64→128, 2-layer, bidirectional)
  · output: (Batch, SeqLen, 256)
    ↓
Self-Attention Module
  · Q/K/V Linear(256→256)
  · Scaled Dot-Product Attention
  · Mean-pool → (Batch, 256)
    ↓
LayerNorm
    ↓
FC(256→128) → GELU → Dropout(0.3)
    ↓
FC(128→5) → Softmax
```
            """)


# ══════════════════════════════════════════════════════════════════════════
# TAB 3: 성능 리포트
# ══════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("📊 모델 학습 결과 리포트")

    # JSON summary 로드
    summary_path = "assets/model_summary.json"
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)

        # ── 핵심 지표 ────────────────────────────────────────────
        st.markdown("#### 🏆 모델 성능 요약")
        all_models = {}
        for name, m in summary.get("ml_models", {}).items():
            all_models[name] = m
        for name, m in summary.get("dl_model", {}).items():
            all_models[name] = m

        metric_cols = st.columns(len(all_models))
        for col, (name, m) in zip(metric_cols, all_models.items()):
            with col:
                st.metric(
                    label=name,
                    value=f"F1: {m['f1_weighted']:.4f}",
                    delta=f"Acc: {m['accuracy']:.4f}",
                )

        # ── 비교 차트 ─────────────────────────────────────────────
        if os.path.exists("assets/model_comparison.png"):
            st.image("assets/model_comparison.png", use_column_width=True)

        # ── DL 학습 곡선 ──────────────────────────────────────────
        if os.path.exists("assets/dl_training_history.png"):
            st.markdown("#### 📈 DL 학습 곡선 (BiLSTM+Attention)")
            st.image("assets/dl_training_history.png", use_column_width=True)

        # ── 혼동 행렬 ─────────────────────────────────────────────
        st.markdown("#### 🎯 혼동 행렬 (Confusion Matrix)")
        cm_files = {
            "Logistic Regression": "assets/cm_logisticregression.png",
            "Random Forest":       "assets/cm_randomforest.png",
            "LinearSVC":           "assets/cm_linearsvc.png",
            "BiLSTM+Attention":    "assets/cm_dl.png",
        }
        cm_cols = st.columns(4)
        for col, (name, path) in zip(cm_cols, cm_files.items()):
            if os.path.exists(path):
                with col:
                    st.caption(name)
                    st.image(path, use_column_width=True)

        # ── SHAP ──────────────────────────────────────────────────
        st.markdown("#### 🔍 SHAP Feature Importance")
        if os.path.exists("assets/shap_summary_bar.png"):
            st.image("assets/shap_summary_bar.png", use_column_width=True)
        if os.path.exists("assets/shap_waterfall.png"):
            st.image("assets/shap_waterfall.png", use_column_width=True)
        if os.path.exists("assets/attention_examples.png"):
            st.markdown("#### 👁 Attention Visualization 예시")
            st.image("assets/attention_examples.png", use_column_width=True)

    else:
        st.warning("학습 결과가 없습니다. 터미널에서 아래 명령을 실행하세요:")
        st.code("python train.py --epochs 30", language="bash")

    # ── 기술 스택 ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🛠 기술 스택")
    stack_data = {
        "구분": ["Frontend", "ML", "DL", "Interpretability", "Data"],
        "기술": ["Streamlit", "scikit-learn (TF-IDF + LR/RF/SVM)", "PyTorch (BiLSTM + Self-Attention)",
                "SHAP + Attention Visualization", "Pandas + 합성 데이터"],
        "역할": ["웹 데모 & 시각화", "빠른 베이스라인 + 해석", "고정밀 텍스트 분류", "모델 의사결정 설명", "미션 카테고리 학습 데이터"],
    }
    st.dataframe(pd.DataFrame(stack_data), hide_index=True, use_container_width=True)
