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
    "생산성": ["할일 목록 우선순위 정리", "집중 작업 2시간 완료", "이메일 받은편지함 정리"],
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
# TAB 1: 갓생 가챠  (3-STEP FLOW)
# STEP 1 → 미션 선택   STEP 2 → 미션 클리어   STEP 3 → 가챠 뽑기 → 카드
# ══════════════════════════════════════════════════════════════════════════

# ── 추가 CSS (스텝 UI) ─────────────────────────────────────────────────────
st.markdown("""
<style>
  .step-bar {
    display:flex; align-items:center; justify-content:center;
    gap:0; margin: 0.5rem 0 1.8rem;
  }
  .step-node {
    width:38px; height:38px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-weight:700; font-size:1rem;
    transition: all .3s;
  }
  .step-active   { background:linear-gradient(135deg,#667eea,#764ba2); color:white; box-shadow:0 0 12px rgba(102,126,234,0.6); }
  .step-done     { background:#27ae60; color:white; }
  .step-inactive { background:#2d2d2d; color:#666; border:2px solid #444; }
  .step-label    { font-size:0.72rem; text-align:center; margin-top:4px; color:#aaa; }
  .step-line     { flex:1; height:3px; max-width:60px; background:#444; margin:0 4px; border-radius:2px; }
  .step-line-done { background:linear-gradient(90deg,#27ae60,#667eea); }
  .mission-selected {
    background:linear-gradient(135deg,rgba(102,126,234,0.15),rgba(118,75,162,0.15));
    border:2px solid rgba(102,126,234,0.5); border-radius:16px;
    padding:24px 28px; text-align:center; margin:12px 0;
  }
  .mission-selected .cat-emoji { font-size:2.8rem; }
  .mission-selected .mission-title {
    font-size:1.4rem; font-weight:700; color:#ecf0f1; margin:10px 0 4px;
  }
  .mission-selected .cat-name { font-size:0.9rem; color:#aaa; }
  .rarity-table {
    background:rgba(255,255,255,0.03); border-radius:14px;
    padding:16px 20px; margin-top:24px;
    border:1px solid rgba(255,255,255,0.08);
  }
  .rarity-row {
    display:flex; align-items:center; gap:10px;
    padding:6px 0; border-bottom:1px solid rgba(255,255,255,0.05);
  }
  .rarity-row:last-child { border-bottom:none; }
  .rarity-badge {
    min-width:90px; text-align:center;
    padding:3px 10px; border-radius:20px;
    font-size:0.75rem; font-weight:700; letter-spacing:.5px;
  }
  .rarity-bar-bg {
    flex:1; height:8px; background:rgba(255,255,255,0.08);
    border-radius:4px; overflow:hidden;
  }
  .rarity-bar-fill { height:100%; border-radius:4px; }
  .rarity-pct { min-width:36px; text-align:right; font-size:0.8rem; color:#aaa; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ── session_state 초기화 ────────────────────────────────────────────────────
for k, v in [("gacha_step", 1), ("gacha_mission", ""), ("gacha_category", ""),
             ("gacha_result", None), ("gacha_history", []),
             ("preview_mission", ""), ("preview_cat", "")]:
    if k not in st.session_state:
        st.session_state[k] = v

with tab1:

    # ── 스텝 인디케이터 ─────────────────────────────────────────────────────
    s = st.session_state["gacha_step"]
    def _sn(n):
        if n < s:  return f'<div class="step-node step-done">✓</div>'
        if n == s: return f'<div class="step-node step-active">{n}</div>'
        return     f'<div class="step-node step-inactive">{n}</div>'
    def _sl(done):
        cls = "step-line-done" if done else ""
        return f'<div class="step-line {cls}"></div>'

    st.markdown(f"""
    <div class="step-bar">
      <div style="text-align:center">
        {_sn(1)}
        <div class="step-label">미션 선택</div>
      </div>
      {_sl(s > 1)}
      <div style="text-align:center">
        {_sn(2)}
        <div class="step-label">미션 클리어</div>
      </div>
      {_sl(s > 2)}
      <div style="text-align:center">
        {_sn(3)}
        <div class="step-label">가챠 뽑기</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════
    # STEP 1 — 랜덤 미션 뽑기
    # ════════════════════════════════════════════════
    if st.session_state["gacha_step"] == 1:

        # 뽑힌 미션 미리보기 (있으면 표시)
        previewed = st.session_state.get("preview_mission", "")
        previewed_cat = st.session_state.get("preview_cat", "")

        st.markdown("""
        <div style="text-align:center; padding: 10px 0 4px;">
          <div style="font-size:1.1rem; color:#aaa; margin-bottom:18px;">
            버튼을 눌러 오늘의 미션을 랜덤으로 뽑아보세요!
          </div>
        </div>
        """, unsafe_allow_html=True)

        # 미션 카드 미리보기 영역
        if previewed:
            color = LABEL_COLORS[previewed_cat]
            emoji = LABEL_EMOJIS[previewed_cat]
            st.markdown(f"""
            <div style="
              background: linear-gradient(135deg, rgba(102,126,234,0.12), rgba(118,75,162,0.12));
              border: 2px solid {color}55;
              border-radius: 20px; padding: 30px 24px;
              text-align: center; margin: 0 auto 20px; max-width: 480px;
              box-shadow: 0 0 24px {color}33;
            ">
              <div style="font-size:3.2rem; margin-bottom:10px;">{emoji}</div>
              <div style="font-size:1.35rem; font-weight:700; color:#ecf0f1; margin-bottom:8px;">
                {previewed}
              </div>
              <div style="display:inline-block; padding:4px 14px; border-radius:20px;
                background:{color}33; color:{color}; font-size:0.85rem; font-weight:600;">
                {previewed_cat}
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="
              background: rgba(255,255,255,0.03);
              border: 2px dashed rgba(255,255,255,0.12);
              border-radius: 20px; padding: 40px 24px;
              text-align: center; margin: 0 auto 20px; max-width: 480px;
            ">
              <div style="font-size:3rem; margin-bottom:10px; opacity:0.3;">🎲</div>
              <div style="color:#555; font-size:1rem;">미션이 여기에 나타납니다</div>
            </div>
            """, unsafe_allow_html=True)

        # 버튼 영역
        all_missions = [(m, cat) for cat, ms in DEFAULT_MISSIONS.items() for m in ms]

        c1, c2 = st.columns([3, 1])
        with c1:
            if st.button("🎲 랜덤 미션 뽑기!", type="primary",
                         use_container_width=True, key="random_mission_btn"):
                picked, picked_cat = random.choice(all_missions)
                st.session_state["preview_mission"] = picked
                st.session_state["preview_cat"]     = picked_cat
                st.rerun()
        with c2:
            if st.button("✅ 이걸로!", use_container_width=True,
                         disabled=not previewed, key="confirm_mission_btn"):
                st.session_state["gacha_mission"]  = previewed
                st.session_state["gacha_category"] = previewed_cat
                st.session_state["gacha_step"]     = 2
                st.session_state["gacha_result"]   = None
                st.rerun()

        # 직접 입력
        with st.expander("✏️ 직접 입력하기"):
            custom = st.text_input("나만의 미션", placeholder="예: 오늘 독서 20분 하기",
                                   key="custom_mission_input")
            if st.button("이 미션으로 시작하기 →", disabled=not custom.strip(),
                         type="primary", use_container_width=True, key="custom_start"):
                ml_model = load_ml_model()
                if ml_model is not None:
                    pred = ml_model.predict([custom.strip()])[0]
                    cat  = LABEL_NAMES[pred]
                else:
                    cat = random.choice(LABEL_NAMES)
                st.session_state["gacha_mission"]  = custom.strip()
                st.session_state["gacha_category"] = cat
                st.session_state["gacha_step"]     = 2
                st.session_state["gacha_result"]   = None
                st.rerun()

    # ════════════════════════════════════════════════
    # STEP 2 — 미션 수행 & 클리어
    # ════════════════════════════════════════════════
    elif st.session_state["gacha_step"] == 2:
        mission  = st.session_state["gacha_mission"]
        category = st.session_state["gacha_category"]
        emoji    = LABEL_EMOJIS[category]
        color    = LABEL_COLORS[category]

        st.markdown("### 🏃 지금 이 미션을 수행하세요!")

        st.markdown(f"""
        <div class="mission-selected">
          <div class="cat-emoji">{emoji}</div>
          <div class="mission-title">{mission}</div>
          <div class="cat-name" style="color:{color};">{category}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")
        st.info("✅ 미션을 완료했다면 아래 버튼을 눌러주세요!")

        c1, c2 = st.columns([2, 1])
        with c1:
            if st.button("✅ 미션 클리어!", type="primary",
                         use_container_width=True, key="clear_btn"):
                st.session_state["gacha_step"] = 3
                st.rerun()
        with c2:
            if st.button("← 다시 선택", use_container_width=True, key="back_btn"):
                st.session_state["gacha_step"]   = 1
                st.session_state["gacha_mission"] = ""
                st.rerun()

    # ════════════════════════════════════════════════
    # STEP 3 — 가챠 뽑기 + 카드
    # ════════════════════════════════════════════════
    elif st.session_state["gacha_step"] == 3:
        mission  = st.session_state["gacha_mission"]
        category = st.session_state["gacha_category"]
        emoji    = LABEL_EMOJIS[category]

        # ── 가챠 뽑기 전 ──────────────────────────────
        if st.session_state["gacha_result"] is None:
            st.markdown("### 🎉 미션 클리어 축하해요!")
            color = LABEL_COLORS[category]
            st.markdown(f"""
            <div class="mission-selected">
              <div class="cat-emoji">{emoji}</div>
              <div class="mission-title">✅ {mission}</div>
              <div class="cat-name" style="color:{color};">클리어 완료!</div>
            </div>
            """, unsafe_allow_html=True)

            # ── 카드 희귀도 쇼케이스 ─────────────────────────
            GRADE_SHOWCASE = {
                "Common":    {"color":"#95a5a6","glow":"rgba(149,165,166,0.3)","label":"커먼","ko":"기본"},
                "Uncommon":  {"color":"#27ae60","glow":"rgba(39,174,96,0.4)","label":"언커먼","ko":"성장"},
                "Rare":      {"color":"#2980b9","glow":"rgba(41,128,185,0.5)","label":"레어","ko":"도약"},
                "Epic":      {"color":"#8e44ad","glow":"rgba(142,68,173,0.6)","label":"에픽","ko":"전진"},
                "Legendary": {"color":"#f39c12","glow":"rgba(243,156,18,0.7)","label":"레전더리","ko":"전설"},
            }
            cards_html = ""
            for gname, info in GRADE_SHOWCASE.items():
                pct  = GRADE_WEIGHTS[gname]
                gemoji = GRADE_EMOJIS[gname]
                is_legendary = gname == "Legendary"
                anim = "animation:glow 1.5s ease-in-out infinite alternate;" if is_legendary else ""
                cards_html += f"""
                <div style="
                  flex:1; min-width:0;
                  background:linear-gradient(160deg,#1a1a2e,#0f1923);
                  border:2px solid {info['color']}88;
                  border-radius:16px; padding:16px 8px 12px;
                  text-align:center;
                  box-shadow: 0 0 18px {info['glow']};
                  {anim}
                ">
                  <div style="font-size:1.8rem; margin-bottom:6px;">{gemoji}</div>
                  <div style="
                    font-size:0.7rem; font-weight:800; letter-spacing:1px;
                    color:{info['color']}; margin-bottom:4px; text-transform:uppercase;
                  ">{info['label']}</div>
                  <div style="font-size:0.65rem; color:#888; margin-bottom:8px;">{info['ko']}</div>
                  <div style="
                    background:{info['color']}22; border-radius:20px;
                    padding:4px 0; font-size:0.9rem; font-weight:800;
                    color:{info['color']};
                  ">{pct}%</div>
                </div>"""

            st.markdown(f"""
            <div style="margin:24px 0 8px;">
              <div style="text-align:center;font-size:0.8rem;color:#666;
                          font-weight:600;letter-spacing:2px;margin-bottom:12px;">
                ✦ 카드 등급 & 확률 ✦
              </div>
              <div style="display:flex;gap:8px;align-items:stretch;">
                {cards_html}
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🎰 가챠 뽑기!", type="primary",
                         use_container_width=True, key="gacha_btn"):
                with st.spinner("✨ 카드 소환 중..."):
                    time.sleep(0.8)
                seed  = int(time.time() * 1000) % 999999
                grade = weighted_gacha(seed)
                q_text, q_author = random.choice(QUOTES[category])
                st.session_state["gacha_result"] = {
                    "grade": grade, "category": category,
                    "mission": mission, "seed": seed,
                    "quote": q_text, "author": q_author,
                }
                st.session_state["gacha_history"].append(
                    {"mission": mission, "grade": grade, "category": category}
                )
                st.rerun()

        # ── 카드 결과 ─────────────────────────────────
        else:
            r      = st.session_state["gacha_result"]
            grade  = r["grade"]
            g_css  = f"card-{grade.lower()}"

            if grade == "Legendary":
                st.balloons()

            st.markdown(f"""
            <div class="card-container {g_css}">
              <div class="grade-badge grade-{grade}">{GRADE_EMOJIS[grade]} {grade.upper()}</div>
              <div style="font-size:3.2rem; margin:14px 0;">{LABEL_EMOJIS[r['category']]}</div>
              <div style="font-size:1.4rem; font-weight:700; margin-bottom:8px;">{r['category']}</div>
              <div class="mission-box">🎯 {r['mission']}</div>
              <div class="quote-text">"{r['quote']}"</div>
              <div class="quote-author">— {r['author']}</div>
              <hr style="border-color:rgba(255,255,255,0.2); margin:18px 0;">
              <div style="font-size:0.9rem; color:#ccc;">{GRADE_MESSAGES[grade]}</div>
              <div style="font-size:0.72rem; color:#555; margin-top:6px;">Seed: {r['seed']}</div>
            </div>
            """, unsafe_allow_html=True)

            if grade == "Epic":
                st.balloons()

            c1, c2 = st.columns(2)
            with c1:
                if st.button("🔄 다시 뽑기", use_container_width=True, key="retry_btn"):
                    st.session_state["gacha_result"] = None
                    st.rerun()
            with c2:
                if st.button("🌱 새 미션 시작", type="primary",
                             use_container_width=True, key="new_mission_btn"):
                    st.session_state["gacha_step"]    = 1
                    st.session_state["gacha_mission"] = ""
                    st.session_state["gacha_result"]  = None
                    st.session_state["preview_mission"] = ""
                    st.rerun()

        # ── 뽑기 기록 ──────────────────────────────────
        if st.session_state["gacha_history"]:
            st.markdown("<br>", unsafe_allow_html=True)
            st.caption("📜 오늘의 뽑기 기록")
            for record in st.session_state["gacha_history"][-5:][::-1]:
                st.markdown(
                    f'<span class="metric-pill">'
                    f'{GRADE_EMOJIS[record["grade"]]} {record["grade"]} '
                    f'| {LABEL_EMOJIS[record["category"]]} {record["category"]}'
                    f'</span>',
                    unsafe_allow_html=True,
                )


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
