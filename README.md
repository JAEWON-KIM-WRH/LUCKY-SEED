# 🌱 Lucky-Seed: AI 갓생 가챠

> **ML + DL 기반 미션 텍스트 분류 & 갓생 가챠 게임화 시스템**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-orange.svg)](https://pytorch.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-green.svg)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)](https://streamlit.io)

---

## 📌 프로젝트 개요

**Lucky-Seed**는 2030 세대의 자기계발 피로감 문제를 **가챠(뽑기) 메카닉**과 **AI 분류 모델**로 해결하는 프로젝트입니다.

**핵심 ML/DL 과제**: 사용자가 입력한 자유 형식의 미션 텍스트를 5개 카테고리(건강/마음챙김/생산성/관계/자기성장)로 자동 분류하여, 맥락에 맞는 명언과 카드를 가챠 보상으로 제공합니다.

---

## 🏗 아키텍처 개요

```
사용자 미션 텍스트 입력
         ↓
┌────────────────────────────────────────────┐
│            Lucky-Seed 분류 엔진             │
│                                            │
│  ┌─────────────────┐  ┌────────────────┐  │
│  │  ML 파이프라인   │  │  DL 파이프라인  │  │
│  │  TF-IDF+LogReg  │  │  BiLSTM+Attn  │  │
│  │  (빠른 베이스라인) │  │  (고정밀 분류)  │  │
│  └────────┬────────┘  └───────┬────────┘  │
│           └──────────┬────────┘           │
│                      ↓                    │
│              카테고리 예측 (5 class)         │
└──────────────────────┬─────────────────────┘
                       ↓
          확률 기반 가챠 등급 결정
          (Legendary 1% ~ Common 60%)
                       ↓
          맥락 매칭 명언 + 카드 렌더링
```

---

## 📁 프로젝트 구조

```
lucky-seed/
├── data/
│   ├── generate_dataset.py     # 합성 한국어 미션 데이터셋 생성
│   ├── mission_dataset.csv     # 생성된 학습 데이터 (~1,000+ samples)
│   └── quotes.csv              # 카테고리별 명언 DB
│
├── models/
│   ├── ml_classifier.py        # scikit-learn: TF-IDF + LR/RF/SVM
│   └── dl_classifier.py        # PyTorch: BiLSTM + Self-Attention
│
├── interpretability/
│   └── shap_analysis.py        # SHAP (ML) + Attention Viz (DL)
│
├── app/
│   └── streamlit_app.py        # 웹 데모 (3개 탭)
│
├── saved_models/               # 학습된 모델 파일
│   ├── best_ml_model.pkl       # 최고 성능 ML 모델
│   ├── logisticregression_model.pkl
│   ├── randomforest_model.pkl
│   ├── linearsvc_model.pkl
│   ├── bilstm_attention.pth    # DL 모델 (PyTorch)
│   └── tokenizer.json          # 문자 토크나이저
│
├── assets/                     # 생성된 시각화 이미지
│   ├── model_comparison.png
│   ├── dl_training_history.png
│   ├── cm_*.png                # 혼동 행렬
│   ├── shap_summary_bar.png
│   └── attention_examples.png
│
├── train.py                    # 통합 학습 스크립트
├── requirements.txt
└── README.md
```

---

## 🚀 빠른 시작

### 1. 설치

```bash
git clone https://github.com/your-username/lucky-seed.git
cd lucky-seed
pip install -r requirements.txt
```

### 2. 모델 학습

```bash
# 기본 (30 에포크)
python train.py

# 커스텀
python train.py --epochs 50 --batch_size 64
```

학습 완료 후 생성되는 파일:
- `saved_models/best_ml_model.pkl` — 최고 성능 ML 모델
- `saved_models/bilstm_attention.pth` — DL 모델
- `assets/*.png` — 성능 시각화 이미지

### 3. 모델 해석 실행

```bash
python interpretability/shap_analysis.py
```

### 4. 웹 데모 실행

```bash
streamlit run app/streamlit_app.py
```

브라우저에서 `http://localhost:8501` 접속

---

## 🤖 모델 상세

### ML 모델 (scikit-learn)

| 모델 | 핵심 특징 | 장점 |
|------|-----------|------|
| **TF-IDF + LogisticRegression** ⭐ | char n-gram(2-4), sublinear_tf | 빠른 추론, 확률 출력, SHAP 호환 |
| TF-IDF + RandomForest | 앙상블 300 트리 | 비선형 패턴 포착 |
| TF-IDF + LinearSVC | SVM 마진 최대화 | 고차원 희소 피처에 강함 |

**TF-IDF 설정:**
- `analyzer="char_wb"`: 형태소 분석 없이 한국어 처리
- `ngram_range=(2, 4)`: 2~4 문자 시퀀스 피처
- `max_features=10,000`: 상위 빈도 특징만 선택

### DL 모델 (PyTorch)

```
BiLSTMAttention
├── Embedding: vocab_size × 64
├── BiLSTM: 64 → 128 (×2, bidirectional) = 256 dim
│   └── Dropout: 0.3
├── SelfAttention:
│   ├── Q, K, V: Linear(256 → 256)
│   ├── Scaled Dot-Product: score = QK^T / √256
│   └── Context: Attention(V).mean(T)
├── LayerNorm(256)
├── FC(256 → 128) → GELU
├── Dropout(0.3)
└── FC(128 → 5) → Softmax
```

**총 파라미터**: ~약 450K (경량 모델)

---

## 📊 성능 기대치

| 모델 | Accuracy | F1 (weighted) | 특이사항 |
|------|----------|---------------|----------|
| LogisticRegression | ~0.92 | ~0.92 | SHAP 해석 가능 |
| RandomForest | ~0.88 | ~0.88 | 느린 추론 |
| LinearSVC | ~0.91 | ~0.91 | 확률 미지원 |
| **BiLSTM+Attention** | **~0.93** | **~0.93** | Attention 시각화 |

*합성 데이터 기준; 실제 사용자 데이터로는 성능이 달라질 수 있습니다.*

---

## 🔍 모델 해석

### SHAP (ML 모델)
- `shap.LinearExplainer`를 사용하여 TF-IDF 피처의 기여도 분석
- 카테고리별 상위 15개 특징 바 차트
- 개별 예측 워터폴 플롯

### Attention Visualization (DL 모델)
- Scaled Dot-Product Attention의 mean 가중치를 문자 단위로 시각화
- 어떤 문자 패턴이 분류 결정에 영향을 미쳤는지 직관적으로 확인

---

## 🎰 가챠 시스템 설계

### 등급 확률

| 등급 | 확률 | 이펙트 |
|------|------|--------|
| ⚪ Common | 60% | 기본 |
| 🟢 Uncommon | 25% | 녹색 글로우 |
| 🔵 Rare | 10% | 파란 글로우 |
| 🟣 Epic | 4% | 보라 글로우 |
| 🌟 Legendary | 1% | 골드 글로우 + 애니메이션 |

### 시드 전략
```python
seed = int(time.time() * 1000) % 999999  # ms 단위 타임스탬프
grade = random.choices(grades, weights=weights, k=1, random_state=seed)
```
"이 보상은 오직 지금 이 순간에만 존재하는 결과"임을 강조

---

## 🗂 산출물 목록

| 구분 | 파일 | 설명 |
|------|------|------|
| ML 모델 | `saved_models/best_ml_model.pkl` | pickle 직렬화 |
| DL 모델 | `saved_models/bilstm_attention.pth` | PyTorch state_dict |
| 토크나이저 | `saved_models/tokenizer.json` | 문자→ID 매핑 |
| 비교 차트 | `assets/model_comparison.png` | ML vs DL 성능 |
| 학습 곡선 | `assets/dl_training_history.png` | Loss/Acc 커브 |
| SHAP | `assets/shap_summary_bar.png` | Feature Importance |
| Attention | `assets/attention_examples.png` | 문자별 가중치 |

---

## 📜 라이선스

MIT License

---

## 👤 제작자

재원 김 (TONY LAMA)
