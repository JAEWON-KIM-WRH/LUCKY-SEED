"""
Lucky-Seed · ML Classifier (scikit-learn)
─────────────────────────────────────────
TF-IDF + Logistic Regression / Random Forest / SVM 파이프라인.
미션 텍스트 → 카테고리 다중 분류 (5 classes).
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

LABEL_NAMES = ["건강", "마음챙김", "생산성", "관계", "자기성장"]


# ── 피처 엔지니어링 ────────────────────────────────────────────────────────

def make_tfidf_vectorizer(ngram_range=(1, 2), max_features=10_000):
    """한국어에 적합한 TF-IDF (문자 n-gram 병행)."""
    return TfidfVectorizer(
        analyzer="char_wb",       # 어절 경계 포함 문자 n-gram → 형태소 분석 없이 한국어 대응
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=True,
        strip_accents=None,
        min_df=1,
    )


# ── 모델 파이프라인 정의 ───────────────────────────────────────────────────

def build_pipelines() -> dict:
    """비교할 3개의 ML 파이프라인 반환."""
    return {
        "LogisticRegression": Pipeline([
            ("tfidf", make_tfidf_vectorizer(ngram_range=(2, 4))),
            ("clf", LogisticRegression(
                max_iter=1000,
                C=5.0,
                solver="lbfgs",
                random_state=42,
            )),
        ]),
        "RandomForest": Pipeline([
            ("tfidf", make_tfidf_vectorizer(ngram_range=(1, 3))),
            ("clf", RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=42,
            )),
        ]),
        "LinearSVC": Pipeline([
            ("tfidf", make_tfidf_vectorizer(ngram_range=(2, 4))),
            ("clf", LinearSVC(
                C=1.0,
                max_iter=2000,
                random_state=42,
            )),
        ]),
    }


# ── 학습 & 평가 ───────────────────────────────────────────────────────────

def train_and_evaluate(
    df: pd.DataFrame,
    model_dir: str = "saved_models",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """모든 ML 모델 학습, 평가, 저장."""
    os.makedirs(model_dir, exist_ok=True)

    X = df["text"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    pipelines = build_pipelines()
    results = {}

    for name, pipe in pipelines.items():
        print(f"\n── [{name}] 학습 중 ──")
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(
            y_test, y_pred, target_names=LABEL_NAMES, output_dict=True
        )

        # 5-Fold CV
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1_weighted", n_jobs=-1)

        results[name] = {
            "accuracy": acc,
            "f1_weighted": f1,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "confusion_matrix": cm,
            "classification_report": report,
            "pipeline": pipe,
            "y_test": y_test,
            "y_pred": y_pred,
        }

        print(f"  Accuracy  : {acc:.4f}")
        print(f"  F1 (wgt)  : {f1:.4f}")
        print(f"  CV F1     : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # 저장 (주 모델: LogisticRegression)
        model_path = os.path.join(model_dir, f"{name.lower()}_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(pipe, f)
        print(f"  Saved → {model_path}")

    # 최고 성능 모델 따로 저장
    best_name = max(results, key=lambda k: results[k]["f1_weighted"])
    best_path = os.path.join(model_dir, "best_ml_model.pkl")
    with open(best_path, "wb") as f:
        pickle.dump(results[best_name]["pipeline"], f)
    print(f"\n[Best ML] {best_name} (F1={results[best_name]['f1_weighted']:.4f}) → {best_path}")

    return results, X_train, X_test, y_train, y_test


# ── 추론 인터페이스 ────────────────────────────────────────────────────────

class MLPredictor:
    """학습된 ML 모델로 단일 텍스트 예측."""

    def __init__(self, model_path: str = "saved_models/best_ml_model.pkl"):
        with open(model_path, "rb") as f:
            self.pipeline = pickle.load(f)
        self.label_names = LABEL_NAMES

    def predict(self, text: str) -> dict:
        """
        Returns
        -------
        dict: {
            'predicted_label': int,
            'predicted_category': str,
            'probabilities': dict[str, float]  (LR/RF only, SVC→None)
        }
        """
        pred = self.pipeline.predict([text])[0]
        result = {
            "predicted_label": int(pred),
            "predicted_category": self.label_names[pred],
            "probabilities": None,
        }
        # 확률 지원 여부
        clf = self.pipeline.named_steps["clf"]
        if hasattr(clf, "predict_proba"):
            proba = self.pipeline.predict_proba([text])[0]
            result["probabilities"] = {
                cat: float(p) for cat, p in zip(self.label_names, proba)
            }
        return result


if __name__ == "__main__":
    from data.generate_dataset import generate_dataset, save_dataset
    save_dataset("data")
    df = pd.read_csv("data/mission_dataset.csv")
    results, *_ = train_and_evaluate(df)
