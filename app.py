# ======================================================
# Impress.AI — Final Streamlit App
# ======================================================

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pickle

# ======================================================
# 기본 설정
# ======================================================
st.set_page_config(
    page_title="Impress.AI",
    page_icon="📸",
    layout="wide"
)

st.markdown(
    """
    <div style="text-align:center; margin-bottom: 30px;">
        <h1 style="font-size:50px; font-weight:800;">
            Impress<span style="color:#3b82f6;">.AI</span>
        </h1>
        <p style="font-size:18px; color:#6b7280;">
            Image-based Content Performance Analysis & Prediction
        </p>
        <p style="font-size:14px; color:#9ca3af;">
            From visual content to actionable insight
        </p>
    </div>
    <hr style="border:none; height:1px; background-color:#e5e7eb; margin-bottom:30px;">
    """,
    unsafe_allow_html=True
)

# ======================================================
# 데이터 로드 (분석 + 기준 분포)
# ======================================================
@st.cache_data
def load_data():
    return pd.read_excel("agent6_final_db.xlsx")

df = load_data()

# ======================================================
# 사이드바
# ======================================================
st.sidebar.header("🔧 필터")

countries = sorted(df["country"].unique())
selected_country = st.sidebar.selectbox(
    "국가 선택",
    options=["ALL"] + countries
)

if selected_country == "ALL":
    df_view = df.copy()
else:
    df_view = df[df["country"] == selected_country]

# ======================================================
# TAB 구성
# ======================================================
tab1, tab2, tab3 = st.tabs([
    "📊 활용도 분석",
    "🔥 반응 & 성과 분석",
    "🤖 CV 기반 성과 예측"
])

# ======================================================
# TAB 1. 활용도 분석
# ======================================================
with tab1:
    st.subheader("📊 이미지 유형 활용도")

    usage = (
        df_view
        .groupby("img_type")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.dataframe(usage, use_container_width=True)

    with col2:
        fig, ax = plt.subplots()
        sns.barplot(data=usage, x="img_type", y="count", ax=ax)
        ax.set_title("Image Type Usage Count")
        ax.set_xlabel("Image Type")
        ax.set_ylabel("Number of Images")
        st.pyplot(fig)

# ======================================================
# TAB 2. 반응 & 성과 분석
# ======================================================
with tab2:
    st.subheader("🔥 이미지 유형별 성과 분포")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        sns.boxplot(data=df_view, x="img_type", y="eng_rate", ax=ax)
        ax.set_yscale("log")
        ax.set_title("Engagement Rate (log scale)")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        sns.boxplot(
            data=df_view,
            x="img_type",
            y="eng_rank_country_type",
            ax=ax
        )
        ax.set_title("Relative Rank within Country & Type")
        st.pyplot(fig)

# ======================================================
# TAB 3. CV 기반 성과 예측
# ======================================================

# ---------- 스타일 ----------
st.markdown(
    """
    <style>
    .result-box {
        background-color: #f7f9fc;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
    }
    .badge {
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 13px;
        font-weight: 600;
    }
    .high { background-color:#dbeafe; color:#1d4ed8; }
    .mid  { background-color:#fef3c7; color:#92400e; }
    .low  { background-color:#fee2e2; color:#991b1b; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- 모델 정의 ----------
class MultiTaskModel(nn.Module):
    def __init__(self, num_country, num_classes=6):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        feat_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.fc_shared = nn.Linear(feat_dim + num_country, 512)
        self.cls_head = nn.Linear(512, num_classes)
        self.reg_head = nn.Linear(512, 1)

    def forward(self, image, country_vec):
        feat = self.backbone(image)
        x = torch.cat([feat, country_vec], dim=1)
        x = self.fc_shared(x)
        return self.cls_head(x), self.reg_head(x)

# ---------- 모델 로드 ----------
@st.cache_resource
def load_model():
    with open("country_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)

    model = MultiTaskModel(
        num_country=len(encoder.categories_[0])
    )
    model.load_state_dict(
        torch.load("final_multitask_rank_model.pth", map_location="cpu")
    )
    model.eval()
    return model, encoder

model, country_encoder = load_model()
country_list = list(country_encoder.categories_[0])

# ---------- Transform ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------- 타입 설명 ----------
TYPE_DESC = {
    "1": "여러 제품을 함께 보여주는 제품 단체샷",
    "2": "제품 1개를 단독으로 강조한 제품 단독샷",
    "3": "제품 제형과 질감을 강조한 질감 클로즈업 이미지",
    "4": "모델과 제품을 함께 배치한 이미지",
    "5": "제품 없이 모델 중심으로 연출한 이미지",
    "6": "여러 인물과 제품을 함께 보여주는 이미지"
}

# ---------- 성과 레벨 ----------
def performance_level_relative(score_0_1):
    if score_0_1 <= 0.1:
        return "상위권", "high"
    elif score_0_1 <= 0.3:
        return "중상위권", "mid"
    else:
        return "중·하위권", "low"

# ---------- TAB 3 UI ----------
with tab3:
    st.subheader("🤖 CV 기반 콘텐츠 성과 예측")

    left, right = st.columns([1, 1.2])

    with left:
        uploaded = st.file_uploader(
            "이미지 업로드",
            type=["jpg", "png", "jpeg"]
        )
        country = st.selectbox("국가 선택", country_list)

        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(image)

    if uploaded:
        img_tensor = transform(image).unsqueeze(0)

        country_vec = country_encoder.transform(
            pd.DataFrame([[country]], columns=["country"])
        )
        country_vec = torch.tensor(country_vec, dtype=torch.float32)

        with torch.no_grad():
            cls_out, reg_out = model(img_tensor, country_vec)
            cls_idx = torch.argmax(cls_out, dim=1).item()
            img_type = str(cls_idx + 1)
            score = float(reg_out.item())  # 0~1
            percentile = score * 100

        level, badge_class = performance_level_relative(score)

        with right:
            st.markdown("<div class='result-box'>", unsafe_allow_html=True)

            st.markdown(
                f"### 이미지 유형: **Type {img_type}**  \n"
                f"{TYPE_DESC.get(img_type)}"
            )

            st.markdown(
                f"<span class='badge {badge_class}'>{level}</span>",
                unsafe_allow_html=True
            )

            st.markdown("### 예상 성과 순위")
            st.progress(min(max(percentile, 0), 100) / 100)
            st.write(f"동일 국가 내 콘텐츠 대비 **상위 약 {percentile:.1f}%**")

            st.markdown("### 🧠 AI 해석")
            st.write(
                f"이 이미지는 **{country} 시장 기준**, "
                f"{TYPE_DESC.get(img_type)}로 분류되었습니다. "
                f"최종 학습된 모델의 예측에 따르면, "
                f"동일 국가 내 유사 콘텐츠 대비 "
                f"**상위 약 {percentile:.1f}% 수준의 성과**가 예상됩니다."
            )

            st.markdown(
                "<p style='font-size:13px; color:#9ca3af;'>"
                "Model Performance (Validation): "
                "Classification Acc ≈ 0.90 · "
                "Engagement Rank Spearman ≈ 0.27"
                "</p>",
                unsafe_allow_html=True
            )

            st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.info("⬅️ 이미지를 업로드하면 예측 결과가 표시됩니다.")
