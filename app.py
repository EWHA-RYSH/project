# ======================================================
# Global Instagram Content Insight Tool (Final App)
# ======================================================

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pickle
import numpy as np

# ======================================================
# 기본 설정
# ======================================================
st.set_page_config(
    page_title="Global Instagram Content Insight Tool",
    page_icon="📸",
    layout="wide"
)

st.title("🌍 Global Instagram Content Insight Tool")
st.caption("국가별 인스타그램 콘텐츠 분석 & CV 기반 성과 예측 데모")

# ======================================================
# 데이터 로드 (분석용)
# ======================================================
@st.cache_data
def load_data():
    return pd.read_excel("agent6_final_db.xlsx")

df = load_data()

required_cols = ["country", "img_type", "eng_rate", "eng_rank_country_type"]
for col in required_cols:
    if col not in df.columns:
        st.error(f"❌ 필수 컬럼 누락: {col}")
        st.stop()

# ======================================================
# 사이드바
# ======================================================
st.sidebar.header("🔧 필터 설정")

countries = sorted(df["country"].unique())
selected_country = st.sidebar.selectbox(
    "국가 선택",
    options=["ALL"] + countries
)

if selected_country != "ALL":
    df_view = df[df["country"] == selected_country]
else:
    df_view = df.copy()

# ======================================================
# TAB 구성
# ======================================================
tab1, tab2, tab3 = st.tabs([
    "📊 활용도 모니터링",
    "🔥 반응 & 성과 분석",
    "🤖 CV 기반 콘텐츠 성과 예측"
])

# ======================================================
# TAB 1. 활용도 모니터링
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
        sns.barplot(
            data=usage,
            x="img_type",
            y="count",
            ax=ax
        )
        ax.set_title("Image Type Usage Count")
        ax.set_xlabel("Image Type")
        ax.set_ylabel("Number of Images")
        st.pyplot(fig)

    st.markdown("""
    **해석 포인트**
    - 많이 쓰이는 유형이 항상 반응이 좋은 것은 아님
    - 국가별 콘텐츠 전략의 관성 확인 가능
    """)

# ======================================================
# TAB 2. 반응 & 성과 분석
# ======================================================
with tab2:
    st.subheader("🔥 이미지 유형별 반응 성과")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Engagement Rate 분포 (log scale)**")
        fig, ax = plt.subplots()
        sns.boxplot(
            data=df_view,
            x="img_type",
            y="eng_rate",
            ax=ax
        )
        ax.set_yscale("log")
        st.pyplot(fig)

    with col2:
        st.markdown("**이미지 유형 내 상대 순위 (낮을수록 상위)**")
        fig, ax = plt.subplots()
        sns.boxplot(
            data=df_view,
            x="img_type",
            y="eng_rank_country_type",
            ax=ax
        )
        st.pyplot(fig)

    st.markdown("""
    **해석 포인트**
    - 동일 이미지 유형 내에서도 성과 격차 존재
    - 국가별 ‘성공 패턴’ 탐색 가능
    """)

# ======================================================
# TAB 3. CV 기반 콘텐츠 성과 예측 (실제 모델)
# ======================================================

# ---------- 스타일 ----------
st.markdown("""
<style>
.result-box {
    background-color: #f7f9fc;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #e0e0e0;
}
.highlight {
    color: #3b82f6;
    font-weight: 700;
}
.small-text {
    color: #666666;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

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

    model = MultiTaskModel(num_country=len(encoder.categories_[0]))
    model.load_state_dict(
        torch.load("final_multitask_rank_model.pth", map_location="cpu")
    )
    model.eval()
    return model, encoder

model, country_encoder = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

TYPE_DESC = {
    "1": "여러 제품을 함께 보여주는 제품 단체샷",
    "2": "한 제품을 단독으로 강조한 제품 단독샷",
    "3": "제품 제형/텍스처를 중심으로 한 제품 질감샷",
    "4": "모델과 제품을 함께 배치한 이미지",
    "5": "제품 없이 모델 중심으로 연출된 이미지",
    "6": "여러 인물과 제품을 함께 보여주는 이미지"
}

def performance_level(p):
    if p <= 10:
        return "매우 높은 반응이 기대됩니다"
    elif p <= 30:
        return "비교적 높은 반응이 예상됩니다"
    elif p <= 60:
        return "평균 이상의 반응을 기대할 수 있습니다"
    else:
        return "반응이 제한적일 가능성이 있습니다"

def generate_explanation(img_type, country, percentile):
    return (
        f"이 이미지는 **{country} 시장 기준**, "
        f"{TYPE_DESC.get(img_type)}로 분류되었습니다. "
        f"동일 국가 내 유사 콘텐츠 대비 "
        f"**상위 약 {percentile:.1f}% 수준의 성과**가 예상되며, "
        f"{performance_level(percentile)}."
    )

with tab3:
    st.subheader("🤖 CV 기반 콘텐츠 성과 예측 (데모)")

    left, right = st.columns([1, 1.2])

    with left:
        uploaded = st.file_uploader(
            "이미지 업로드",
            type=["jpg", "png", "jpeg"]
        )
        country = st.selectbox(
            "국가 선택",
            country_encoder.categories_[0]
        )

        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, use_container_width=True)

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
            percentile = float(reg_out.item() * 100)
            percentile = max(0, min(100, percentile))

        with right:
            st.markdown("<div class='result-box'>", unsafe_allow_html=True)
            st.markdown(
                f"### 이미지 유형: <span class='highlight'>Type {img_type}</span>",
                unsafe_allow_html=True
            )
            st.markdown("### 예상 성과 순위")
            st.progress(percentile / 100)
            st.markdown(f"상위 **{percentile:.1f}%**")
            st.markdown("### 🧠 AI 해석")
            st.write(generate_explanation(img_type, country, percentile))
            st.markdown("</div>", unsafe_allow_html=True)

    else:
        with right:
            st.info("⬅️ 이미지를 업로드하면 예측 결과가 표시됩니다.")

# ======================================================
# Footer
# ======================================================
st.markdown("---")
st.caption("AmorePacific AI Challenge | Global Content Insight Tool")
