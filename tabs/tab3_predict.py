import streamlit as st
import pandas as pd
import torch
import numpy as np
from PIL import Image

from models.cv_model import load_model_bundle, get_image_transform, TYPE_DESC
from utils.eda_metrics import get_country_ecdf_percentile, performance_level
from components.layout import render_page_header, get_type_name
from components.design_tokens import (
    get_text_style, get_bg_style, get_border_style,
    TEXT_COLORS, FONT_SIZES, SPACING, BORDER_RADIUS, BRAND_COLORS
)

def render(df_ref):
    model, country_encoder, mu, sigma = load_model_bundle()
    country_list = list(country_encoder.categories_[0])
    transform = get_image_transform()
    
    if "selected_country" not in st.session_state:
        st.session_state.selected_country = country_list[0]
    selected_country = st.session_state.selected_country
    
    render_page_header(
        "AI 예측",
        countries=country_list,
        selected_country=selected_country,
        description="게시 전 콘텐츠 성과를 예측하여 최적의 콘텐츠 전략을 수립할 수 있습니다."
    )
    
    current_country = st.session_state.get("selected_country", selected_country)
    if current_country != selected_country:
        selected_country = current_country
    
    st.markdown("---")
    
    # 컬럼 높이 맞추기를 위한 CSS
    st.markdown("""
        <style>
        .stColumn:first-child > div {
            min-height: 400px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        uploaded = st.file_uploader(
            "이미지 업로드",
            type=["jpg", "jpeg", "png"],
            help="성과를 예측할 이미지를 업로드하세요"
        )
        
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(np.array(image), use_container_width=True)
    
    with col2:
        if uploaded:
            img_tensor = transform(image).unsqueeze(0)
            
            country_vec = country_encoder.transform(
                pd.DataFrame([[selected_country]], columns=["country"])
            )
            country_vec = torch.tensor(country_vec, dtype=torch.float32)
            
            with torch.no_grad():
                cls_out, reg_out = model(img_tensor, country_vec)
            
            img_type = int(torch.argmax(cls_out, dim=1).item()) + 1
            pred_z = float(reg_out.item())
            pred_logeng = pred_z * sigma + mu
            percent = get_country_ecdf_percentile(df_ref, selected_country, pred_logeng)
            
            type_name = TYPE_DESC.get(img_type, f"Type {img_type}")
            level, _ = performance_level(percent)
            
            # 예측 결과 카드 HTML
            bg_color = '#DCFCE7' if percent >= 80 else '#FEF9C3' if percent >= 50 else '#FEE2E2'
            text_color = '#166534' if percent >= 80 else '#854D0E' if percent >= 50 else '#991B1B'
            
            result_html = f"""
            <div style="{get_bg_style('white')} {get_border_style('default')} border-radius: {BORDER_RADIUS['lg']}; padding: {SPACING['2xl']}; box-shadow: 0 1px 2px rgba(0,0,0,0.05);">
                <div style="{get_text_style('md', 'tertiary')} margin-bottom: {SPACING['sm']};">
                    예측 성과
                </div>
                <div style="{get_text_style('5xl', 'primary', family='bold')} margin-bottom: {SPACING['md']};">
                    {percent:.1f}%
                </div>
                <div style="display: inline-block; padding: {SPACING['xs']} {SPACING['md']}; border-radius: {BORDER_RADIUS['sm']}; background-color: {bg_color}; color: {text_color}; {get_text_style('sm', weight='semibold')} margin-bottom: {SPACING['lg']};">
                    {level}
                </div>
                
                <div style="border-top: 1px solid #E5E7EB; padding-top: {SPACING['lg']}; margin-top: {SPACING['lg']};">
                    <div style="{get_text_style('base', 'tertiary')} margin-bottom: {SPACING['xs']};">
                        이미지 타입
                    </div>
                    <div style="{get_text_style('lg', 'primary', weight='semibold')} margin-bottom: {SPACING['lg']};">
                        Type {img_type} · {type_name}
                    </div>
                    
                    <div style="{get_text_style('base', 'tertiary')} line-height: 1.6;">
                        이 이미지는 <strong>{selected_country}</strong> 시장 내 전체 콘텐츠 대비 
                        <strong>{level}</strong> 수준의 상대적 성과 위치에 해당합니다.
                    </div>
                </div>
            </div>
            """
            
            # st.html 사용 (Streamlit 1.28.0+)
            try:
                st.html(result_html)
            except AttributeError:
                # st.html이 없는 경우 st.markdown 사용
                st.markdown(result_html, unsafe_allow_html=True)
        else:
            placeholder_html = f"""
            <div style="
                background: linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 100%);
                border: 1px solid #BAE6FD;
                border-radius: 12px;
                padding: 48px 32px;
                text-align: center;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
                min-height: 400px;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
            ">
                <div style="font-size: {FONT_SIZES['6xl']}; margin-bottom: {SPACING['lg']}; opacity: 0.6;">
                    📸
                </div>
                <div style="{get_text_style('lg', 'primary', weight='semibold')} color: #0C4A6E; line-height: 1.6;">
                    이미지를 업로드하면 콘텐츠 성과 예측 결과가 표시됩니다.
                </div>
            </div>
            """
            
            # st.html 사용 (Streamlit 1.28.0+)
            try:
                st.html(placeholder_html)
            except AttributeError:
                # st.html이 없는 경우 st.markdown 사용
                st.markdown(placeholder_html, unsafe_allow_html=True)
