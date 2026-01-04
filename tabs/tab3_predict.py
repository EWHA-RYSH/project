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
    
    # 파일 업로더는 먼저 렌더링
    uploaded = st.file_uploader(
        "이미지 업로드",
        type=["jpg", "jpeg", "png"],
        help="성과를 예측할 이미지를 업로드하세요"
    )
    
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        
        # 컬럼 레이아웃을 위한 강력한 CSS
        st.markdown("""
            <style>
            /* 컬럼이 가로로 배치되도록 강제 - 모든 선택자 사용 */
            div[data-testid="column-container"],
            .stColumns,
            div[data-baseweb="block"] > div[data-testid="column-container"] {
                display: flex !important;
                flex-direction: row !important;
                width: 100% !important;
                gap: 1rem !important;
            }
            div[data-testid="column-container"] > div,
            .stColumns > div,
            div[data-baseweb="block"] > div[data-testid="column-container"] > div {
                display: flex !important;
                flex-direction: column !important;
                width: auto !important;
                max-width: none !important;
                flex-shrink: 1 !important;
            }
            div[data-testid="column-container"] > div:first-child,
            .stColumns > div:first-child {
                flex: 1 1 0% !important;
                min-width: 0 !important;
            }
            div[data-testid="column-container"] > div:last-child,
            .stColumns > div:last-child {
                flex: 1.5 1 0% !important;
                min-width: 0 !important;
            }
            [data-testid="column"] {
                width: auto !important;
                max-width: none !important;
                flex: 1 1 0% !important;
            }
            [data-testid="column"]:first-child {
                flex: 1 1 0% !important;
            }
            [data-testid="column"]:last-child {
                flex: 1.5 1 0% !important;
            }
            [data-testid="column"] > div {
                width: auto !important;
                max-width: none !important;
            }
            div[data-testid="stImage"] img {
                width: 100% !important;
                height: auto !important;
                object-fit: contain !important;
            }
            </style>
            <script>
            (function() {
                function forceLayout() {
                    const containers = document.querySelectorAll('[data-testid="column-container"], .stColumns');
                    containers.forEach(container => {
                        container.style.cssText = 'display: flex !important; flex-direction: row !important; width: 100% !important; gap: 1rem !important;';
                        const divs = container.querySelectorAll(':scope > div');
                        divs.forEach((d, i) => {
                            d.style.cssText = 'display: flex !important; flex-direction: column !important; width: auto !important; max-width: none !important; flex: ' + (i === 0 ? '1 1 0%' : '1.5 1 0%') + ' !important;';
                        });
                    });
                    document.querySelectorAll('[data-testid="column"]').forEach((el, i) => {
                        el.style.cssText = 'width: auto !important; max-width: none !important; flex: ' + (i === 0 ? '1 1 0%' : '1.5 1 0%') + ' !important;';
                    });
                }
                if (document.readyState === 'loading') {
                    document.addEventListener('DOMContentLoaded', forceLayout);
                } else {
                    forceLayout();
                }
                const obs = new MutationObserver(forceLayout);
                obs.observe(document.body, { childList: true, subtree: true });
                setInterval(forceLayout, 200);
            })();
            </script>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            st.image(np.array(image))
        
        with col2:
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
            
            # 예측 결과 카드 - Streamlit 네이티브 컴포넌트 사용
            bg_color = '#DCFCE7' if percent >= 80 else '#FEF9C3' if percent >= 50 else '#FEE2E2'
            text_color = '#166534' if percent >= 80 else '#854D0E' if percent >= 50 else '#991B1B'
            
            # 카드 컨테이너
            st.markdown(f"""
                <div style="
                    {get_bg_style('white')} 
                    {get_border_style('default')} 
                    border-radius: {BORDER_RADIUS['lg']}; 
                    padding: {SPACING['2xl']}; 
                    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
                ">
            """, unsafe_allow_html=True)
            
            # 예측 성과 제목
            st.markdown(f'<div style="{get_text_style(\'md\', \'tertiary\')} margin-bottom: {SPACING[\'sm\']};">예측 성과</div>', unsafe_allow_html=True)
            
            # 퍼센트 표시
            st.markdown(f'<div style="{get_text_style(\'5xl\', \'primary\', family=\'bold\')} margin-bottom: {SPACING[\'md\']};">{percent:.1f}%</div>', unsafe_allow_html=True)
            
            # 레벨 배지
            st.markdown(f"""
                <div style="
                    display: inline-block; 
                    padding: {SPACING['xs']} {SPACING['md']}; 
                    border-radius: {BORDER_RADIUS['sm']}; 
                    background-color: {bg_color}; 
                    color: {text_color}; 
                    {get_text_style('sm', weight='semibold')} 
                    margin-bottom: {SPACING['lg']};
                ">{level}</div>
            """, unsafe_allow_html=True)
            
            # 구분선
            st.markdown(f'<div style="border-top: 1px solid #E5E7EB; padding-top: {SPACING[\'lg\']}; margin-top: {SPACING[\'lg\']};"></div>', unsafe_allow_html=True)
            
            # 이미지 타입
            st.markdown(f'<div style="{get_text_style(\'base\', \'tertiary\')} margin-bottom: {SPACING[\'xs\']};">이미지 타입</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="{get_text_style(\'lg\', \'primary\', weight=\'semibold\')} margin-bottom: {SPACING[\'lg\']};">Type {img_type} · {type_name}</div>', unsafe_allow_html=True)
            
            # 설명 텍스트
            st.markdown(f"""
                <div style="{get_text_style('base', 'tertiary')} line-height: 1.6;">
                    이 이미지는 <strong>{selected_country}</strong> 시장 내 전체 콘텐츠 대비 
                    <strong>{level}</strong> 수준의 상대적 성과 위치에 해당합니다.
                </div>
            """, unsafe_allow_html=True)
            
            # 카드 닫기
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        # 이미지가 없을 때 플레이스홀더
        st.markdown(f"""
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
        """, unsafe_allow_html=True)
