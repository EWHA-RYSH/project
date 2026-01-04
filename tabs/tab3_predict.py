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
    
    # 컬럼 레이아웃을 위한 CSS/JavaScript - 페이지 상단에 배치
    st.markdown("""
        <style>
        /* 컬럼 레이아웃 강제 */
        .stColumns,
        div[data-testid="column-container"] {
            display: flex !important;
            flex-direction: row !important;
            width: 100% !important;
            gap: 1rem !important;
        }
        .stColumns > div,
        div[data-testid="column-container"] > div {
            display: flex !important;
            flex-direction: column !important;
            width: auto !important;
            max-width: none !important;
            flex-shrink: 1 !important;
        }
        .stColumns > div:first-child,
        div[data-testid="column-container"] > div:first-child {
            flex: 1 1 0% !important;
            min-width: 0 !important;
        }
        .stColumns > div:last-child,
        div[data-testid="column-container"] > div:last-child {
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
        /* 폰트 적용 */
        [data-testid="column"] *,
        .stColumns * {
            font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', 'Malgun Gothic', sans-serif !important;
        }
        </style>
        <script>
        (function() {
            function forceLayout() {
                // 모든 컬럼 컨테이너 찾기
                const selectors = ['.stColumns', '[data-testid="column-container"]'];
                selectors.forEach(selector => {
                    const containers = document.querySelectorAll(selector);
                    containers.forEach(container => {
                        container.style.cssText = 'display: flex !important; flex-direction: row !important; width: 100% !important; gap: 1rem !important;';
                        const divs = Array.from(container.children);
                        divs.forEach((d, i) => {
                            d.style.cssText = 'display: flex !important; flex-direction: column !important; width: auto !important; max-width: none !important; flex: ' + (i === 0 ? '1 1 0%' : '1.5 1 0%') + ' !important;';
                        });
                    });
                });
                // 개별 컬럼 요소
                const columnEls = document.querySelectorAll('[data-testid="column"]');
                columnEls.forEach((el, i) => {
                    el.style.cssText = 'width: auto !important; max-width: none !important; flex: ' + (i === 0 ? '1 1 0%' : '1.5 1 0%') + ' !important;';
                });
            }
            // 즉시 실행
            forceLayout();
            // DOM 로드 후 실행
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', forceLayout);
            }
            // 변경 감지
            const observer = new MutationObserver(forceLayout);
            observer.observe(document.body, { childList: true, subtree: true, attributes: true });
            // 주기적 실행
            setInterval(forceLayout, 50);
        })();
        </script>
    """, unsafe_allow_html=True)
    
    # 파일 업로더는 먼저 렌더링
    uploaded = st.file_uploader(
        "이미지 업로드",
        type=["jpg", "jpeg", "png"],
        help="성과를 예측할 이미지를 업로드하세요"
    )
    
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        
        # 예측 수행
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
        
        # 이미지를 base64로 인코딩
        import base64
        from io import BytesIO
        
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # 예측 결과 카드 - 스타일을 변수로 먼저 할당
        bg_color = '#DCFCE7' if percent >= 80 else '#FEF9C3' if percent >= 50 else '#FEE2E2'
        text_color = '#166534' if percent >= 80 else '#854D0E' if percent >= 50 else '#991B1B'
        
        # 스타일 함수 호출을 변수로 할당
        card_bg = get_bg_style('white')
        card_border = get_border_style('default')
        title_style = get_text_style('md', 'tertiary')
        percent_style = get_text_style('5xl', 'primary', family='bold')
        badge_style = get_text_style('sm', weight='semibold')
        label_style = get_text_style('base', 'tertiary')
        value_style = get_text_style('lg', 'primary', weight='semibold')
        desc_style = get_text_style('base', 'tertiary')
        
        # 폰트 패밀리
        font_family = "font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', 'Malgun Gothic', sans-serif !important;"
        
        # 이미지와 결과를 가로로 배치하는 HTML 레이아웃
        layout_html = f"""
        <div style="display: flex; flex-direction: row; gap: 1rem; width: 100%; align-items: flex-start;">
            <div style="flex: 1 1 0%; min-width: 0;">
                <img src="data:image/jpeg;base64,{img_str}" style="width: 100%; height: auto; object-fit: contain; border-radius: 8px;" />
            </div>
            <div style="flex: 1.5 1 0%; min-width: 0;">
                <div style="{card_bg} {card_border} border-radius: {BORDER_RADIUS['lg']}; padding: {SPACING['2xl']}; box-shadow: 0 1px 2px rgba(0,0,0,0.05); {font_family}">
                    <div style="{title_style} margin-bottom: {SPACING['sm']}; {font_family}">예측 성과</div>
                    <div style="{percent_style} margin-bottom: {SPACING['md']}; {font_family}">{percent:.1f}%</div>
                    <div style="display: inline-block; padding: {SPACING['xs']} {SPACING['md']}; border-radius: {BORDER_RADIUS['sm']}; background-color: {bg_color}; color: {text_color}; {badge_style} margin-bottom: {SPACING['lg']}; {font_family}">{level}</div>
                    <div style="border-top: 1px solid #E5E7EB; padding-top: {SPACING['lg']}; margin-top: {SPACING['lg']};">
                        <div style="{label_style} margin-bottom: {SPACING['xs']}; {font_family}">이미지 타입</div>
                        <div style="{value_style} margin-bottom: {SPACING['lg']}; {font_family}">Type {img_type} · {type_name}</div>
                        <div style="{desc_style} line-height: 1.6; {font_family}">이 이미지는 <strong>{selected_country}</strong> 시장 내 전체 콘텐츠 대비 <strong>{level}</strong> 수준의 상대적 성과 위치에 해당합니다.</div>
                    </div>
                </div>
            </div>
        </div>
        """
        
        # HTML 렌더링
        try:
            st.html(layout_html)
        except (AttributeError, Exception):
            st.markdown(layout_html, unsafe_allow_html=True)
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
