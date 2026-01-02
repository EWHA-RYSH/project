import streamlit as st
import os
from typing import Optional
from utils.charts import get_country_name

TYPE_DESC = {
    1: "제품 단체샷",
    2: "제품 단독샷",
    3: "제품 질감샷",
    4: "제품+모델",
    5: "라이프스타일",
    6: "혼합형"
}


def render_page_header(title, country=None, n_posts=None, countries=None, selected_country=None, description=None, subtitle=None):
    st.markdown(
        f"""
        <div class="page-title" style="
            font-family: 'Arita-Dotum-Bold', 'Arita-dotum-Medium', 'Arita-Dotum-Medium', sans-serif !important;
            font-size: 1.75rem;
            font-weight: 800 !important;
            color: #1F2937;
            margin-bottom: 16px;
            line-height: 1.4;
            letter-spacing: -0.02em;
        ">
            {title}
        </div>
        """,
        unsafe_allow_html=True
    )
    
    if description:
        st.markdown(
            f"""
            <div class="page-description" style="
                font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;
                font-size: 13px;
                color: #6B7280;
                font-weight: 400;
                line-height: 1.6;
                margin-top: 0;
                margin-bottom: 36px;
            ">
                {description}
            </div>
            """,
            unsafe_allow_html=True
        )
    
    if countries and selected_country:
        st.markdown(
            f"""
            <div style="
                font-size: 12px;
                color: #6B7280;
                margin-bottom: 10px;
            ">
                분석 대상
            </div>
            """,
            unsafe_allow_html=True
        )
        new_country = st.selectbox(
            "",
            countries,
            index=countries.index(selected_country) if selected_country in countries else 0,
            label_visibility="collapsed",
            format_func=lambda x: get_country_name(x),
            key=f"country_selector_{title.replace(' ', '_')}"
        )
        st.session_state.selected_country = new_country

def render_kpi_card(label, value, subtext=None, highlight=False):
    highlight_style = "border-left: 4px solid #1F5795;" if highlight else ""
    subtext_html = f'<div class="kpi-subtext" style="font-size: 12px; color: #9CA3AF; margin-top: 4px;">{subtext}</div>' if subtext else ''
    
    st.markdown(
        f'<div class="kpi-card-wrapper" style="background-color: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 8px; padding: 20px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); {highlight_style} width: 100%; box-sizing: border-box;"><div class="kpi-label" style="font-size: 13px; color: #6B7280; margin-bottom: 8px;">{label}</div><div class="kpi-value" style="font-size: 18px; font-weight: 700; color: #1F2937;">{value}</div>{subtext_html}</div>',
        unsafe_allow_html=True
    )

def render_insight_box(bullets):
    bullets_html = "".join([f"<li style='margin-bottom: 8px; font-family: \'Arita-Dotum-Medium\', \'Arita-dotum-Medium\', sans-serif !important;'>{bullet}</li>" for bullet in bullets])
    
    st.markdown(
        f"""
        <div style="
            background-color: #F9FAFB;
            border-left: 4px solid #1F5795;
            border-radius: 4px;
            padding: 16px 20px;
            margin: 20px 0;
            font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;
        ">
            <div style="font-size: 14px; font-weight: 600; color: #1F2937; margin-bottom: 12px; font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;">
                주요 인사이트
            </div>
            <ul style="margin: 0; padding-left: 20px; color: #374151; font-size: 14px; line-height: 1.6; font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;">
                {bullets_html}
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_insight_bullets(bullets: list[str], title: Optional[str] = None):
    """
    인사이트 불릿 리스트를 렌더링합니다.
    
    Args:
        bullets: 인사이트 불릿 리스트 (HTML 포함 가능)
        title: 선택적 제목 (없으면 제목 없이 렌더링)
    """
    if not bullets or len(bullets) == 0:
        return
    
    # bullets를 위계에 따라 HTML로 변환
    bullets_html = ""
    for bullet in bullets:
        # 🧠로 시작: 보조 설명 톤
        if bullet.strip().startswith("🧠"):
            bullets_html += f'<div style="margin-bottom: 10px; font-family: Arita-Dotum-Medium, Arita-dotum-Medium, sans-serif !important; color: #111827; font-size: 14px; line-height: 1.6;">{bullet}</div>'
        # 📍로 시작: 결론
        elif bullet.strip().startswith("📍"):
            bullets_html += f'<div style="margin-bottom: 12px; font-family: Arita-Dotum-Medium, Arita-dotum-Medium, sans-serif !important; color: #111827; font-size: 14px; line-height: 1.6;">{bullet}</div>'
        # 기타: 기본 스타일
        else:
            bullets_html += f'<div style="margin-bottom: 8px; font-family: Arita-Dotum-Medium, Arita-dotum-Medium, sans-serif !important; color: #374151; font-size: 14px; line-height: 1.6;">{bullet}</div>'
    
    title_html = ""
    if title:
        title_html = f'<div style="font-size: 18px; font-weight: 700; color: #1F2937; margin-bottom: 16px; font-family: Arita-Dotum-Bold, Arita-Dotum-Medium, Arita-dotum-Medium, sans-serif !important;">{title}</div>'
    
    # HTML 구성 (왼쪽 라인: 연한 회색, 두께 2px)
    html_content = f'''<div style="background-color: #FFFFFF; border: 1px solid #E5E7EB; border-left: 2px solid #D1D5DB; border-radius: 4px; padding: 20px 24px; margin: 20px 0; box-shadow: 0 1px 2px rgba(0,0,0,0.05); font-family: Arita-Dotum-Medium, Arita-dotum-Medium, sans-serif !important;">
{title_html}
<div style="margin: 0; font-family: Arita-Dotum-Medium, Arita-dotum-Medium, sans-serif !important;">
{bullets_html}
</div>
</div>'''
    
    # st.html 사용 (Streamlit 1.28.0+)
    try:
        st.html(html_content)
    except AttributeError:
        # st.html이 없는 경우 st.markdown 사용
        st.markdown(html_content, unsafe_allow_html=True)

def render_action_items(items):
    items_html = "".join([
        f"<li style='margin-bottom: 12px; font-family: \'Arita-Dotum-Medium\', \'Arita-dotum-Medium\', sans-serif !important;'><strong style='font-family: \'Arita-Dotum-Medium\', \'Arita-dotum-Medium\', sans-serif !important;'>{item['action']}:</strong> {item['reason']}</li>"
        for item in items
    ])
    
    st.markdown(
        f"""
        <div style="
            background-color: #F9FAFB;
            border-left: 4px solid #1F5795;
            border-radius: 4px;
            padding: 16px 20px;
            margin: 20px 0;
            font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;
        ">
            <div style="font-size: 14px; font-weight: 600; color: #1F2937; margin-bottom: 12px; font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;">
                권장 조치사항
            </div>
            <ul style="margin: 0; padding-left: 20px; color: #374151; font-size: 14px; line-height: 1.6; font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;">
                {items_html}
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

def get_type_name(img_type):
    return TYPE_DESC.get(int(img_type), f"Type {img_type}")

def section_gap(height=40):
    st.markdown(
        f"<div style='height:{height}px'></div>",
        unsafe_allow_html=True
    )

def render_image_type_guide():
    import base64
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assets_dir = os.path.join(base_dir, "assets")
    
    type_data = [
        (1, "제품 단체샷", "여러 제품을 함께 배치한 이미지"),
        (2, "제품 단독샷", "하나의 제품을 중심으로 구성한 이미지"),
        (3, "제품 질감샷", "질감·패키지 디테일을 강조한 이미지"),
        (4, "제품 + 모델", "모델과 제품을 함께 배치한 이미지"),
        (5, "라이프스타일", "일상 맥락 속에서 제품을 노출한 이미지"),
        (6, "혼합형", "명확한 분류가 어려운 혼합 구성 이미지"),
    ]
    
    cards_html = ""
    for type_num, type_name, type_desc in type_data:
        img_path = os.path.join(assets_dir, f"{type_num}.jpg")
        b64_img = ""
        
        if os.path.exists(img_path):
            with open(img_path, "rb") as f:
                img_data = f.read()
                b64_img = base64.b64encode(img_data).decode()
        
        if b64_img:
            img_tag = f'<img src="data:image/jpeg;base64,{b64_img}" alt="Type {type_num}" style="width: 100%; height: 100%; object-fit: cover; display: block;" />'
        else:
            img_tag = '<div style="display: flex; align-items: center; justify-content: center; height: 100%; color: #9CA3AF; font-size: 12px;">이미지 없음</div>'
        
        cards_html += f'<div class="type-card"><div class="type-card-header"><span class="type-chip">Type {type_num}</span><span class="type-title">{type_name}</span></div><div class="type-image-wrapper">{img_tag}</div><div class="type-description">{type_desc}</div></div>'
    
    html_content = f"""<div class="type-guide">
<div class="type-grid">
{cards_html}
</div>
</div>"""
    
    st.markdown(html_content, unsafe_allow_html=True)

