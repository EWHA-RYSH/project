import streamlit as st
import os
from typing import Optional
from utils.charts import get_country_name
from components.design_tokens import (
    FONT_SIZES, TEXT_COLORS, FONT_WEIGHTS, FONT_FAMILIES,
    BG_COLORS, BORDER_COLORS, SPACING, BORDER_RADIUS, BRAND_COLORS,
    get_text_style, get_bg_style, get_border_style
)

TYPE_DESC = {
    1: "제품 단체샷",
    2: "제품 단독샷",
    3: "제품 질감샷",
    4: "제품+모델",
    5: "제품 없는 모델샷",
    6: "제품 모델 단체샷"
}


def render_page_header(title, country=None, n_posts=None, countries=None, selected_country=None, description=None, subtitle=None):
    st.markdown(
        f"""
        <div class="page-title" style="{get_text_style('xl', 'primary', weight='extrabold', family='bold')} font-size: 1.75rem; margin-bottom: {SPACING['lg']}; line-height: 1.4; letter-spacing: -0.02em;">
            {title}
        </div>
        """,
        unsafe_allow_html=True
    )
    
    if description:
        st.markdown(
            f"""
            <div class="page-description" style="{get_text_style('base', 'tertiary', 'normal', 'medium')} line-height: 1.6; margin-top: 0; margin-bottom: 36px;">
                {description}
            </div>
            """,
            unsafe_allow_html=True
        )
    
    if countries and selected_country:
        st.markdown(
            f"""
            <div style="{get_text_style('sm', 'tertiary')} margin-bottom: 10px;">
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
    highlight_style = f"border-left: 4px solid {BRAND_COLORS['primary']};" if highlight else ""
    subtext_html = f'<div class="kpi-subtext" style="{get_text_style("sm", "muted")} margin-top: auto; padding-top: {SPACING["sm"]};">{subtext}</div>' if subtext else ''
    
    st.markdown(
        f'<div class="kpi-card-wrapper" style="{get_bg_style("white")} {get_border_style("default")} border-radius: {BORDER_RADIUS["md"]}; padding: {SPACING["xl"]}; box-shadow: 0 1px 2px rgba(0,0,0,0.05); {highlight_style} width: 100%; box-sizing: border-box; height: 100%; min-height: 140px; display: flex; flex-direction: column; overflow: visible;"><div style="flex: 1;"><div class="kpi-label" style="{get_text_style("base", "tertiary")} margin-bottom: {SPACING["sm"]};">{label}</div><div class="kpi-value" style="{get_text_style("xl", "primary", family="bold")}">{value}</div></div>{subtext_html}</div>',
        unsafe_allow_html=True
    )

def render_insight_box(bullets):
    bullets_html = "".join([f"<li style='margin-bottom: {SPACING['sm']}; {get_text_style('md', 'secondary', 'normal', 'medium')}'>{bullet}</li>" for bullet in bullets])
    
    st.markdown(
        f"""
        <div style="{get_bg_style('light')} border-left: 4px solid {BRAND_COLORS['primary']}; border-radius: {BORDER_RADIUS['sm']}; padding: {SPACING['lg']} {SPACING['xl']}; margin: {SPACING['xl']} 0; {get_text_style('md', 'primary', 'normal', 'medium')}">
            <div style="{get_text_style('md', 'primary', 'semibold', 'medium')} margin-bottom: {SPACING['md']};">
                주요 인사이트
            </div>
            <ul style="margin: 0; padding-left: {SPACING['xl']}; {get_text_style('md', 'secondary', 'normal', 'medium')} line-height: 1.6;">
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
        bullet_clean = bullet.strip()
        
        # 📍 전략 요약: 이모지 + <b>전략 요약</b> + 설명
        if bullet_clean.startswith("📍"):
            # 이모지와 <b> 태그 제거하고 텍스트 추출
            text = bullet_clean.replace("📍", "").strip()
            # <b>전략 요약</b> 또는 <b>전략 요약</b>: 패턴 처리
            if "<b>전략 요약</b>" in text or "<b>전략 요약</b>:" in text:
                label = "전략 요약"
                desc = text.replace("<b>전략 요약</b>", "").replace("<b>전략 요약</b>:", "").replace(":", "").strip()
                label_style = get_text_style("md", "primary", "bold", "bold")
                desc_style = get_text_style("md", "secondary", "normal", "medium")
                bullets_html += '<div style="margin-bottom: ' + SPACING["lg"] + ';"><div style="display: flex; align-items: flex-start; gap: ' + SPACING["xs"] + '; margin-bottom: ' + SPACING["xs"] + ';"><span style="font-size: ' + FONT_SIZES["md"] + '; font-family: \'Arita-Dotum-Medium\', \'Arita-dotum-Medium\', \'Malgun Gothic\', sans-serif !important;">📌</span><div style="' + label_style + ' font-weight: 800; font-family: \'Arita-Dotum-Bold\', \'Arita-Dotum-Medium\', \'Arita-dotum-Medium\', sans-serif !important;">' + label + ':</div></div><div style="padding-left: ' + SPACING["xl"] + '; ' + desc_style + ' line-height: 1.6; font-family: \'Arita-Dotum-Medium\', \'Arita-dotum-Medium\', \'Malgun Gothic\', sans-serif !important;">' + desc + '</div></div>'
            else:
                text_style = get_text_style("md", "primary", "normal", "medium")
                bullets_html += '<div style="margin-bottom: ' + SPACING["md"] + '; ' + text_style + ' line-height: 1.6; font-family: \'Arita-Dotum-Medium\', \'Arita-dotum-Medium\', \'Malgun Gothic\', sans-serif !important;">' + text + '</div>'
        
        # 🧠 분석: 이모지 + <b>분석</b> + 설명
        elif bullet_clean.startswith("🧠"):
            text = bullet_clean.replace("🧠", "").strip()
            if "<b>분석</b>" in text:
                label = "분석"
                desc = text.replace("<b>분석</b>", "").replace("<b>분석</b>:", "").replace(":", "").strip()
                label_style = get_text_style("md", "primary", "bold", "bold")
                desc_style = get_text_style("md", "secondary", "normal", "medium")
                bullets_html += '<div style="margin-bottom: 0;"><div style="display: flex; align-items: flex-start; gap: ' + SPACING["xs"] + '; margin-bottom: ' + SPACING["xs"] + ';"><span style="font-size: ' + FONT_SIZES["md"] + '; font-family: \'Arita-Dotum-Medium\', \'Arita-dotum-Medium\', \'Malgun Gothic\', sans-serif !important;">🧠</span><div style="' + label_style + ' font-weight: 800; font-family: \'Arita-Dotum-Bold\', \'Arita-Dotum-Medium\', \'Arita-dotum-Medium\', sans-serif !important;">' + label + '</div></div><div style="padding-left: ' + SPACING["xl"] + '; ' + desc_style + ' line-height: 1.6; font-family: \'Arita-Dotum-Medium\', \'Arita-dotum-Medium\', \'Malgun Gothic\', sans-serif !important;">' + desc + '</div></div>'
            else:
                text_style = get_text_style("md", "primary", "normal", "medium")
                bullets_html += '<div style="margin-bottom: ' + SPACING["lg"] + '; ' + text_style + ' line-height: 1.6; font-family: \'Arita-Dotum-Medium\', \'Arita-dotum-Medium\', \'Malgun Gothic\', sans-serif !important;">' + text + '</div>'
        
        # 👉로 시작: 요약 문장 (볼드 적용)
        elif bullet_clean.startswith("👉"):
            text_style = get_text_style("md", "primary", family="bold")
            bullets_html += '<div style="margin-bottom: ' + SPACING["md"] + '; ' + text_style + ' line-height: 1.6; font-family: \'Arita-Dotum-Medium\', \'Arita-dotum-Medium\', \'Malgun Gothic\', sans-serif !important;">' + bullet + '</div>'
        
        # 🔎로 시작: 분석 항목 (볼드 해제)
        elif "🔎" in bullet:
            # <b> 태그 제거하고 일반 스타일 적용
            bullet_text = bullet.replace("<b>", "").replace("</b>", "")
            text_style = get_text_style("md", "secondary", "normal", "medium")
            bullets_html += '<div style="margin-bottom: ' + SPACING["sm"] + '; ' + text_style + ' line-height: 1.6; font-family: \'Arita-Dotum-Medium\', \'Arita-dotum-Medium\', \'Malgun Gothic\', sans-serif !important;">' + bullet_text + '</div>'
        
        # 기타: 기본 스타일
        else:
            text_style = get_text_style("md", "secondary", "normal", "medium")
            bullets_html += '<div style="margin-bottom: ' + SPACING["sm"] + '; ' + text_style + ' line-height: 1.6; font-family: \'Arita-Dotum-Medium\', \'Arita-dotum-Medium\', \'Malgun Gothic\', sans-serif !important;">' + bullet + '</div>'
    
    # 제목을 박스 밖으로 빼기 (참여율 분포, 활용 분포처럼)
    if title:
        # 제목을 먼저 렌더링 (박스 밖) - section-title 클래스 사용하여 동일한 스타일 적용
        title_html = '<div class="section"><h4 class="section-title">' + title + '</h4></div>'
        st.markdown(title_html, unsafe_allow_html=True)
    
    # HTML 구성 (패턴 요약과 같은 스타일: 파란색 배경, 파란색 왼쪽 테두리)
    # 폰트를 더 강력하게 적용하기 위해 CSS 스타일 태그 추가
    css_style = '<style>.insight-container, .insight-container *, .insight-container span, .insight-container div { font-family: \'Arita-Dotum-Medium\', \'Arita-dotum-Medium\', \'Malgun Gothic\', sans-serif !important; }</style>'
    container_style = "background-color: rgba(31, 87, 149, 0.06); border-left: 4px solid " + BRAND_COLORS["primary"] + "; border-radius: " + BORDER_RADIUS["sm"] + "; padding: " + SPACING["lg"] + " " + SPACING["xl"] + "; margin: 0 0 " + SPACING["xl"] + " 0; font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', 'Malgun Gothic', sans-serif !important;"
    inner_style = "margin: 0; font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', 'Malgun Gothic', sans-serif !important;"
    html_content = css_style + '<div class="insight-container" style="' + container_style + '"><div style="' + inner_style + '">' + bullets_html + '</div></div>'
    
    # st.markdown 사용 (st.html이 HTML을 텍스트로 표시하는 문제 해결)
    st.markdown(html_content, unsafe_allow_html=True)

def render_action_items(items):
    items_html = "".join([
        f"<li style='margin-bottom: {SPACING['md']}; {get_text_style('md', 'secondary', 'normal', 'medium')}'><strong style='{get_text_style('md', 'secondary', 'normal', 'medium')}'>{item['action']}:</strong> {item['reason']}</li>"
        for item in items
    ])
    
    st.markdown(
        f"""
        <div style="{get_bg_style('light')} border-left: 4px solid {BRAND_COLORS['primary']}; border-radius: {BORDER_RADIUS['sm']}; padding: {SPACING['lg']} {SPACING['xl']}; margin: {SPACING['xl']} 0; {get_text_style('md', 'primary', 'normal', 'medium')}">
            <div style="{get_text_style('md', 'primary', 'semibold', 'medium')} margin-bottom: {SPACING['md']};">
                권장 조치사항
            </div>
            <ul style="margin: 0; padding-left: {SPACING['xl']}; {get_text_style('md', 'secondary', 'normal', 'medium')} line-height: 1.6;">
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
        (5, "제품 없는 모델샷", "제품 없이 모델 중심으로 연출된 이미지"),
        (6, "제품 모델 단체샷", "여러 인물과 제품을 함께 보여주는 이미지"),
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
            img_tag = f'<div style="display: flex; align-items: center; justify-content: center; height: 100%; {get_text_style("sm", "muted")}">이미지 없음</div>'
        
        cards_html += f'<div class="type-card"><div class="type-card-header"><span class="type-chip">Type {type_num}</span><span class="type-title">{type_name}</span></div><div class="type-image-wrapper">{img_tag}</div><div class="type-description">{type_desc}</div></div>'
    
    html_content = f"""<div class="type-guide">
<div class="type-grid">
{cards_html}
</div>
</div>"""
    
    st.markdown(html_content, unsafe_allow_html=True)

