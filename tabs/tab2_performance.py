import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re

from utils.data_loader import load_meta_df
from components.design_tokens import (
    get_text_style, get_bg_style, get_border_style, TEXT_COLORS, FONT_SIZES, 
    SPACING, BRAND_COLORS, FONT_WEIGHTS, FONT_FAMILIES, BORDER_RADIUS, BORDER_COLORS, BG_COLORS
)
from utils.eda_metrics import (
    preprocess_country_data,
    get_image_type_distribution,
    get_performance_summary,
    get_top_percentile_metrics,
    get_stability_metrics,
    get_response_characteristics,
    get_usage_vs_performance
)
from utils.metrics import (
    compute_performance_kpis,
    format_percentage,
    format_engagement_rate
)
from utils.charts import plot_usage_vs_engagement, apply_chart_style, BRAND_COLORS, CHART_PALETTE, LIGHT_BLUE_HIGHLIGHT, DEFAULT_BAR_COLOR, MEDIAN_COLOR, MEAN_COLOR
from utils.insights_store import load_tab_insights
from components.layout import (
    render_page_header,
    render_kpi_card,
    render_action_items,
    render_insight_bullets,
    get_type_name,
    render_image_type_guide,
    section_gap
)

def render():
    # JSON 인사이트 로드
    insights = load_tab_insights("tab2")
    
    df_meta = load_meta_df()
    selected_country = st.session_state.get("selected_country", sorted(df_meta["country"].unique())[0])
    df_country = preprocess_country_data(df_meta, selected_country)
    
    if len(df_country) == 0:
        st.warning(f"{selected_country}에 대한 데이터가 없습니다.")
        return
    
    # 페이지 헤더 (국가 선택기 포함)
    countries = sorted(df_meta["country"].unique())
    render_page_header(
        "성과 분석",
        countries=countries,
        selected_country=selected_country,
        n_posts=len(df_country),
        description="국가별 콘텐츠 성과 데이터를 기반으로 이미지 유형별 참여 패턴과 활용 효율을 비교하여 "
                    "성과가 높은 콘텐츠 유형과 최적화 기회를 도출합니다."
    )
    
    current_country = st.session_state.get("selected_country", selected_country)
    if current_country != selected_country:
        selected_country = current_country
        df_country = preprocess_country_data(df_meta, selected_country)
        if len(df_country) == 0:
            st.warning(f"{selected_country}에 대한 데이터가 없습니다.")
            return
    
    section_gap(16)
    with st.expander("📁 이미지 유형 기준", expanded=False):
        st.markdown(
            f"""
            <div style="{get_text_style('md', 'tertiary')} line-height: 1.6; margin-bottom: {SPACING['xl']};">
                Type 1~6은 게시물의 이미지 구성 방식이며, KPI 해석/성과 비교의 기준으로 사용됩니다.<br>
            </div>
            """,
            unsafe_allow_html=True
        )
        render_image_type_guide()
    
    section_gap(24)
    
    kpis = compute_performance_kpis(df_country)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if kpis['best_engagement']['type']:
            best_name = get_type_name(kpis['best_engagement']['type'])
            render_kpi_card(
                "최고 참여율 타입",
                f"{best_name}",
                subtext=f"Type {kpis['best_engagement']['type']} · 참여율: {format_engagement_rate(kpis['best_engagement']['value'])}",
                highlight=True
            )
        else:
            render_kpi_card("최고 참여율 타입", "N/A")
    
    with col2:
        if kpis['underused_opportunity']['type']:
            underused_name = get_type_name(kpis['underused_opportunity']['type'])
            render_kpi_card(
                "과소 활용 기회",
                f"{underused_name}",
                subtext=f"Type {kpis['underused_opportunity']['type']} · 높은 참여율({format_engagement_rate(kpis['underused_opportunity']['engagement'])})이나 낮은 활용도({format_percentage(kpis['underused_opportunity']['usage'])})"
            )
        else:
            render_kpi_card("과소 활용 기회", "N/A")
    
    with col3:
        stability_label = "안정적" if kpis['stability']['label'] == "Stable" else "변동적" if kpis['stability']['label'] == "Volatile" else kpis['stability']['label']
        render_kpi_card(
            "안정성",
            stability_label,
            subtext="성과 일관성"
        )
    
    section_gap(48)
    
    type_count, type_ratio = get_image_type_distribution(df_country)
    
    # 4개 탭으로 구성
    tab1, tab2, tab3, tab4 = st.tabs([
        "성과 비교・반응 성격",
        "고성과 분석",
        "안정성 분석",
        "전략 인사이트"
    ])
    
    # ============================================
    # 탭 1: 성과 비교・반응 성격
    # ============================================
    with tab1:
        perf_summary = get_performance_summary(df_country)
        response_char = get_response_characteristics(df_country)
        country_insight = insights.get(selected_country, {})
        strategy_insights = country_insight.get("strategy_insights", {})
        performance_bullets = country_insight.get("performance_comparison", {}).get("bullets", [])
        
        # 참여율 분포
        st.markdown(
            """
            <div class="section">
                <h4 class="section-title">참여율 분포</h4>
                <div class="section-desc">이미지 타입별 참여율의 분포를 비교하고,
유형별 반응 수준과 변동 폭을 함께 확인합니다.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        section_gap(16)
        
        # 상단 결론 배너
        if len(perf_summary) > 0:
            max_idx = perf_summary["eng_mean"].idxmax()
            max_type = int(perf_summary.loc[max_idx, "img_type"])
            max_value = perf_summary.loc[max_idx, "eng_mean"]
            max_name = get_type_name(max_type)
            
            # 성과 구조 해석에서 핵심 결론 추출
            conclusion_text = f"Type {max_type}({max_name})가 참여율 {format_engagement_rate(max_value)}로 최고 성과를 기록합니다."
            if performance_bullets:
                for bullet in performance_bullets:
                    bullet_clean = bullet.strip()
                    if ("상위 콘텐츠" in bullet_clean or "성과 지표" in bullet_clean or "비선형" in bullet_clean or 
                        "참여율과 반응 지표" in bullet_clean or "다수 콘텐츠의 누적" in bullet_clean):
                        # 핵심 문장만 추출 (1-2문장), 마침표 보존
                        sentences = bullet_clean.split('.')
                        if len(sentences) > 0:
                            conclusion_text = sentences[0].strip()
                            if not conclusion_text.endswith('.'):
                                conclusion_text += '.'
                            if len(sentences) > 1 and len(conclusion_text) < 80:
                                second_sentence = sentences[1].strip()
                                if second_sentence:
                                    if not second_sentence.endswith('.'):
                                        second_sentence += '.'
                                    conclusion_text += ' ' + second_sentence
                        break
            
            st.markdown(
                f"""
                <div style="background-color: rgba(31, 87, 149, 0.08); border-left: 4px solid {BRAND_COLORS['primary']}; padding: {SPACING['md']} {SPACING['lg']}; margin-bottom: {SPACING['lg']};">
                    <div style="font-size: {FONT_SIZES['md']}; font-weight: 400; color: {TEXT_COLORS['primary']}; line-height: 1.6; font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', 'Malgun Gothic', sans-serif !important;">
                        {conclusion_text}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # 차트
        if len(perf_summary) > 0:
            # Top 1만 연한 블루로 강조
            max_idx = perf_summary["eng_mean"].idxmax()
            colors = []
            text_values = []
            for idx, row in perf_summary.iterrows():
                if idx == max_idx:
                    colors.append(LIGHT_BLUE_HIGHLIGHT)  # Top 1만 연한 블루
                else:
                    colors.append(DEFAULT_BAR_COLOR)  # 나머지는 #E1E4EA
                # 값 라벨 추가 (참여율은 소수점 표시)
                text_values.append(f"{row['eng_mean']:.4f}")
            
            fig = px.bar(
                perf_summary,
                x="img_type",
                y="eng_mean",
                labels={"img_type": "이미지 타입", "eng_mean": ""},
                title="이미지 타입별 평균 참여율",
                text=text_values
            )
            fig.update_traces(
                marker_color=colors, 
                width=0.6,
                textposition="outside",
                textfont=dict(size=11, color="#6B7280", family="Arita-Dotum-Medium, Arita-dotum-Medium, sans-serif")
            )
            fig = apply_chart_style(fig)
            fig.update_layout(
                bargap=0.4, 
                showlegend=False, 
                height=400,
                yaxis=dict(title=None),
                margin=dict(l=40, r=20, t=70, b=40),
                title=dict(
                    x=0.5,
                    xanchor="center",
                    y=0.94,
                    yanchor="top",
                    font=dict(size=17, color="#111827", family="Arita-Dotum-Medium, Arita-dotum-Medium, sans-serif")
                )
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=f"tab1_engagement_{selected_country}")
        
        section_gap(48)
        
        # 좋아요/댓글 수 분포
        st.markdown(
            """
            <div class="section">
                <h4 class="section-title">좋아요・댓글 분포</h4>
                <div class="section-desc">이미지 타입별 좋아요와 댓글 수의 분포를 비교하여,
각 유형의 절대적 반응 규모와 분산 정도를 파악합니다.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        section_gap(16)
        
        # 상단 결론 배너
        if len(perf_summary) > 0:
            # 반응 성격 분석에서 핵심 결론 추출
            conclusion_text = "좋아요와 댓글 반응 패턴을 확인합니다."
            if performance_bullets:
                for bullet in performance_bullets:
                    bullet_clean = bullet.strip()
                    if ("좋아요와 댓글" in bullet_clean or "확산형 반응" in bullet_clean or 
                        "반응의 양과 질" in bullet_clean or "좋아요 중심" in bullet_clean or 
                        "댓글 기여도" in bullet_clean):
                        # 핵심 문장만 추출 (1-2문장), 마침표 보존
                        sentences = bullet_clean.split('.')
                        if len(sentences) > 0:
                            conclusion_text = sentences[0].strip()
                            if not conclusion_text.endswith('.'):
                                conclusion_text += '.'
                            if len(sentences) > 1 and len(conclusion_text) < 80:
                                second_sentence = sentences[1].strip()
                                if second_sentence:
                                    if not second_sentence.endswith('.'):
                                        second_sentence += '.'
                                    conclusion_text += ' ' + second_sentence
                        break
            
            st.markdown(
                f"""
                <div style="background-color: rgba(31, 87, 149, 0.08); border-left: 4px solid {BRAND_COLORS['primary']}; padding: {SPACING['md']} {SPACING['lg']}; margin-bottom: {SPACING['lg']};">
                    <div style="font-size: {FONT_SIZES['md']}; font-weight: 400; color: {TEXT_COLORS['primary']}; line-height: 1.6; font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', 'Malgun Gothic', sans-serif !important;">
                        {conclusion_text}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # 차트
        col1, col2 = st.columns(2)
        with col1:
            if len(perf_summary) > 0:
                # 평균과 중앙값 모두 막대로 표시 
                fig1 = px.bar(
                    perf_summary,
                    x="img_type",
                    y=["likes_mean", "likes_median"],
                    labels={"img_type": "이미지 타입", "value": "", "variable": ""},
                    title="좋아요 수",
                    barmode="group",
                    color_discrete_map={"likes_mean": MEAN_COLOR, "likes_median": MEDIAN_COLOR}
                )
                # 평균은 진한 회색, 중앙값은 연한 회색 (댓글 수 차트와 동일)
                if len(fig1.data) >= 2:
                    fig1.data[0].marker.color = MEAN_COLOR  # 평균 - #9CA3AF
                    fig1.data[0].name = "평균"
                    fig1.data[1].marker.color = MEDIAN_COLOR  # 중앙값 - #E5E7EB
                    fig1.data[1].name = "중앙값"
                fig1.update_traces(width=0.6)
                fig1 = apply_chart_style(fig1)
                fig1.update_layout(
                    bargap=0.4, 
                    height=400,
                    showlegend=True,
                    yaxis=dict(title=None),
                    margin=dict(l=40, r=40, t=70, b=60),
                    title=dict(
                        x=0.5,
                        xanchor="center",
                        y=0.94,
                        yanchor="top",
                        font=dict(size=17, color="#111827", family="Arita-Dotum-Medium, Arita-dotum-Medium, sans-serif")
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="top",
                        y=-0.15,
                        xanchor="left",
                        x=0,
                        font=dict(family="Arita-Dotum-Medium, Arita-dotum-Medium, sans-serif", size=12),
                        itemwidth=30,
                        tracegroupgap=5,
                        itemsizing="constant",
                        bgcolor="rgba(255,255,255,0)",
                        bordercolor="rgba(255,255,255,0)"
                    )
                )
                st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False}, key=f"tab1_likes_{selected_country}")
        
        with col2:
            if len(perf_summary) > 0:
                # 댓글 수는 둘 다 막대 유지, 색 대비 더 벌리기
                fig2 = px.bar(
                    perf_summary,
                    x="img_type",
                    y=["comments_mean", "comments_median"],
                    labels={"img_type": "이미지 타입", "value": "", "variable": ""},
                    title="댓글 수",
                    barmode="group",
                    color_discrete_map={"comments_mean": CHART_PALETTE[2], "comments_median": CHART_PALETTE[6]}
                )
                # 평균은 #9CA3AF, 중앙값은 #E5E7EB
                if len(fig2.data) >= 2:
                    fig2.data[0].marker.color = MEAN_COLOR  # 평균 - #9CA3AF
                    fig2.data[0].name = "평균"
                    fig2.data[1].marker.color = MEDIAN_COLOR  # 중앙값 - #E5E7EB
                    fig2.data[1].name = "중앙값"
                fig2.update_traces(width=0.5)  # 막대 폭 약간 줄이기
                fig2 = apply_chart_style(fig2)
                fig2.update_layout(
                    bargap=0.4, 
                    height=400,
                    showlegend=True,
                    yaxis=dict(title=None),
                    margin=dict(l=40, r=40, t=70, b=60),
                    title=dict(
                        x=0.5,
                        xanchor="center",
                        y=0.94,
                        yanchor="top",
                        font=dict(size=17, color="#111827", family="Arita-Dotum-Medium, Arita-dotum-Medium, sans-serif")
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="top",
                        y=-0.15,
                        xanchor="left",
                        x=0,
                        font=dict(family="Arita-Dotum-Medium, Arita-dotum-Medium, sans-serif", size=12),
                        itemwidth=30,
                        tracegroupgap=5,
                        itemsizing="constant",
                        bgcolor="rgba(255,255,255,0)",
                        bordercolor="rgba(255,255,255,0)"
                    )
                )
                st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False}, key=f"tab1_comments_{selected_country}")
        
        section_gap(48)
        
        # 활용도 vs 성과 분석
        st.markdown(
            """
            <div class="section">
                <h4 class="section-title">활용도 vs 성과 분석</h4>
                <div class="section-desc">각 이미지 유형의 활용 비중과 실제 성과를 비교하여 운영 효율성을 분석합니다.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        section_gap(16)
        
        # 상단 결론 배너
        usage_vs_perf_data = strategy_insights.get("usage_vs_performance", {})
        if usage_vs_perf_data:
            # 핵심 판단 문장 추출
            key_insight = ""
            if usage_vs_perf_data.get("comparison_analysis"):
                text = usage_vs_perf_data["comparison_analysis"]
                if "📍" in text:
                    key_insight = text.split(":", 1)[1].strip() if ":" in text else text.replace("📍", "").strip()
            elif usage_vs_perf_data.get("actual_performance"):
                text = usage_vs_perf_data["actual_performance"]
                if "🏆" in text:
                    key_insight = text.split(":", 1)[1].strip() if ":" in text else text.replace("🏆", "").strip()
            
            if key_insight:
                # 마침표 확인 및 추가
                if not key_insight.endswith('.'):
                    key_insight += '.'
                
                st.markdown(
                    f"""
                    <div style="background-color: rgba(31, 87, 149, 0.08); border-left: 4px solid {BRAND_COLORS['primary']}; padding: {SPACING['md']} {SPACING['lg']}; margin-bottom: {SPACING['lg']};">
                        <div style="font-size: {FONT_SIZES['md']}; font-weight: 400; color: {TEXT_COLORS['primary']}; line-height: 1.6; font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', 'Malgun Gothic', sans-serif !important;">
                            {key_insight}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        
        # 차트
        perf_summary = get_performance_summary(df_country)
        plot_usage_vs_engagement(
            type_ratio,
            perf_summary,
            selected_country,
            key_suffix="tab1"
        )
        
        section_gap(48)
        
        # 상세 통계 보기
        with st.expander("상세 통계 보기", expanded=False):
            st.markdown("##### 이미지 유형별 평균 성과")
            perf_display = perf_summary.copy()
            perf_display.columns = [
                "이미지 타입",
                "개수",
                "평균 좋아요",
                "중앙값 좋아요",
                "평균 댓글",
                "중앙값 댓글",
                "평균 참여율",
                "중앙값 참여율"
            ]
            if "평균 참여율" in perf_display.columns:
                perf_display["평균 참여율"] = perf_display["평균 참여율"].apply(lambda x: format_engagement_rate(x))
            if "중앙값 참여율" in perf_display.columns:
                perf_display["중앙값 참여율"] = perf_display["중앙값 참여율"].apply(lambda x: format_engagement_rate(x))
            st.dataframe(perf_display, use_container_width=True, hide_index=True)
    
    # ============================================
    # 탭 2: 고성과 분석
    # ============================================
    with tab2:
        prob_10, conc_10, threshold_10 = get_top_percentile_metrics(df_country, 10)
        prob_30, conc_30, threshold_30 = get_top_percentile_metrics(df_country, 30)
        
        st.markdown(
            """
            <div class="section">
                <h4 class="section-title">고성과 달성 가능성</h4>
                <div class="section-desc">각 이미지 유형이 상위 10% 및 30% 성과를 달성할 확률과 상위 성과 내에서의 집중도를 확인하여, 고성과 달성 가능성이 높은 콘텐츠 유형을 파악합니다.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        section_gap(16)
        
        col1, col2 = st.columns(2)
        
        # Top 10% 박스
        with col1:
            if len(prob_10) > 0 and len(conc_10) > 0:
                best_prob_type = prob_10.loc[prob_10["p_top10"].idxmax(), "img_type"]
                best_prob_value = prob_10.loc[prob_10["p_top10"].idxmax(), "p_top10"]
                best_prob_name = get_type_name(best_prob_type)
                
                best_conc_type = conc_10.loc[conc_10["share_in_top10"].idxmax(), "img_type"]
                best_conc_value = conc_10.loc[conc_10["share_in_top10"].idxmax(), "share_in_top10"]
                best_conc_name = get_type_name(best_conc_type)
                
                st.markdown(
                    f"""
                    <div class="kpi-card-wrapper" style="{get_bg_style('white')} {get_border_style('default')} border-radius: {BORDER_RADIUS['md']}; padding: {SPACING['xl']}; box-shadow: 0 1px 2px rgba(0,0,0,0.05);">
                        <div style="background: rgba(31, 87, 149, 0.10); border: 1px solid rgba(31, 87, 149, 0.25); color: {BRAND_COLORS['primary']}; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 700; white-space: nowrap; font-family: 'Arita-Dotum-Bold', sans-serif !important; display: inline-block; margin-bottom: {SPACING['lg']};">
                            Top 10%
                        </div>
                        <div style="margin-bottom: {SPACING['xl']};">
                            <div style="{get_text_style('sm', 'tertiary', family='medium')} font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important; margin-bottom: {SPACING['xs']};">
                                달성 확률 최고
                            </div>
                            <div style="font-size: 24px !important; font-weight: 900 !important; color: {BRAND_COLORS['primary']} !important; font-family: 'Arita-Dotum-Bold', 'Arita-Dotum-Medium', sans-serif !important; margin-bottom: {SPACING['xs']}; line-height: 1.2; letter-spacing: -0.3px; text-shadow: 0.3px 0 0 currentColor;">
                                {best_prob_name}
                            </div>
                            <div style="{get_text_style('lg', 'accent', 'semibold', family='bold')} font-family: 'Arita-Dotum-Bold', 'Arita-Dotum-Medium', sans-serif !important; margin-bottom: {SPACING['sm']};">
                                {best_prob_value*100:.1f}%
                            </div>
                            <div style="{get_text_style('xs', 'muted', family='medium')} font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;">
                                Type {best_prob_type} · 전체 게시물 중 상위 10% 성과 달성 확률
                            </div>
                        </div>
                        <div style="border-top: 1px solid {BORDER_COLORS['light']}; padding-top: {SPACING['lg']};">
                            <div style="{get_text_style('sm', 'tertiary', family='medium')} font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important; margin-bottom: {SPACING['xs']};">
                                집중도 최고
                            </div>
                            <div style="font-size: 24px !important; font-weight: 900 !important; color: {BRAND_COLORS['primary']} !important; font-family: 'Arita-Dotum-Bold', 'Arita-Dotum-Medium', sans-serif !important; margin-bottom: {SPACING['xs']}; line-height: 1.2; letter-spacing: -0.3px; text-shadow: 0.3px 0 0 currentColor;">
                                {best_conc_name}
                            </div>
                            <div style="{get_text_style('lg', 'accent', 'semibold', family='bold')} font-family: 'Arita-Dotum-Bold', 'Arita-Dotum-Medium', sans-serif !important; margin-bottom: {SPACING['sm']};">
                                {best_conc_value*100:.1f}%
                            </div>
                            <div style="{get_text_style('xs', 'muted', family='medium')} font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;">
                                Type {best_conc_type} · 상위 10% 성과 내에서 차지하는 비율
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.info("Top 10% 성과 데이터가 없습니다.")
        
        # Top 30% 박스
        with col2:
            if len(prob_30) > 0 and len(conc_30) > 0:
                best_prob30_type = prob_30.loc[prob_30["p_top30"].idxmax(), "img_type"]
                best_prob30_value = prob_30.loc[prob_30["p_top30"].idxmax(), "p_top30"]
                best_prob30_name = get_type_name(best_prob30_type)
                
                best_conc30_type = conc_30.loc[conc_30["share_in_top30"].idxmax(), "img_type"]
                best_conc30_value = conc_30.loc[conc_30["share_in_top30"].idxmax(), "share_in_top30"]
                best_conc30_name = get_type_name(best_conc30_type)
                
                st.markdown(
                    f"""
                    <div class="kpi-card-wrapper" style="{get_bg_style('white')} {get_border_style('default')} border-radius: {BORDER_RADIUS['md']}; padding: {SPACING['xl']}; box-shadow: 0 1px 2px rgba(0,0,0,0.05);">
                        <div style="background: rgba(31, 87, 149, 0.10); border: 1px solid rgba(31, 87, 149, 0.25); color: {BRAND_COLORS['primary']}; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 700; white-space: nowrap; font-family: 'Arita-Dotum-Bold', sans-serif !important; display: inline-block; margin-bottom: {SPACING['lg']};">
                            Top 30%
                        </div>
                        <div style="margin-bottom: {SPACING['xl']};">
                            <div style="{get_text_style('sm', 'tertiary', family='medium')} font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important; margin-bottom: {SPACING['xs']};">
                                달성 확률 최고
                            </div>
                            <div style="font-size: 24px !important; font-weight: 900 !important; color: {BRAND_COLORS['primary']} !important; font-family: 'Arita-Dotum-Bold', 'Arita-Dotum-Medium', sans-serif !important; margin-bottom: {SPACING['xs']}; line-height: 1.2; letter-spacing: -0.3px; text-shadow: 0.3px 0 0 currentColor;">
                                {best_prob30_name}
                            </div>
                            <div style="{get_text_style('lg', 'accent', 'semibold', family='bold')} font-family: 'Arita-Dotum-Bold', 'Arita-Dotum-Medium', sans-serif !important; margin-bottom: {SPACING['sm']};">
                                {best_prob30_value*100:.1f}%
                            </div>
                            <div style="{get_text_style('xs', 'muted', family='medium')} font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;">
                                Type {best_prob30_type} · 전체 게시물 중 상위 30% 성과 달성 확률
                            </div>
                        </div>
                        <div style="border-top: 1px solid {BORDER_COLORS['light']}; padding-top: {SPACING['lg']};">
                            <div style="{get_text_style('sm', 'tertiary', family='medium')} font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important; margin-bottom: {SPACING['xs']};">
                                집중도 최고
                            </div>
                            <div style="font-size: 24px !important; font-weight: 900 !important; color: {BRAND_COLORS['primary']} !important; font-family: 'Arita-Dotum-Bold', 'Arita-Dotum-Medium', sans-serif !important; margin-bottom: {SPACING['xs']}; line-height: 1.2; letter-spacing: -0.3px; text-shadow: 0.3px 0 0 currentColor;">
                                {best_conc30_name}
                            </div>
                            <div style="{get_text_style('lg', 'accent', 'semibold', family='bold')} font-family: 'Arita-Dotum-Bold', 'Arita-Dotum-Medium', sans-serif !important; margin-bottom: {SPACING['sm']};">
                                {best_conc30_value*100:.1f}%
                            </div>
                            <div style="{get_text_style('xs', 'muted', family='medium')} font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;">
                                Type {best_conc30_type} · 상위 30% 성과 내에서 차지하는 비율
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.info("Top 30% 성과 데이터가 없습니다.")
        
        section_gap(48)
        
        # Top 10% vs Top 30% 비교 차트
        if len(prob_10) > 0 and len(prob_30) > 0:
            st.markdown(
                """
                <div class="section">
                    <h4 class="section-title">Top 10% vs Top 30% 달성 확률 비교</h4>
                    <div class="section-desc">각 이미지 유형이 상위 10%와 30% 성과 구간에 진입할 확률을 비교하여,
고성과 달성 가능성의 차이를 확인합니다.</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            section_gap(16)
            
            comparison_df = pd.DataFrame({
                "img_type": prob_10["img_type"],
                "Top 10%": prob_10["p_top10"],
                "Top 30%": prob_30["p_top30"]
            })
            
            fig = px.bar(
                comparison_df,
                x="img_type",
                y=["Top 10%", "Top 30%"],
                labels={"img_type": "이미지 타입", "value": "", "variable": "기준"},
                title="이미지 타입별 고성과 달성 확률",
                barmode="group",
                color_discrete_map={"Top 10%": MEAN_COLOR, "Top 30%": MEDIAN_COLOR}
            )
            # Top 10%는 #9CA3AF, Top 30%는 #E5E7EB
            if len(fig.data) >= 2:
                fig.data[0].marker.color = MEAN_COLOR  # Top 10% - #9CA3AF
                fig.data[0].name = "Top 10%"
                fig.data[1].marker.color = MEDIAN_COLOR  # Top 30% - #E5E7EB
                fig.data[1].name = "Top 30%"
            # 모든 막대 너비 통일 (더 작게 조정)
            fig.update_traces(width=0.4)
            fig = apply_chart_style(fig)
            fig.update_layout(
                bargap=0.4, 
                height=400,
                showlegend=True,
                yaxis=dict(title=None),
                margin=dict(l=40, r=20, t=40, b=40),
                title=dict(
                    x=0.5,
                    xanchor="center",
                    y=0.94,
                    yanchor="top",
                    font=dict(size=17, color="#111827", family="Arita-Dotum-Medium, Arita-dotum-Medium, sans-serif")
                ),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.15,
                    xanchor="center",
                    x=0.5,
                    font=dict(family="Arita-Dotum-Medium, Arita-dotum-Medium, sans-serif", size=12),
                    itemwidth=30,
                    tracegroupgap=5,
                    itemsizing="constant",
                    bgcolor="rgba(255,255,255,0)",
                    bordercolor="rgba(255,255,255,0)"
                )
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=f"tab2_highperf_{selected_country}")
        
        # 패턴 요약 박스 (고성과 분석 구조적 결론)
        country_insight = insights.get(selected_country, {})
        high_perf_insight = country_insight.get("high_performance_analysis", {})
        if high_perf_insight:
            section_gap(40)
            summary = high_perf_insight.get("summary", "")
            bullets = high_perf_insight.get("bullets", [])
            
            if summary or bullets:
                # 변수 미리 추출
                sm_size = FONT_SIZES["sm"]
                base_size = FONT_SIZES["base"]
                md_size = FONT_SIZES["md"]
                primary_color = BRAND_COLORS["primary"]
                text_primary = TEXT_COLORS["primary"]
                spacing_md = SPACING["md"]
                spacing_lg = SPACING["lg"]
                spacing_xl = SPACING["xl"]
                spacing_xs = SPACING["xs"]
                spacing_sm = SPACING["sm"]
                
                content_html = ""
                
                # 패턴 요약 (summary) - 제목 없이 내용만
                if summary:
                    # summary와 첫 번째 bullet 사이 여백은 유지 (spacing_xl)
                    content_html += f'<div style="margin-bottom: {spacing_xl};"><div style="font-size: {md_size}; font-weight: 400; color: {text_primary}; line-height: 1.6; font-family: \'Arita-Dotum-Medium\', \'Arita-dotum-Medium\', \'Malgun Gothic\', sans-serif !important;">{summary}</div></div>'
                
                # 고성과 분포 특성으로 보기
                if bullets and len(bullets) > 0:
                    # 아이콘 매핑 (제목에 따라 적절한 아이콘 선택)
                    icon_map = {
                        "고성과 분포 특성": "📊",
                        "고성과 이미지 타입 집중도": "🎯",
                        "고성과 분포 특성으로 보기": "📊",
                        "고성과 이미지 타입 집중도로 보기": "🎯"
                    }
                    
                    for i, bullet in enumerate(bullets):
                        bullet_clean = bullet.strip()
                        # <b> 태그 제거
                        bullet_text = bullet_clean.replace("<b>", "").replace("</b>", "")
                        # 🔎 제거
                        bullet_text = bullet_text.replace("🔎", "").strip()
                        
                        # 마지막 항목인지 확인 (항상 마지막 bullet 항목은 여백 없음)
                        is_last = (i == len(bullets) - 1)
                        margin_bottom = "0" if is_last else f"{spacing_xl}"
                        
                        # 제목 추출 (콜론 앞부분)
                        if ":" in bullet_text:
                            title, content = bullet_text.split(":", 1)
                            title = title.strip()
                            content = content.strip()
                            
                            # 아이콘 선택
                            icon = "📋"  # 기본 아이콘
                            for key, value in icon_map.items():
                                if key in title:
                                    icon = value
                                    break
                            
                            content_html += f'<div style="margin-bottom: {margin_bottom};"><div style="font-size: {md_size}; font-weight: 700; color: {primary_color}; margin-bottom: {spacing_sm}; font-family: \'Arita-Dotum-Bold\', \'Arita-Dotum-Medium\', \'Arita-dotum-Medium\', sans-serif !important;">{icon} {title}</div><div style="font-size: {md_size}; font-weight: 400; color: {text_primary}; line-height: 1.6; font-family: \'Arita-Dotum-Medium\', \'Arita-dotum-Medium\', \'Malgun Gothic\', sans-serif !important;">{content}</div></div>'
                        else:
                            # 콜론이 없는 경우 전체를 내용으로 표시
                            content_html += f'<div style="margin-bottom: {margin_bottom};"><div style="font-size: {md_size}; font-weight: 400; color: {text_primary}; line-height: 1.6; font-family: \'Arita-Dotum-Medium\', \'Arita-dotum-Medium\', \'Malgun Gothic\', sans-serif !important;">{bullet_text}</div></div>'
                
                if content_html:
                    try:
                        st.html(
                            f'<div style="background-color: rgba(31, 87, 149, 0.06); border-left: 4px solid {primary_color}; padding: {spacing_lg} {spacing_xl}; margin-bottom: {spacing_sm}; font-family: \'Arita-Dotum-Medium\', \'Arita-dotum-Medium\', \'Malgun Gothic\', sans-serif !important;"><div style="font-size: {base_size}; font-weight: 700; color: {primary_color}; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: {spacing_xs}; font-family: \'Arita-Dotum-Bold\', \'Arita-Dotum-Medium\', \'Arita-dotum-Medium\', sans-serif !important;">📊 패턴 요약</div>{content_html}</div>'
                        )
                    except AttributeError:
                        st.markdown(
                            f'<div style="background-color: rgba(31, 87, 149, 0.06); border-left: 4px solid {primary_color}; padding: {spacing_lg} {spacing_xl}; margin-bottom: {spacing_sm}; font-family: \'Arita-Dotum-Medium\', \'Arita-dotum-Medium\', \'Malgun Gothic\', sans-serif !important;"><div style="font-size: {base_size}; font-weight: 700; color: {primary_color}; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: {spacing_xs}; font-family: \'Arita-Dotum-Bold\', \'Arita-Dotum-Medium\', \'Arita-dotum-Medium\', sans-serif !important;">📊 패턴 요약</div>{content_html}</div>',
                            unsafe_allow_html=True
                        )
        
        # 상세 통계 보기
        with st.expander("상세 통계 보기", expanded=False):
            st.markdown("##### Top 10% 달성 확률")
            if len(prob_10) > 0:
                prob_display = prob_10.copy()
                prob_display.columns = ["이미지 타입", "Top 10% 달성 확률"]
                prob_display["Top 10% 달성 확률"] = prob_display["Top 10% 달성 확률"].apply(lambda x: f"{x*100:.1f}%")
                st.dataframe(prob_display, use_container_width=True, hide_index=True)
            
            if len(conc_10) > 0:
                conc_display = conc_10.copy()
                conc_display.columns = ["이미지 타입", "Top 10% 내 비율"]
                conc_display["Top 10% 내 비율"] = conc_display["Top 10% 내 비율"].apply(lambda x: f"{x*100:.1f}%")
                st.dataframe(conc_display, use_container_width=True, hide_index=True)
            
            st.caption(f"💡 Top 10% 기준선: 참여율 {threshold_10:.6f} 이상")
            
            st.markdown("##### Top 30% 달성 확률")
            if len(prob_30) > 0:
                prob30_display = prob_30.copy()
                prob30_display.columns = ["이미지 타입", "Top 30% 달성 확률"]
                prob30_display["Top 30% 달성 확률"] = prob30_display["Top 30% 달성 확률"].apply(lambda x: f"{x*100:.1f}%")
                st.dataframe(prob30_display, use_container_width=True, hide_index=True)
            
            if len(conc_30) > 0:
                conc30_display = conc_30.copy()
                conc30_display.columns = ["이미지 타입", "Top 30% 내 비율"]
                conc30_display["Top 30% 내 비율"] = conc30_display["Top 30% 내 비율"].apply(lambda x: f"{x*100:.1f}%")
                st.dataframe(conc30_display, use_container_width=True, hide_index=True)
            
            st.caption(f"💡 Top 30% 기준선: 참여율 {threshold_30:.6f} 이상")
    
    # ============================================
    # 탭 3: 안정성 분석
    # ============================================
    with tab3:
        stability = get_stability_metrics(df_country)
        
        st.markdown(
            f"""
            <div class="section" style="margin-bottom: 8px;">
                <h4 class="section-title">성과 안정성 분석</h4>
                <div class="section-desc" style="margin-bottom: 0;">표준편차(STD), IQR(사분위수 범위), 변동계수(CV)를 통해 이미지 타입별 성과의 변동성과 안정성을 측정합니다.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # 안정성 인사이트 표시 (헤더 하이라이트 형태)
        country_insight = insights.get(selected_country, {})
        stability_data = country_insight.get("stability_analysis", {})
        
        if stability_data and stability_data.get("type"):
            stability_type = stability_data.get("type")
            keyword = stability_data.get("keyword", "")
            interpretation = stability_data.get("interpretation", {})
            
            # 키워드가 있으면 텍스트에서 키워드 부분을 강조
            if keyword and keyword in stability_type:
                # 키워드 부분을 볼드 + 색상 변경으로 강조 ('유형' 부분)
                highlighted_type = stability_type.replace(
                    keyword, 
                    f'<span class="stability-keyword" style="font-weight: 700 !important; color: {BRAND_COLORS["primary"]} !important; font-family: \'Arita-Dotum-Bold\', \'Arita-Dotum-Medium\', \'Arita-dotum-Medium\', sans-serif !important;">{keyword}</span>'
                )
            else:
                highlighted_type = stability_type
            
            # 요약을 Insight Callout 형태로 표시
            st.markdown(
                f"""
                <style>
                .stability-keyword {{
                    font-weight: 700 !important;
                    color: {BRAND_COLORS['primary']} !important;
                    font-family: 'Arita-Dotum-Bold', 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;
                }}
                </style>
                <div style="margin-top: {SPACING['lg']}; margin-bottom: {SPACING['md']};">
                    <div style="background-color: rgba(31, 87, 149, 0.06); border-radius: {BORDER_RADIUS['sm']}; padding: {SPACING['lg']} {SPACING['xl']}; border: none; box-shadow: none; font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;">
                        <div style="display: flex; align-items: flex-start; gap: {SPACING['sm']};">
                            <div style="color: {BRAND_COLORS['primary']}; font-size: 16px; line-height: 1.4; flex-shrink: 0; margin-top: 2px;">📌</div>
                            <div style="flex: 1;">
                                <div style="font-size: 11px; font-weight: 700; color: {BRAND_COLORS['primary']}; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: {SPACING['xs']}; font-family: 'Arita-Dotum-Bold', 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;">
                                    INSIGHT
                                </div>
                                <div style="font-size: {FONT_SIZES['xl']}; font-weight: 500; color: {TEXT_COLORS['primary']}; line-height: 1.5; word-break: keep-all; font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;">
                                    {highlighted_type}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # 해석 토글 (디스클로저 라인 형태 - 카드 배경 없이)
            if interpretation and interpretation.get("bullets"):
                interpretation_title = interpretation.get("title", "성과 안정성 구조 및 지표 해석")
                interpretation_bullets = interpretation.get("bullets", [])
                
                # 디스클로저 라인 형태의 토글 (HTML details/summary 직접 사용)
                bullets_html = ""
                for bullet in interpretation_bullets:
                    bullets_html += f'<div style="margin-bottom: {SPACING["sm"]}; {get_text_style("md", "secondary", "normal", "medium")} line-height: 1.6;">{bullet}</div>'
                
                st.markdown(
                    f"""
                    <div class="stability-interpretation-wrapper" style="border-top: 1px solid #E5E7EB; border-bottom: 1px solid #E5E7EB; padding: 12px 0; margin: 16px 0;">
                        <details class="stability-details" style="cursor: pointer;">
                            <summary class="stability-interpretation-summary">
                                {interpretation_title}
                            </summary>
                            <div class="stability-interpretation-content" style="margin-top: 16px; padding-left: 0;">
                                {bullets_html}
                            </div>
                        </details>
                    </div>
                    <style>
                    .stability-interpretation-wrapper {{
                        font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;
                    }}
                    .stability-details {{
                        background: transparent !important;
                        border: none !important;
                        padding: 0 !important;
                        margin: 0 !important;
                    }}
                    .stability-interpretation-summary {{
                        font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;
                        font-size: 14px !important;
                        font-weight: 500 !important;
                        color: #6B7280 !important;
                        list-style: none !important;
                        padding: 0 !important;
                        padding-left: 20px !important;
                        margin: 0 !important;
                        cursor: pointer !important;
                        user-select: none !important;
                        position: relative !important;
                        display: block !important;
                    }}
                    .stability-interpretation-summary::-webkit-details-marker {{
                        display: none !important;
                    }}
                    .stability-interpretation-summary::marker {{
                        display: none !important;
                        content: '' !important;
                    }}
                    .stability-interpretation-summary::before {{
                        content: '+' !important;
                        position: absolute !important;
                        left: 0 !important;
                        top: 0 !important;
                        color: {BRAND_COLORS['primary']} !important;
                        font-weight: 600 !important;
                        font-size: 16px !important;
                        width: 16px !important;
                        text-align: center !important;
                        line-height: 1.4 !important;
                        font-family: 'Arita-Dotum-Bold', 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;
                    }}
                    .stability-details[open] .stability-interpretation-summary::before {{
                        content: '−' !important;
                    }}
                    .stability-interpretation-summary:hover {{
                        color: {BRAND_COLORS['primary']} !important;
                    }}
                    .stability-interpretation-content {{
                        font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;
                        color: #374151 !important;
                    }}
                    .stability-interpretation-content div {{
                        font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;
                        color: #374151 !important;
                    }}
                    </style>
                    """,
                    unsafe_allow_html=True
                )
            
            section_gap(24)

        #그래프 표시    
        if len(stability) > 0:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(
                    f"""
                    <div style="margin-bottom: 8px; font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;">
                        <div style="{get_text_style('md', 'secondary', 'semibold')} margin-bottom: 2px; font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;">표준편차 (STD)</div>
                        <div style="{get_text_style('sm', 'tertiary')} font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;">성과 변동성 측정</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                # 최고값 막대는 연한 하늘색으로 강조
                max_idx = stability["eng_std"].idxmax()
                colors = []
                for idx, row in stability.iterrows():
                    if idx == max_idx:
                        colors.append(LIGHT_BLUE_HIGHLIGHT)  # 최고값은 연한 하늘색
                    else:
                        colors.append(DEFAULT_BAR_COLOR)  # 나머지는 #E1E4EA
                
                fig1 = px.bar(
                    stability,
                    x="img_type",
                    y="eng_std",
                    labels={"img_type": "이미지 타입", "eng_std": ""},
                    title=None
                )
                fig1.update_traces(marker_color=colors, width=0.6)
                fig1 = apply_chart_style(fig1)
                fig1.update_layout(
                    bargap=0.4, 
                    showlegend=False, 
                    height=300,
                    yaxis=dict(title=None),
                    margin=dict(l=40, r=10, t=20, b=40),
                    title=dict(text=""),
                    xaxis=dict(title=None),
                    autosize=True
                )
                st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False}, key=f"tab3_std_{selected_country}")
            
            with col2:
                st.markdown(
                    f"""
                    <div style="margin-bottom: 8px; font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;">
                        <div style="{get_text_style('md', 'secondary', 'semibold')} margin-bottom: 2px; font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;">IQR (사분위수 범위)</div>
                        <div style="{get_text_style('sm', 'tertiary')} font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;">중간 50% 퍼짐 정도</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                # 최고값 막대는 연한 하늘색으로 강조
                max_idx = stability["eng_iqr"].idxmax()
                colors = []
                for idx, row in stability.iterrows():
                    if idx == max_idx:
                        colors.append(LIGHT_BLUE_HIGHLIGHT)  # 최고값은 연한 하늘색
                    else:
                        colors.append(DEFAULT_BAR_COLOR)  # 나머지는 #E1E4EA
                
                fig2 = px.bar(
                    stability,
                    x="img_type",
                    y="eng_iqr",
                    labels={"img_type": "이미지 타입", "eng_iqr": ""},
                    title=None
                )
                fig2.update_traces(marker_color=colors, width=0.6)
                fig2 = apply_chart_style(fig2)
                fig2.update_layout(
                    bargap=0.4, 
                    showlegend=False, 
                    height=300,
                    yaxis=dict(title=None),
                    margin=dict(l=40, r=10, t=20, b=40),
                    title=dict(text=""),
                    xaxis=dict(title=None),
                    autosize=True
                )
                st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False}, key=f"tab3_iqr_{selected_country}")
            
            with col3:
                st.markdown(
                    f"""
                    <div style="margin-bottom: 8px; font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;">
                        <div style="{get_text_style('md', 'secondary', 'semibold')} margin-bottom: 2px; font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;">변동계수 (CV)</div>
                        <div style="{get_text_style('sm', 'tertiary')} font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;">상대적 변동성</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                # 최고값 막대는 연한 하늘색으로 강조
                max_idx = stability["eng_cv"].idxmax()
                colors = []
                for idx, row in stability.iterrows():
                    if idx == max_idx:
                        colors.append(LIGHT_BLUE_HIGHLIGHT)  # 최고값은 연한 하늘색
                    else:
                        colors.append(DEFAULT_BAR_COLOR)  # 나머지는 #E1E4EA
                
                fig3 = px.bar(
                    stability,
                    x="img_type",
                    y="eng_cv",
                    labels={"img_type": "이미지 타입", "eng_cv": ""},
                    title=None
                )
                fig3.update_traces(marker_color=colors, width=0.6)
                fig3 = apply_chart_style(fig3)
                fig3.update_layout(
                    bargap=0.4, 
                    showlegend=False, 
                    height=300,
                    yaxis=dict(title=None),
                    margin=dict(l=40, r=10, t=20, b=40),
                    title=dict(text=""),
                    xaxis=dict(title=None),
                    autosize=True
                )
                st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False}, key=f"tab3_cv_{selected_country}")

        # 상세 통계 보기
        with st.expander("상세 통계 보기", expanded=False):
            if len(stability) > 0:
                stability_display = stability.copy()
                stability_display.columns = [
                    "이미지 타입",
                    "개수",
                    "평균 참여율",
                    "표준편차 (STD)",
                    "IQR",
                    "변동계수 (CV)"
                ]
                st.dataframe(stability_display, use_container_width=True, hide_index=True)
    
    # ============================================
    # 탭 4: 전략 인사이트
    # ============================================
    with tab4:
        usage_vs_perf, underused, overused = get_usage_vs_performance(df_country, 10)
        country_insight = insights.get(selected_country, {})
        strategy_insights = country_insight.get("strategy_insights", {})
        from utils.charts import get_country_name
        perf_summary = get_performance_summary(df_country)
        
        # ============================================
        # 1️⃣ 종합 인사이트 (최상단)
        # ============================================
        st.markdown(
            """
            <div class="section">
                <h4 class="section-title">종합 인사이트</h4>
                <div class="section-desc">선택된 국가의 콘텐츠 운영 현황과 성과 핵심을 종합적으로 요약합니다.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        section_gap(4)
        
        country_summary = strategy_insights.get("summary", {})
        usage_vs_perf_data = strategy_insights.get("usage_vs_performance", {})
        
        if country_summary:
            insight = country_summary.get("insight", "")
            current_status = country_summary.get("current_status", "")
            performance_core = country_summary.get("performance_core", "")
            
            # 짧은 판단 문구 추출 
            conclusion = insight.replace("인사이트:", "").strip() if "인사이트:" in insight else insight.strip()
            
            # 현재 운영에서 데이터 추출
            current_ops = current_status.replace("현황:", "").strip() if "현황:" in current_status else current_status.strip()
            # 47% 추출
            usage_percent_match = re.search(r'(\d+(?:\.\d+)?%)', current_ops)
            usage_percent = usage_percent_match.group(1) if usage_percent_match else ""
            # "제품 중심(유형 1·2)" 또는 유사한 텍스트 추출
            type_info_match = re.search(r'(제품 중심|유형\s*\d+(?:·\d+)?)', current_ops)
            type_info = type_info_match.group(1) if type_info_match else current_ops.split()[0] if current_ops else ""
            # 유형 번호 추출
            type_numbers_match = re.search(r'유형\s*(\d+(?:·\d+)?)', current_ops)
            if type_numbers_match:
                type_info = f"제품 중심(유형 {type_numbers_match.group(1)})"
            # 전체 설명 텍스트 (간략화)
            if len(current_ops) > 30:
                current_ops_summary = current_ops[:30] + "..."
            else:
                current_ops_summary = current_ops
            
            # 성과 핵심에서 데이터 추출
            perf_key = performance_core.replace("성과 핵심:", "").strip() if "성과 핵심:" in performance_core else performance_core.strip()
            # "1.5~2배" 또는 "1.8×" 추출
            multiplier_match = re.search(r'(\d+(?:\.\d+)?~?\d*(?:\.\d+)?[배×])', perf_key)
            multiplier = multiplier_match.group(1) if multiplier_match else ""
            # "유형 4" 추출
            type_match = re.search(r'유형\s*(\d+)', perf_key)
            type_num = f"유형 {type_match.group(1)}" if type_match else "유형 4"
            # 전체 설명 텍스트 (간략화)
            if len(perf_key) > 30:
                perf_key_summary = perf_key[:30] + "..."
            else:
                perf_key_summary = perf_key
            
            # Insight Scorecard HTML 생성
            # 1. 짧은 판단 문구 (먼저 렌더링) - 이모지 추가
            judgment_html = f"""
            <div style="margin-top: {SPACING['md']}; margin-bottom: {SPACING['sm']};">
                <div style="font-size: {FONT_SIZES['lg']}; font-weight: 600; color: {TEXT_COLORS['primary']}; font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', 'Malgun Gothic', sans-serif !important;">
                    📍 {conclusion}
                </div>
            </div>
            """
            try:
                st.html(judgment_html)
            except AttributeError:
                st.markdown(judgment_html, unsafe_allow_html=True)
            
            # 2. KPI 카드 2개 (가로 배치)
            col1, col2 = st.columns(2)
            
            with col1:
                # 카드 1: 현황
                card1_html = f"""
                <div style="background-color: {BG_COLORS['white']}; border: 1px solid {BORDER_COLORS['default']}; border-radius: {BORDER_RADIUS['sm']}; padding: {SPACING['lg']} {SPACING['xl']}; box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05); font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', 'Malgun Gothic', sans-serif !important; overflow: visible; min-height: auto;">
                    <div style="font-size: {FONT_SIZES['sm']}; font-weight: 600; color: {TEXT_COLORS['secondary']}; margin-bottom: {SPACING['sm']}; font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', 'Malgun Gothic', sans-serif !important;">
                        📊 현황
                    </div>
                    <div style="font-size: {FONT_SIZES['2xl']}; font-weight: 700; color: {BRAND_COLORS['primary']}; margin-bottom: {SPACING['xs']}; font-family: 'Arita-Dotum-Bold', 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;">
                        {usage_percent if usage_percent else "—"}
                    </div>
                    <div style="font-size: {FONT_SIZES['base']}; font-weight: 400; color: {TEXT_COLORS['primary']}; line-height: 1.5; font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', 'Malgun Gothic', sans-serif !important;">
                        {current_ops_summary if current_ops_summary else type_info if type_info else ""}
                    </div>
                </div>
                """
                try:
                    st.html(card1_html)
                except AttributeError:
                    st.markdown(card1_html, unsafe_allow_html=True)
            
            with col2:
                # 카드 2: 성과 핵심
                card2_html = f"""
                <div style="background-color: {BG_COLORS['white']}; border: 1px solid {BORDER_COLORS['default']}; border-radius: {BORDER_RADIUS['sm']}; padding: {SPACING['lg']} {SPACING['xl']}; box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05); font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', 'Malgun Gothic', sans-serif !important; overflow: visible; min-height: auto;">
                    <div style="font-size: {FONT_SIZES['sm']}; font-weight: 600; color: {TEXT_COLORS['secondary']}; margin-bottom: {SPACING['sm']}; font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', 'Malgun Gothic', sans-serif !important;">
                        🎯 성과 핵심
                    </div>
                    <div style="font-size: {FONT_SIZES['2xl']}; font-weight: 700; color: {BRAND_COLORS['primary']}; margin-bottom: {SPACING['xs']}; font-family: 'Arita-Dotum-Bold', 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;">
                        {multiplier if multiplier else "—"}
                    </div>
                    <div style="font-size: {FONT_SIZES['base']}; font-weight: 400; color: {TEXT_COLORS['primary']}; line-height: 1.5; font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', 'Malgun Gothic', sans-serif !important;">
                        {perf_key_summary if perf_key_summary else f"{type_num} 반응률"}
                    </div>
                </div>
                """
                try:
                    st.html(card2_html)
                except AttributeError:
                    st.markdown(card2_html, unsafe_allow_html=True)
        
        section_gap(32)
        
        # ============================================
        # 활용도 vs 성과 분석 (그래프)
        # ============================================
        st.markdown(
            """
            <div class="section">
                <h4 class="section-title">활용도 vs 성과 분석</h4>
                <div class="section-desc">각 이미지 유형의 활용 비중과 실제 성과를 비교하여 운영 효율성을 분석합니다.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        section_gap(16)
        
        # 차트 위에 핵심 판단 인사이트 박스 배치 (3가지 모두 표시)
        if usage_vs_perf_data:
            # 텍스트 추출
            status_text = ""
            perf_text = ""
            comp_text = ""
            
            # 📈 활용 현황
            if usage_vs_perf_data.get("current_status"):
                text = usage_vs_perf_data["current_status"]
                if "📈" in text:
                    status_text = text.split(":", 1)[1].strip() if ":" in text else text.replace("📈", "").replace("활용 현황:", "").strip()
                    if status_text and not status_text.endswith('.'):
                        status_text += '.'
            
            # 🏆 실제 성과
            if usage_vs_perf_data.get("actual_performance"):
                text = usage_vs_perf_data["actual_performance"]
                if "🏆" in text:
                    perf_text = text.split(":", 1)[1].strip() if ":" in text else text.replace("🏆", "").replace("실제 성과:", "").strip()
                    if perf_text and not perf_text.endswith('.'):
                        perf_text += '.'
            
            # 📍 비교 분석
            if usage_vs_perf_data.get("comparison_analysis"):
                text = usage_vs_perf_data["comparison_analysis"]
                if "📍" in text:
                    comp_text = text.split(":", 1)[1].strip() if ":" in text else text.replace("📍", "").replace("비교 분석:", "").strip()
                    if comp_text and not comp_text.endswith('.'):
                        comp_text += '.'
            
            # 하나의 박스에 모두 표시
            if status_text or perf_text or comp_text:
                # HTML 문자열 직접 생성 (변수 미리 추출) - 패턴 요약과 동일한 스타일 적용
                sm_size = FONT_SIZES["sm"]
                base_size = FONT_SIZES["base"]
                md_size = FONT_SIZES["md"]
                primary_color = BRAND_COLORS["primary"]
                text_primary = TEXT_COLORS["primary"]
                text_secondary = TEXT_COLORS["secondary"]
                spacing_md = SPACING["md"]
                spacing_lg = SPACING["lg"]
                spacing_xl = SPACING["xl"]
                spacing_xs = SPACING["xs"]
                spacing_sm = SPACING["sm"]
                
                content_html = ""
                
                if status_text:
                    content_html += f'<div style="margin-bottom: {spacing_xl};"><div style="font-size: {md_size}; font-weight: 700; color: {primary_color}; margin-bottom: {spacing_sm}; font-family: \'Arita-Dotum-Bold\', \'Arita-Dotum-Medium\', \'Arita-dotum-Medium\', sans-serif !important;">📈 활용 현황</div><div style="font-size: {md_size}; font-weight: 400; color: {text_primary}; line-height: 1.6; font-family: \'Arita-Dotum-Medium\', \'Arita-dotum-Medium\', \'Malgun Gothic\', sans-serif !important;">{status_text}</div></div>'
                
                if perf_text:
                    content_html += f'<div style="margin-bottom: {spacing_xl};"><div style="font-size: {md_size}; font-weight: 700; color: {primary_color}; margin-bottom: {spacing_sm}; font-family: \'Arita-Dotum-Bold\', \'Arita-Dotum-Medium\', \'Arita-dotum-Medium\', sans-serif !important;">🏆 실제 성과</div><div style="font-size: {md_size}; font-weight: 400; color: {text_primary}; line-height: 1.6; font-family: \'Arita-Dotum-Medium\', \'Arita-dotum-Medium\', \'Malgun Gothic\', sans-serif !important;">{perf_text}</div></div>'
                
                if comp_text:
                    content_html += f'<div><div style="font-size: {md_size}; font-weight: 700; color: {primary_color}; margin-bottom: {spacing_sm}; font-family: \'Arita-Dotum-Bold\', \'Arita-Dotum-Medium\', \'Arita-dotum-Medium\', sans-serif !important;">📍 비교 분석</div><div style="font-size: {md_size}; font-weight: 400; color: {text_primary}; line-height: 1.6; font-family: \'Arita-Dotum-Medium\', \'Arita-dotum-Medium\', \'Malgun Gothic\', sans-serif !important;">{comp_text}</div></div>'
                
                # st.html 사용 (패턴 요약과 동일한 방식)
                if content_html:
                    try:
                        st.html(
                            f'<div style="background-color: rgba(31, 87, 149, 0.06); border-left: 4px solid {primary_color}; padding: {spacing_lg} {spacing_xl}; margin-bottom: {spacing_sm}; font-family: \'Arita-Dotum-Medium\', \'Arita-dotum-Medium\', \'Malgun Gothic\', sans-serif !important;">{content_html}</div>'
                        )
                    except AttributeError:
                        # st.html이 없는 경우 st.markdown 사용
                        st.markdown(
                            f'<div style="background-color: rgba(31, 87, 149, 0.06); border-left: 4px solid {primary_color}; padding: {spacing_lg} {spacing_xl}; margin-bottom: {spacing_sm}; font-family: \'Arita-Dotum-Medium\', \'Arita-dotum-Medium\', \'Malgun Gothic\', sans-serif !important;">{content_html}</div>',
                            unsafe_allow_html=True
                        )
        
        # 차트는 인사이트를 뒷받침하는 근거 역할
        section_gap(16)
        plot_usage_vs_engagement(
            type_ratio,
            perf_summary,
            selected_country,
            key_suffix="tab4"
        )
        
        section_gap(40)
        
        # ============================================
        # 2️⃣ 콘텐츠 유형별 전략 제안 (국가 기준)
        # ============================================
        st.markdown(
            f"""
            <style>
            .strategy-section div,
            .strategy-section *,
            .strategy-content,
            .strategy-content * {{
                font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', 'Malgun Gothic', sans-serif !important;
            }}
            </style>
            <div class="section strategy-section">
                <h4 class="section-title">콘텐츠 유형별 전략 제안</h4>
                <div class="section-desc" style="margin-bottom: {SPACING['xl']};">선택된 국가에서, 활용 비중 대비 성과가 과대/과소 평가된 콘텐츠 유형을 기준으로 운영 전략을 제안합니다.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        section_gap(24)
        
        underused_insights = strategy_insights.get("underused_types", [])
        overused_insights = strategy_insights.get("overused_types", [])
        
        # 1️⃣ 과소 활용 타입
        if len(underused) > 0:
            st.markdown(
                f"""
                <div class="strategy-content" style="margin-bottom: {SPACING['xl']};">
                    <div style="font-size: {FONT_SIZES['lg']}; font-weight: 900; color: {TEXT_COLORS['primary']}; margin-bottom: {SPACING['md']}; text-shadow: 0.3px 0 0 currentColor; letter-spacing: -0.2px; font-family: 'Arita-Dotum-Bold', 'Arita-Dotum-Medium', 'Arita-dotum-Medium', 'Malgun Gothic', sans-serif !important;">
                        과소 활용 타입
                    </div>
                    <div style="font-size: {FONT_SIZES['base']}; color: {TEXT_COLORS['secondary']}; line-height: 1.6; margin-bottom: {SPACING['lg']}; font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', 'Malgun Gothic', sans-serif !important;">
                        활용 비중은 낮지만, 참여율·상위 10% 진입 확률이 높아 추가 투입 시 성과 확장이 기대되는 유형입니다.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # 타입별 리스트
            for idx, row in underused.iterrows():
                type_num = int(row["img_type"])
                type_name = get_type_name(type_num)
                usage_pct = row["usage_share"] * 100
                eng_rate = row["eng_mean"]
                top10_pct = row.get("p_top10", 0) * 100
                
                # 해당 타입의 인사이트 찾기
                type_insight = ""
                for insight_text in underused_insights:
                    if f"유형 {type_num}" in insight_text or f"Type {type_num}" in insight_text:
                        if ":" in insight_text:
                            type_insight = insight_text.split(":", 1)[1].strip()
                        else:
                            type_insight = insight_text.strip()
                        break
                
                # 성과 근거 텍스트 생성
                perf_reasons = []
                if eng_rate > 0:
                    perf_reasons.append(f"평균 참여율 {format_engagement_rate(eng_rate)}")
                if top10_pct > 0:
                    perf_reasons.append(f"Top10% 진입 확률 {top10_pct:.1f}%")
                perf_reason_text = " / ".join(perf_reasons) if perf_reasons else "성과 데이터 없음"
                
                st.markdown(
                    f"""
                    <div style="border-left: 3px solid {BRAND_COLORS['primary']}; background-color: rgba(31, 87, 149, 0.03); padding: {SPACING['lg']} {SPACING['xl']}; margin-bottom: {SPACING['md']}; border-radius: {BORDER_RADIUS['sm']}; font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', 'Arita-Dotum-Medium', 'Malgun Gothic', sans-serif !important;">
                        <div style="font-size: {FONT_SIZES['base']}; font-weight: 600; color: {TEXT_COLORS['primary']}; margin-bottom: {SPACING['sm']}; font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', 'Arita-Dotum-Medium', 'Malgun Gothic', sans-serif !important;">
                            유형 {type_num} · {type_name}
                        </div>
                        <div style="font-size: {FONT_SIZES['sm']}; color: {TEXT_COLORS['secondary']}; margin-bottom: {SPACING['xs']}; line-height: 1.6; font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', 'Arita-Dotum-Medium', 'Malgun Gothic', sans-serif !important;">
                            - 활용도: {format_percentage(usage_pct)}
                        </div>
                        <div style="font-size: {FONT_SIZES['sm']}; color: {TEXT_COLORS['secondary']}; margin-bottom: {SPACING['xs']}; line-height: 1.6; font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', 'Arita-Dotum-Medium', 'Malgun Gothic', sans-serif !important;">
                            - 성과 근거: {perf_reason_text}
                        </div>
                        <div style="font-size: {FONT_SIZES['sm']}; color: {TEXT_COLORS['secondary']}; line-height: 1.6; font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', 'Arita-Dotum-Medium', 'Malgun Gothic', sans-serif !important;">
                            - 해석: {type_insight if type_insight else "활용 비중 대비 성과 효율이 높아 확장 우선 대상"}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        
        section_gap(32)
        
        # 2️⃣ 과대 활용 타입
        if len(overused) > 0:
            st.markdown(
                f"""
                <div class="strategy-content" style="margin-bottom: {SPACING['xl']}; margin-top: {SPACING['xl']};">
                    <div style="font-size: {FONT_SIZES['lg']}; font-weight: 900; color: {TEXT_COLORS['primary']}; margin-bottom: {SPACING['md']}; text-shadow: 0.3px 0 0 currentColor; letter-spacing: -0.2px; font-family: 'Arita-Dotum-Bold', 'Arita-Dotum-Medium', 'Arita-dotum-Medium', 'Malgun Gothic', sans-serif !important;">
                        과대 활용 타입
                    </div>
                    <div style="font-size: {FONT_SIZES['base']}; color: {TEXT_COLORS['secondary']}; line-height: 1.6; margin-bottom: {SPACING['lg']}; font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', 'Malgun Gothic', sans-serif !important;">
                        운영 비중은 높으나, 성과 지표가 이를 따라가지 못해 투입 대비 효율이 낮은 유형입니다.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # 타입별 리스트
            for idx, row in overused.iterrows():
                type_num = int(row["img_type"])
                type_name = get_type_name(type_num)
                usage_pct = row["usage_share"] * 100
                eng_rate = row["eng_mean"]
                top10_pct = row.get("p_top10", 0) * 100
                
                # 해당 타입의 인사이트 찾기
                type_insight = ""
                for insight_text in overused_insights:
                    if f"유형 {type_num}" in insight_text or f"Type {type_num}" in insight_text or (f"유형 {type_num}," in insight_text):
                        if ":" in insight_text:
                            type_insight = insight_text.split(":", 1)[1].strip()
                        else:
                            type_insight = insight_text.strip()
                        break
                
                # 성과 근거 텍스트 생성
                perf_reasons = []
                if top10_pct > 0:
                    perf_reasons.append(f"Top10% 진입 확률 {top10_pct:.1f}%")
                elif eng_rate > 0:
                    perf_reasons.append(f"평균 참여율 {format_engagement_rate(eng_rate)}")
                perf_reason_text = " / ".join(perf_reasons) if perf_reasons else "성과 데이터 없음"
                
                st.markdown(
                    f"""
                    <div style="border-left: 3px solid {BRAND_COLORS['primary']}; background-color: rgba(31, 87, 149, 0.03); padding: {SPACING['lg']} {SPACING['xl']}; margin-bottom: {SPACING['md']}; border-radius: {BORDER_RADIUS['sm']}; font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', 'Arita-Dotum-Medium', 'Malgun Gothic', sans-serif !important;">
                        <div style="font-size: {FONT_SIZES['base']}; font-weight: 600; color: {TEXT_COLORS['primary']}; margin-bottom: {SPACING['sm']}; font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', 'Arita-Dotum-Medium', 'Malgun Gothic', sans-serif !important;">
                            유형 {type_num} · {type_name}
                        </div>
                        <div style="font-size: {FONT_SIZES['sm']}; color: {TEXT_COLORS['secondary']}; margin-bottom: {SPACING['xs']}; line-height: 1.6; font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', 'Arita-Dotum-Medium', 'Malgun Gothic', sans-serif !important;">
                            - 활용도: {format_percentage(usage_pct)}
                        </div>
                        <div style="font-size: {FONT_SIZES['sm']}; color: {TEXT_COLORS['secondary']}; margin-bottom: {SPACING['xs']}; line-height: 1.6; font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', 'Arita-Dotum-Medium', 'Malgun Gothic', sans-serif !important;">
                            - 성과 근거: {perf_reason_text}
                        </div>
                        <div style="font-size: {FONT_SIZES['sm']}; color: {TEXT_COLORS['secondary']}; line-height: 1.6; font-family: 'Arita-Dotum-Medium', 'Arita-dotum-Medium', 'Arita-Dotum-Medium', 'Malgun Gothic', sans-serif !important;">
                            - 해석: {type_insight if type_insight else "물량 대비 성과 효율이 낮아 점진적 축소 필요"}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        
        section_gap(40)
        
        # ============================================
        # 3️⃣ 국가별 상세 근거 보기 (토글)
        # ============================================
        if underused_insights or overused_insights:
            with st.expander("국가별 상세 근거 보기", expanded=False):
                country_name = get_country_name(selected_country)
                
                if underused_insights:
                    st.markdown(
                        f"""
                        <div class="strategy-content" style="margin-bottom: {SPACING['md']};">
                            <div style="font-size: {FONT_SIZES['sm']}; font-weight: 700; color: {BRAND_COLORS['primary']}; margin-bottom: {SPACING['sm']}; font-family: 'Arita-Dotum-Bold', 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;">
                                {country_name} - 과소 활용 타입 판정 근거
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    for insight_text in underused_insights:
                        clean_text = insight_text.strip()
                        st.markdown(
                            f'<div class="strategy-content" style="font-size: {FONT_SIZES["sm"]}; color: {TEXT_COLORS["secondary"]}; line-height: 1.6; margin-bottom: {SPACING["sm"]}; padding-left: {SPACING["md"]}; font-family: \'Arita-Dotum-Medium\', \'Arita-dotum-Medium\', \'Malgun Gothic\', sans-serif !important;">• {clean_text}</div>',
                            unsafe_allow_html=True
                        )
                
                if overused_insights:
                    st.markdown(
                        f"""
                        <div class="strategy-content" style="margin-top: {SPACING['lg']}; margin-bottom: {SPACING['md']};">
                            <div style="font-size: {FONT_SIZES['sm']}; font-weight: 700; color: #6B7280; margin-bottom: {SPACING['sm']}; font-family: 'Arita-Dotum-Bold', 'Arita-Dotum-Medium', 'Arita-dotum-Medium', sans-serif !important;">
                                {country_name} - 과대 활용 타입 판정 근거
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    for insight_text in overused_insights:
                        clean_text = insight_text.strip()
                        st.markdown(
                            f'<div class="strategy-content" style="font-size: {FONT_SIZES["sm"]}; color: {TEXT_COLORS["secondary"]}; line-height: 1.6; margin-bottom: {SPACING["sm"]}; padding-left: {SPACING["md"]}; font-family: \'Arita-Dotum-Medium\', \'Arita-dotum-Medium\', \'Malgun Gothic\', sans-serif !important;">• {clean_text}</div>',
                            unsafe_allow_html=True
                        )
        
        section_gap(40)
    
    section_gap(48)

