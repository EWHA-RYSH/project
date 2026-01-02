# ======================================================
# Insights Store — JSON 인사이트 로더
# ======================================================

import json
import os
import streamlit as st

@st.cache_data(show_spinner=False)
def load_tab_insights(tab_key: str) -> dict:
    """
    탭별 인사이트 JSON 파일을 로드합니다.
    
    Args:
        tab_key: 탭 키 (예: "tab1")
        
    Returns:
        dict: 인사이트 데이터 (파일 없음/파싱 실패 시 빈 dict 반환)
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base_dir, "data", "insights", f"{tab_key}.json")
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}
    except Exception:
        return {}

