import streamlit as st

# 페이지 기본 설정 (반드시 첫 번째 Streamlit 명령어여야 함)
st.set_page_config(
    page_title="스마트 스토어 데이터 분석",
    page_icon="📊",
    layout="wide"
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
from wordcloud import WordCloud
import matplotlib.font_manager as fm
from konlpy.tag import Okt
import platform
import os
from utils import (
    generate_wordcloud_data, 
    create_wordcloud, 
    simple_sentiment_analysis, 
    analyze_options,
    get_font_path,
    get_stopwords,
    add_stopword,
    reset_stopwords,
    remove_stopword,
    DEFAULT_STOPWORDS,
    analyze_positive_review_categories,
    analyze_neutral_review_categories,
    analyze_negative_review_categories,
    check_sales_columns,
    get_sales_periods,
    analyze_top_products_by_period,
    analyze_sales_efficiency,
    analyze_price_segments,
    analyze_review_sales_correlation,
    calculate_sales_growth_pattern,
    get_sales_summary_stats
)

# 한글 폰트 설정
korean_font_path = get_font_path()
korean_font_prop = None

if korean_font_path:
    try:
        korean_font_prop = fm.FontProperties(fname=korean_font_path)
        plt.rcParams['font.family'] = korean_font_prop.get_name()
        print(f"한글 폰트 설정 완료: {korean_font_path}")
    except Exception as e:
        print(f"폰트 설정 오류: {e}")
        # 폰트 파일 경로를 직접 사용
        plt.rcParams['font.family'] = korean_font_path
else:
    # 폰트 경로를 찾을 수 없는 경우 시스템 내장 폰트 사용 시도
    try:
        # Windows
        if platform.system() == 'Windows':
            plt.rcParams['font.family'] = 'Malgun Gothic'
            korean_font_prop = fm.FontProperties(family='Malgun Gothic')
        # macOS
        elif platform.system() == 'Darwin':
            plt.rcParams['font.family'] = 'AppleGothic'
            korean_font_prop = fm.FontProperties(family='AppleGothic')
        # Linux (Streamlit Cloud)
        else:
            # 나눔고딕 폰트 시도
            available_fonts = [f.name for f in fm.fontManager.ttflist]
            nanum_fonts = [f for f in available_fonts if 'Nanum' in f]
            
            if nanum_fonts:
                plt.rcParams['font.family'] = nanum_fonts[0]
                korean_font_prop = fm.FontProperties(family=nanum_fonts[0])
                print(f"나눔 폰트 설정: {nanum_fonts[0]}")
            else:
                plt.rcParams['font.family'] = 'DejaVu Sans'
                korean_font_prop = fm.FontProperties(family='DejaVu Sans')
                print("기본 폰트 사용: DejaVu Sans")
    except Exception as e:
        print(f"폰트 설정 실패: {e}")
        st.warning("한글 폰트를 설정할 수 없습니다. 시각화에서 한글이 제대로 표시되지 않을 수 있습니다.")

plt.rcParams['axes.unicode_minus'] = False

# 전역 폰트 속성 설정 함수
def set_korean_font(ax):
    """matplotlib axes에 한글 폰트를 설정합니다."""
    if korean_font_prop:
        ax.set_xlabel(ax.get_xlabel(), fontproperties=korean_font_prop)
        ax.set_ylabel(ax.get_ylabel(), fontproperties=korean_font_prop)
        ax.set_title(ax.get_title(), fontproperties=korean_font_prop)
        
        # x축, y축 틱 레이블에 폰트 적용
        for label in ax.get_xticklabels():
            label.set_fontproperties(korean_font_prop)
        for label in ax.get_yticklabels():
            label.set_fontproperties(korean_font_prop)

# CSS 스타일 추가
st.markdown("""
<style>
    .main-title {
        text-align: center;
        padding: 2rem 0;
        color: #1E3A8A;
    }
    .subtitle {
        text-align: center;
        color: #6B7280;
        margin-bottom: 3rem;
    }
    .container {
        max-width: 650px;
        margin: 0 auto;
        padding: 0 1rem;
    }
    /* Streamlit columns 간격 조정 */
    .stColumn {
        padding-left: 0.25rem !important;
        padding-right: 0.25rem !important;
    }
    .card {
        padding: 1rem;
        border-radius: 0.5rem;
        background: white;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        height: 140px;
        display: flex;
        flex-direction: column;
        margin-bottom: 1.5rem;
        max-width: 500px;
        margin-left: auto;
        margin-right: auto;
        cursor: pointer;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 12px -1px rgba(0, 0, 0, 0.15);
    }
    .card-title {
        color: #2563EB;
        font-size: 1.15rem;
        font-weight: bold;
        margin-bottom: 0.3rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .card-content {
        color: #4B5563;
        font-size: 0.9rem;
        line-height: 1.35;
        flex-grow: 1;
    }
    .feature-icon {
        font-size: 1.4rem;
    }
    .start-section {
        text-align: center;
        margin-top: 2rem;
        padding: 2rem;
    }
    /* 탭 폰트 크기 확대 - 더 강력한 선택자 사용 */
    div[data-testid="stTabs"] button[data-baseweb="tab"] {
        font-size: 2.0rem !important;
        font-weight: 600 !important;
        padding: 0.75rem 1rem !important;
    }
    div[data-testid="stTabs"] [data-baseweb="tab-list"] button {
        font-size: 2.0rem !important;
        font-weight: 600 !important;
        padding: 0.75rem 1rem !important;
    }
    div[data-testid="stTabs"] [data-baseweb="tab-list"] {
        font-size: 2.0rem !important;
    }
    .stTabs > div > div > div > div > button {
        font-size: 2.0rem !important;
        font-weight: 600 !important;
        padding: 0.75rem 1rem !important;
    }
    /* 추가적인 탭 텍스트 타겟팅 */
    [data-testid="stTabs"] button {
        font-size: 2.0rem !important;
        font-weight: 600 !important;
    }
    /* 불용어 버튼을 컴팩트하게 만들기 */
    .stButton > button {
        font-size: 0.7rem !important;
        padding: 0.1rem 0.3rem !important;
        height: 1.5rem !important;
        min-height: 1.5rem !important;
        width: auto !important;
        min-width: 45px !important;
        max-width: 80px !important;
        margin: 1px !important;
        border-radius: 0.25rem !important;
    }
</style>
""", unsafe_allow_html=True)

# 제목과 부제목
st.markdown("<h1 class='main-title'>네이버 스마트 스토어 데이터 분석</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>스마트 스토어의 데이터를 분석하여 인사이트를 도출하는 서비스입니다.</p>", unsafe_allow_html=True)

# 함수: 불용어 관리 UI 생성
def render_stopwords_ui():
    """불용어 관리 UI를 표시합니다."""
    # 구분선과 제목
    st.markdown("---")
    st.subheader("🔧 불용어 관리")
    st.info("불용어는 워드클라우드에서 제외되는 단어입니다. 불필요하게 자주 등장하는 단어를 추가하면 더 의미 있는 분석이 가능합니다.")
    
    # 현재 불용어 목록과 추가 기능을 좌우로 배치
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # 현재 불용어 목록 표시
        st.markdown("**📋 현재 불용어 목록**")
        current_stopwords = get_stopwords()
        
        # 불용어를 더 많은 열로 표시 (8열로 증가)
        if current_stopwords:
            cols = st.columns(8)  # 6열에서 8열로 증가
            for i, word in enumerate(sorted(current_stopwords)):
                with cols[i % 8]:
                    if st.button(f"✕ {word}", key=f"remove_{word}", help=f"'{word}' 삭제"):
                        remove_stopword(word)
                        st.rerun()
        else:
            st.write("등록된 불용어가 없습니다.")
    
    with col2:
        # 새 불용어 추가
        st.markdown("**➕ 불용어 추가**")
        
        # 폼을 사용해서 엔터키와 버튼 클릭 모두 처리
        with st.form("add_stopword_form", clear_on_submit=True):
            new_stopword = st.text_input("추가할 단어", placeholder="예: 제품, 상품")
            
            col2_1, col2_2 = st.columns(2)
            with col2_1:
                submitted = st.form_submit_button("추가", use_container_width=True)
            
            with col2_2:
                if st.form_submit_button("초기화", use_container_width=True):
                    reset_stopwords()
                    st.rerun()
            
            # 엔터키나 추가 버튼 클릭 시 실행
            if submitted and new_stopword.strip():
                add_stopword(new_stopword)
                st.rerun()

# 함수: 파일 유형 자동 감지
def detect_file_type(df):
    """업로드된 파일의 유형을 자동으로 감지합니다"""
    # 리뷰 파일 감지
    potential_review_columns = ['REVIEW_CONTENT', 'review_content', '리뷰내용', '내용', 'CONTENT']
    if any(col in df.columns for col in potential_review_columns):
        return "review"
    
    # 옵션 비율 파일 감지
    potential_option_columns = ['OPTION_INFO', 'option_info', '옵션정보', '옵션명', '상품옵션']
    potential_count_columns = ['COUNT', 'count', '수량', '판매량', '판매수량']
    if any(col in df.columns for col in potential_option_columns) and any(col in df.columns for col in potential_count_columns):
        return "option"
    
    # 스토어 전체 판매현황 파일 감지
    potential_sales_columns = ['상품명', '매출', '판매건수', '기본판매가격']
    if '상품명' in df.columns and any('매출' in str(col) for col in df.columns):
        return "sales"
    
    # 기타 파일은 sales로 간주
    return "sales"

# 함수: 리뷰 데이터프레임 컬럼 이름 확인 및 수정
def check_review_columns(df):
    """리뷰 데이터 컬럼 이름 확인 및 표준화"""
    # 리뷰 내용을 담는 컬럼 확인
    potential_review_columns = ['REVIEW_CONTENT', 'review_content', '리뷰내용', '내용', 'CONTENT']
    review_col = None
    
    for col in potential_review_columns:
        if col in df.columns:
            review_col = col
            break
    
    if review_col and review_col != 'review_content':
        df = df.rename(columns={review_col: 'review_content'})
    
    return df

# 함수: 옵션 데이터프레임 컬럼 이름 확인 및 수정
def check_option_columns(df):
    """옵션 데이터 컬럼 이름 확인 및 표준화"""
    # 옵션 정보를 담는 컬럼 확인
    potential_option_columns = ['OPTION_INFO', 'option_info', '옵션정보', '옵션명', '상품옵션']
    option_col = None
    
    for col in potential_option_columns:
        if col in df.columns:
            option_col = col
            break
    
    # 수량/판매량 정보를 담는 컬럼 확인
    potential_count_columns = ['COUNT', 'count', '수량', '판매량', '판매수량']
    count_col = None
    
    for col in potential_count_columns:
        if col in df.columns:
            count_col = col
            break
    
    # 컬럼명 표준화
    if option_col and option_col != 'option_info':
        df = df.rename(columns={option_col: 'option_info'})
    
    if count_col and count_col != 'count':
        df = df.rename(columns={count_col: 'count'})
    
    return df

# 사이드바 - 파일 업로드 및 메뉴
with st.sidebar:
    st.header("데이터 업로드")
    uploaded_file = st.file_uploader("스마트 스토어 데이터 파일", type=["xlsx", "csv"], help="리뷰 분석, 옵션 비율, 판매 현황 등의 파일을 업로드하세요.")
    
    # 파일 타입 설명
    with st.expander("파일 타입 설명"):
        st.info("""
        • 리뷰 분석 파일: 리뷰 내용 컬럼을 포함한 파일
        • 옵션 비율 파일: 옵션 정보와 판매량/수량 컬럼을 포함한 파일
        • 판매 현황 파일: 기타 판매 관련 파일
        
        파일 유형은 자동으로 감지됩니다.
        """)
    
    st.header("분석 메뉴")
    # 세션 상태 초기화
    if 'analysis_option' not in st.session_state:
        st.session_state.analysis_option = "홈"
    
    analysis_option = st.radio(
        "분석 유형 선택",
        ["홈", "리뷰 분석 - 워드클라우드", "리뷰 분석 - 감정분석", "옵션 분석", "스토어 전체 판매현황"],
        index=["홈", "리뷰 분석 - 워드클라우드", "리뷰 분석 - 감정분석", "옵션 분석", "스토어 전체 판매현황"].index(st.session_state.analysis_option) if st.session_state.analysis_option in ["홈", "리뷰 분석 - 워드클라우드", "리뷰 분석 - 감정분석", "옵션 분석", "스토어 전체 판매현황"] else 0
    )
    
    # 라디오 버튼 선택이 변경되면 세션 상태 업데이트
    if analysis_option != st.session_state.analysis_option:
        st.session_state.analysis_option = analysis_option

# 데이터 저장 변수
review_df = None
option_df = None
sales_df = None

# 메인 화면
if st.session_state.analysis_option == "홈":
    st.markdown('<div class="container">', unsafe_allow_html=True)
    
    # 첫 번째 행
    col1, col2 = st.columns([1, 1], gap="small")
    
    with col1:
        # 리뷰 워드클라우드 분석 카드
        st.markdown("""
        <div class="card">
            <div class="card-title">
                <span class="feature-icon">📊</span>
                리뷰 워드클라우드 분석
            </div>
            <div class="card-content">
                • 고객 리뷰에서 자주 등장하는 키워드를 시각화<br>
                • 불용어 관리로 분석 정확도 향상<br>
                • 직관적인 워드클라우드와 Top 20 키워드 차트
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # 리뷰 감정 분석 카드
        st.markdown("""
        <div class="card">
            <div class="card-title">
                <span class="feature-icon">😊</span>
                리뷰 감정 분석
            </div>
            <div class="card-content">
                • 고객 리뷰의 감정 분석 (긍정/중립/부정)<br>
                • 감정 분포 시각화<br>
                • 고객 만족도 트렌드 파악
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # 두 번째 행
    col3, col4 = st.columns([1, 1], gap="small")
    
    with col3:
        # 옵션 분석 카드
        st.markdown("""
        <div class="card">
            <div class="card-title">
                <span class="feature-icon">🎯</span>
                옵션 분석
            </div>
            <div class="card-content">
                • 상품 옵션별 판매 비율 분석<br>
                • 인기 옵션 Top 10 시각화<br>
                • 재고 관리 및 마케팅 전략 수립 지원
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        # 스토어 전체 판매 현황 카드
        st.markdown("""
        <div class="card">
            <div class="card-title">
                <span class="feature-icon">📈</span>
                스토어 전체 판매 현황
            </div>
            <div class="card-content">
                • 일별/월별 판매 추이 분석<br>
                • 매출 및 주문 데이터 시각화<br>
                • 성장률 및 성과 지표 분석
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 시작하기 섹션
    st.markdown("""
    <div class="start-section">
        <h2 style="color: #1E3A8A;">시작하기</h2>
        <p style="color: #6B7280;">
            👈 왼쪽 사이드바에서 파일을 업로드하거나 분석 메뉴를 선택하여 시작해보세요.<br>
            파일 없이도 샘플 데이터로 각 분석 기능을 체험할 수 있습니다.<br>
            📖 각 분석에 대한 상세한 설명은 '분석 가이드' 메뉴에서 확인하세요.
        </p>
    </div>
    """, unsafe_allow_html=True)

elif st.session_state.analysis_option == "📖 분석 가이드":
    st.header("📖 분석 가이드")
    
    # 가이드 타입 선택
    guide_type = st.selectbox(
        "가이드를 보고 싶은 분석을 선택하세요:",
        ["워드클라우드", "감정분석", "옵션분석", "판매현황"]
    )
    
    st.markdown("---")
    
    if guide_type == "워드클라우드":
        st.subheader("📊 리뷰 워드클라우드 분석 가이드")
        st.markdown("""
        ### 🔍 워드클라우드란?
        워드클라우드는 텍스트 데이터에서 **자주 등장하는 단어를 크기로 표현**한 시각화 방법입니다.
        
        ### 📈 제공하는 분석
        1. **워드클라우드 시각화**: 리뷰에서 자주 언급되는 키워드를 시각적으로 표현
        2. **상위 20개 키워드 차트**: 언급 횟수가 많은 순서대로 막대그래프로 표시
        
        ### 🔧 불용어 관리
        **불용어**는 분석에서 제외할 단어들입니다. '제품', '상품', '구매' 등 너무 일반적인 단어를 제외하여 더 의미있는 분석이 가능합니다.
        
        ### 💡 활용 팁
        - **고객 관심사 파악**: 자주 언급되는 키워드로 고객이 중요하게 생각하는 요소 확인
        - **제품 개선점 발견**: 부정적 키워드를 통해 개선이 필요한 부분 파악
        - **마케팅 포인트 도출**: 긍정적 키워드를 마케팅 문구에 활용
        """)
        
        if st.button("🚀 워드클라우드 분석 시작하기", use_container_width=True):
            st.session_state.analysis_option = "리뷰 분석 - 워드클라우드"
            st.rerun()
    
    elif guide_type == "감정분석":
        st.subheader("😊 리뷰 감정분석 가이드")
        st.markdown("""
        ### 🎯 감정분석이란?
        고객 리뷰의 **감정을 자동으로 분류**하여 긍정, 중립, 부정으로 나누는 분석입니다.
        
        ### 🏷️ 세부 카테고리 분석
        - **긍정**: 맛, 식감, 배송, 가격, 서비스, 품질, 외관, 양 (8개 카테고리)
        - **중립**: 일반적, 애매한 맛, 보통 품질 등 (7개 카테고리)
        - **부정**: 맛 문제, 품질 문제, 배송 문제 등 (8개 카테고리)
        
        ### 📈 제공하는 분석
        1. **감정 분포**: 전체 리뷰의 감정 비율을 막대그래프와 파이차트로 표시
        2. **카테고리별 분석**: 각 감정별로 세부 카테고리 분석 및 주요 키워드 추출
        
        ### 💡 활용 방법
        - **고객 만족도 측정**: 긍정/부정 비율로 전반적인 만족도 파악
        - **개선점 발견**: 부정 카테고리에서 주요 문제점 확인
        - **마케팅 전략**: 긍정 키워드를 활용한 홍보 포인트 도출
        """)
        
        if st.button("🚀 감정분석 시작하기", use_container_width=True):
            st.session_state.analysis_option = "리뷰 분석 - 감정분석"
            st.rerun()
    
    elif guide_type == "옵션분석":
        st.subheader("🎯 옵션 분석 가이드")
        st.markdown("""
        ### 🛍️ 옵션 분석이란?
        상품의 **다양한 옵션별 판매 수량**을 분석하여 어떤 옵션이 가장 인기가 있는지 파악하는 분석입니다.
        
        ### 📊 제공하는 분석
        1. **상위 10개 옵션**: 판매량이 많은 순서대로 옵션 순위 표시
        2. **판매량 시각화**: 막대그래프로 옵션별 판매량 비교
        
        ### 💡 활용 방법
        - **재고 관리**: 인기 옵션의 재고를 충분히 확보
        - **마케팅 전략**: 인기 옵션을 메인으로 홍보
        - **상품 기획**: 고객 선호도가 높은 옵션 특성 파악
        """)
        
        if st.button("🚀 옵션 분석 시작하기", use_container_width=True):
            st.session_state.analysis_option = "옵션 분석"
            st.rerun()
    
    elif guide_type == "판매현황":
        st.subheader("📈 스토어 전체 판매현황 가이드")
        st.markdown("""
        ### 🏪 판매현황 분석이란?
        스토어의 **전체 상품 판매 데이터**를 다각도로 분석하여 매출 성과와 트렌드를 파악하는 종합 분석입니다.
        
        ### 📊 제공하는 4가지 분석
        
        #### 1. 🏆 매출 랭킹
        - 기간별 매출 상위 10개 상품 순위
        
        #### 2. ⚡ 가격대비 매출지수 분석
        **가격대비 매출지수 = 해당 기간 매출 ÷ 기본판매가격**
        - 상품 가격 대비 얼마나 효율적으로 매출을 올렸는지 측정
        - 높은 지수 = 가격 대비 매출 효율성이 좋음
        
        #### 3. 💰 가격대별 분석
        - 저가(10,000원 미만), 중저가(10,000~30,000원), 중가(30,000~50,000원), 고가(50,000원 이상)
        - 각 가격대별 상품 수와 평균 매출 분석
        
        #### 4. ⭐ 리뷰-매출 상관관계
        **상관계수 해석:**
        - 0.7~1.0: 강한 양의 상관관계 (리뷰 점수 ↑ → 매출 ↑)
        - 0.3~0.7: 중간 양의 상관관계
        - 0.0~0.3: 약한 상관관계
        
        ### 💡 활용 방법
        - **매출 전략**: 매출 상위 상품의 성공 요인 분석
        - **가격 전략**: 가격대별 매출 성과 비교
        - **품질 관리**: 리뷰 점수와 매출의 관계 파악
        """)
        
        if st.button("🚀 판매현황 분석 시작하기", use_container_width=True):
            st.session_state.analysis_option = "스토어 전체 판매현황"
            st.rerun()

elif uploaded_file is None and st.session_state.analysis_option not in ["홈", "📖 분석 가이드"]:
    # 파일이 업로드되지 않았지만 분석이 선택된 경우 샘플 데이터 사용
    try:
        # 샘플 데이터 로드
        if st.session_state.analysis_option in ["리뷰 분석 - 워드클라우드", "리뷰 분석 - 감정분석"]:
            review_df = pd.read_excel("data/reviewcontents (4).xlsx")
            review_df = check_review_columns(review_df)
        elif st.session_state.analysis_option == "스토어 전체 판매현황":
            sales_df = pd.read_excel("data/스토어전체판매현황 (2).xlsx")
        
        if st.session_state.analysis_option == "옵션 분석":
            option_df = pd.read_excel("data/옵션비율 (2).xlsx")
            option_df = check_option_columns(option_df)
        
        # 분석 실행
        if st.session_state.analysis_option == "리뷰 분석 - 워드클라우드":
            st.header("리뷰 워드클라우드 분석")
            
            # 불용어 관리 UI 표시
            render_stopwords_ui()
            
            # 섹션 구분을 위한 간격과 구분선
            st.markdown("---")
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("📊 워드클라우드 분석 결과")
            st.markdown("<br>", unsafe_allow_html=True)
            
            with st.spinner("워드클라우드 생성 중..."):
                word_count, top_words = generate_wordcloud_data(review_df, 'review_content')
                
                # 워드클라우드 생성
                if word_count:
                    wc = create_wordcloud(word_count)
                    
                    # 워드클라우드와 상위 20개 단어를 좌우로 배치
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        # 워드클라우드 제목 추가 (중앙 정렬)
                        st.markdown("<h3 style='text-align: center;'>워드클라우드</h3>", unsafe_allow_html=True)
                        
                        # 워드클라우드 표시
                        fig1, ax = plt.subplots(figsize=(8, 8))
                        ax.imshow(wc, interpolation='bilinear')
                        ax.axis('off')
                        plt.tight_layout(pad=0)
                        st.pyplot(fig1, use_container_width=True)
                        plt.close(fig1)  # 메모리 정리
                    
                    with col2:
                        # 상위 20개 단어 표시 (중앙 정렬)
                        st.markdown("<h3 style='text-align: center;'>상위 20개 단어</h3>", unsafe_allow_html=True)
                        
                        # 상위 단어 막대 그래프
                        top_words_df = pd.DataFrame({
                            '단어': list(top_words.keys()),
                            '언급 횟수': list(top_words.values())
                        })
                        
                        # 리뷰수 기준 내림차순 정렬 (높은 수가 위쪽에)
                        top_words_df = top_words_df.sort_values('언급 횟수', ascending=True)
                        
                        # 상위 20개 단어 차트 생성
                        fig2, ax = plt.subplots(figsize=(8, 8))
                        bars = ax.barh(top_words_df['단어'], top_words_df['언급 횟수'], color='steelblue')
                        
                        # 리뷰 수 표시
                        for i, bar in enumerate(bars):
                            width = bar.get_width()
                            ax.text(width + width*0.02, bar.get_y() + bar.get_height()/2, 
                                    f'{int(width):,}', 
                                    va='center', fontsize=10)
                    
                        # x축 범위 조정 (여백 줄이기)
                        if len(top_words) > 0:
                            max_count = max(top_words.values())
                            ax.set_xlim(0, max_count * 1.15)  # 텍스트 위한 여유 공간
                    
                        # y축 레이블 폰트 크기 조정
                        ax.tick_params(axis='y', labelsize=10)
                        ax.tick_params(axis='x', labelsize=10)
                    
                        # 한글 폰트 적용
                        set_korean_font(ax)
                    
                        # 그래프 제목 및 레이아웃 조정
                        ax.set_title('')
                        plt.tight_layout(pad=0)
                        st.pyplot(fig2, use_container_width=True)
                        plt.close(fig2)  # 메모리 정리
                else:
                    st.warning("분석할 리뷰 데이터가 충분하지 않습니다.")
        
        elif st.session_state.analysis_option == "리뷰 분석 - 감정분석":
            st.header("리뷰 감정분석")
            
            with st.spinner("감정 분석 중..."):
                # 감정 분석 수행
                df_sentiment, sentiment_counts = simple_sentiment_analysis(review_df, 'review_content')
                
                # 감정 분석 결과 표시
                col1, col2 = st.columns(2)
                
                with col1:
                    # 감정별 리뷰 수 막대 그래프
                    fig, ax = plt.subplots(figsize=(6, 4))
                    
                    # 감정별 색상 매핑
                    emotion_colors = {'긍정': '#28a745', '중립': '#ffa500', '부정': '#dc3545'}
                    colors = [emotion_colors[emotion] for emotion in sentiment_counts['감정']]
                    
                    ax.bar(sentiment_counts['감정'], sentiment_counts['리뷰 수'], color=colors)
                    ax.set_title('감정별 리뷰 수', pad=20)
                    ax.set_ylabel('리뷰 수')
                    for i, v in enumerate(sentiment_counts['리뷰 수']):
                        ax.text(i, v + max(sentiment_counts['리뷰 수']) * 0.01, str(v), ha='center', va='bottom')
                    
                    # y축 범위 조정 (위쪽 여백 확보)
                    max_val = max(sentiment_counts['리뷰 수'])
                    ax.set_ylim(0, max_val * 1.15)
                    
                    # 한글 폰트 적용
                    set_korean_font(ax)
                    
                    st.pyplot(fig)
                
                with col2:
                    # 감정 비율 파이 차트
                    fig, ax = plt.subplots(figsize=(6, 4))
                    
                    # 감정별 색상 매핑
                    emotion_colors = {'긍정': '#28a745', '중립': '#ffa500', '부정': '#dc3545'}
                    colors = [emotion_colors[emotion] for emotion in sentiment_counts['감정']]
                    
                    ax.pie(sentiment_counts['리뷰 수'], labels=sentiment_counts['감정'], 
                           autopct='%1.1f%%', colors=colors, startangle=90)
                    ax.set_title('감정 분포 비율', pad=20)
                    ax.axis('equal')
                    
                    # 한글 폰트 적용
                    set_korean_font(ax)
                    
                    st.pyplot(fig)
                
                # 섹션 구분
                st.markdown("---")
                st.markdown("<br>", unsafe_allow_html=True)
                
                # 감정별 리뷰 분석
                st.subheader("감정별 리뷰 카테고리 분석")
                
                # 탭 폰트 크기 강제 적용
                st.markdown("""
                <style>
                .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
                    font-size: 24px !important;
                    font-weight: 600 !important;
                }
                </style>
                """, unsafe_allow_html=True)
                
                # 탭 생성
                tab1, tab2, tab3 = st.tabs(["긍정 리뷰", "중립 리뷰", "부정 리뷰"])
                
                with tab1:
                    # 긍정 리뷰 카테고리 분석
                    st.markdown("### 📊 긍정 리뷰 카테고리 분석")
                    with st.spinner("긍정 리뷰 카테고리 분석 중..."):
                        positive_category_analysis = analyze_positive_review_categories(df_sentiment, 'review_content')
                        
                        if not positive_category_analysis.empty:
                            st.dataframe(positive_category_analysis, use_container_width=True, hide_index=True)
                            
                            st.markdown("<br>", unsafe_allow_html=True)
                            
                            # 카테고리별 리뷰 수 시각화
                            if len(positive_category_analysis) > 0:
                                fig, ax = plt.subplots(figsize=(8, 4))
                                ax.bar(positive_category_analysis['카테고리'], positive_category_analysis['리뷰 수'], color='#28a745')
                                ax.set_title('긍정 리뷰 카테고리별 언급 빈도')
                                ax.set_ylabel('리뷰 수')
                                ax.tick_params(axis='x', rotation=45)
                                
                                # 한글 폰트 적용
                                set_korean_font(ax)
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                        else:
                            st.info("긍정 리뷰에서 분석 가능한 카테고리를 찾을 수 없습니다.")
                
                with tab2:
                    # 중립 리뷰 카테고리 분석
                    st.markdown("### 📊 중립 리뷰 카테고리 분석")
                    with st.spinner("중립 리뷰 카테고리 분석 중..."):
                        neutral_category_analysis = analyze_neutral_review_categories(df_sentiment, 'review_content')
                        
                        if not neutral_category_analysis.empty:
                            st.dataframe(neutral_category_analysis, use_container_width=True, hide_index=True)
                            
                            st.markdown("<br>", unsafe_allow_html=True)
                            
                            # 카테고리별 리뷰 수 시각화
                            if len(neutral_category_analysis) > 0:
                                fig, ax = plt.subplots(figsize=(8, 4))
                                
                                # 막대 너비 설정 (카테고리 수에 따라 조정)
                                bar_width = max(0.3, min(0.6, 2.0 / len(neutral_category_analysis)))
                                
                                bars = ax.bar(range(len(neutral_category_analysis)), 
                                            neutral_category_analysis['리뷰 수'], 
                                            width=bar_width, 
                                            color='#ffa500')
                                
                                # 막대 위에 숫자 표시
                                for i, v in enumerate(neutral_category_analysis['리뷰 수']):
                                    ax.text(i, v + max(neutral_category_analysis['리뷰 수']) * 0.02, 
                                           str(v), ha='center', va='bottom')
                                
                                # y축 범위 조정 (위쪽 여백 확보)
                                max_val = max(neutral_category_analysis['리뷰 수'])
                                ax.set_ylim(0, max_val * 1.15)
                                
                                # x축 설정
                                ax.set_xticks(range(len(neutral_category_analysis)))
                                ax.set_xticklabels(neutral_category_analysis['카테고리'], rotation=45)
                                
                                ax.set_title('중립 리뷰 카테고리별 언급 빈도')
                                ax.set_ylabel('리뷰 수')
                                
                                # 한글 폰트 적용
                                set_korean_font(ax)
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                        else:
                            st.info("중립 리뷰에서 분석 가능한 카테고리를 찾을 수 없습니다.")
                
                with tab3:
                    # 부정 리뷰 카테고리 분석
                    st.markdown("### 📊 부정 리뷰 카테고리 분석")
                    with st.spinner("부정 리뷰 카테고리 분석 중..."):
                        negative_category_analysis = analyze_negative_review_categories(df_sentiment, 'review_content')
                        
                        if not negative_category_analysis.empty:
                            st.dataframe(negative_category_analysis, use_container_width=True, hide_index=True)
                            
                            st.markdown("<br>", unsafe_allow_html=True)
                            
                            # 카테고리별 리뷰 수 시각화
                            if len(negative_category_analysis) > 0:
                                fig, ax = plt.subplots(figsize=(8, 4))
                                
                                # 막대 너비 설정 (카테고리 수에 따라 조정)
                                bar_width = max(0.3, min(0.6, 2.0 / len(negative_category_analysis)))
                                
                                bars = ax.bar(range(len(negative_category_analysis)), 
                                            negative_category_analysis['리뷰 수'], 
                                            width=bar_width, 
                                            color='#dc3545')
                                
                                # 막대 위에 숫자 표시
                                for i, v in enumerate(negative_category_analysis['리뷰 수']):
                                    ax.text(i, v + max(negative_category_analysis['리뷰 수']) * 0.02, 
                                           str(v), ha='center', va='bottom')
                                
                                # y축 범위 조정 (위쪽 여백 확보)
                                max_val = max(negative_category_analysis['리뷰 수'])
                                ax.set_ylim(0, max_val * 1.15)
                                
                                # x축 설정
                                ax.set_xticks(range(len(negative_category_analysis)))
                                ax.set_xticklabels(negative_category_analysis['카테고리'], rotation=45)
                                
                                ax.set_title('부정 리뷰 카테고리별 언급 빈도')
                                ax.set_ylabel('리뷰 수')
                                
                                # 한글 폰트 적용
                                set_korean_font(ax)
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                        else:
                            st.info("부정 리뷰에서 분석 가능한 카테고리를 찾을 수 없습니다.")
        
        elif st.session_state.analysis_option == "옵션 분석":
            st.header("옵션 분석")
            
            with st.spinner("옵션 분석 중..."):
                # 옵션 분석 수행
                top_options = analyze_options(option_df, 'option_info', 'count')
                
                # 상위 10개 옵션 표시
                st.subheader("상위 10개 옵션")
                
                # 표를 적절한 크기로 표시 (떨림 방지)
                st.markdown("""
                <style>
                div[data-testid="stDataFrame"] {
                    width: 800px !important;
                    max-width: 800px !important;
                    overflow: visible !important;
                    margin-left: 50px !important;
                }
                </style>
                """, unsafe_allow_html=True)
                st.dataframe(top_options)
                
                # 간격 추가
                st.markdown("<br><br>", unsafe_allow_html=True)
                
                # 상위 10개 옵션 막대 그래프
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # 막대 그래프 생성 (인덱스를 X축 위치로 사용)
                x_positions = range(len(top_options))
                bars = ax.bar(x_positions, top_options['count'], color='steelblue')
                
                # X축 레이블 설정 (옵션명)
                ax.set_xticks(x_positions)
                ax.set_xticklabels(top_options['option_info'], rotation=45, ha='right')
                
                # 막대 위에 판매량 표시
                for i, v in enumerate(top_options['count']):
                    ax.text(i, v + max(top_options['count']) * 0.01, 
                           f'{v:,}', ha='center', va='bottom')
                
                # Y축 범위 조정 (위쪽 여백 확보)
                max_val = max(top_options['count'])
                ax.set_ylim(0, max_val * 1.15)
                
                ax.set_title('상위 10개 옵션 판매량')
                ax.set_ylabel('판매량')
                
                # 한글 폰트 적용
                set_korean_font(ax)
                
                plt.tight_layout()
                st.pyplot(fig)
        
        elif st.session_state.analysis_option == "스토어 전체 판매현황":
            st.header("🏪 스토어 전체 판매현황 분석")
            
            # 사용 가능한 기간 가져오기
            available_periods = get_sales_periods(sales_df)
            
            if len(available_periods) == 0:
                st.error("매출 데이터를 찾을 수 없습니다.")
            else:
                # 기간 선택 필터
                st.subheader("📅 분석 기간 선택")
                
                # selectbox 커서 스타일 추가
                st.markdown("""
                <style>
                div[data-baseweb="select"] {
                    cursor: pointer !important;
                }
                div[data-baseweb="select"] > div {
                    cursor: pointer !important;
                }
                </style>
                """, unsafe_allow_html=True)
                
                selected_period = st.selectbox(
                    "매출 분석 기간을 선택하세요:",
                    available_periods,
                    index=len(available_periods) - 1 if '1년' in available_periods else 0
                )
                
                # 섹션 구분
                st.markdown("<br>", unsafe_allow_html=True)
                
                # 매출 요약 통계
                st.subheader("📊 매출 요약 통계")
                summary_stats = get_sales_summary_stats(sales_df, selected_period)
                
                if summary_stats:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("총 매출", f"{summary_stats['총매출']:,}원")
                    with col2:
                        st.metric("평균 매출", f"{summary_stats['평균매출']:,}원")
                    with col3:
                        st.metric("상품 수", f"{summary_stats['상품수']:,}개")
                    with col4:
                        st.metric("최대 매출", f"{summary_stats['최대매출']:,}원")
                
                # 섹션 구분
                st.markdown("---")
                st.markdown("<br>", unsafe_allow_html=True)
                
                                                    # 분석 탭 생성
                st.subheader("📈 상세 분석")
                
                # 탭 폰트 크기 강제 적용
                st.markdown("""
                <style>
                .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
                    font-size: 24px !important;
                    font-weight: 600 !important;
                }
                </style>
                """, unsafe_allow_html=True)
                
                tab1, tab2, tab3, tab4 = st.tabs(["매출 랭킹", "가격대비 매출지수", "가격대별 분석", "리뷰-매출 상관관계"])
                
                with tab1:
                    st.subheader(f"🏆 {selected_period} 매출 상위 10개 상품")
                    top_products = analyze_top_products_by_period(sales_df, selected_period, 10)
                    
                    if not top_products.empty:
                        st.dataframe(top_products, use_container_width=True, hide_index=True)
                        
                        # 표와 그래프 사이 간격 추가
                        st.markdown("<br><br>", unsafe_allow_html=True)
                        
                        # 매출 랭킹 시각화
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        # 막대 그래프 생성
                        bars = ax.bar(range(len(top_products)), 
                                     top_products[f'{selected_period} 매출'], 
                                     color='steelblue')
                        
                        # 상품명을 x축 레이블로 설정 (회전)
                        ax.set_xticks(range(len(top_products)))
                        ax.set_xticklabels([name[:15] + '...' if len(name) > 15 else name 
                                           for name in top_products['상품명']], 
                                          rotation=45, ha='right')
                        
                        # 막대 위에 매출 표시
                        for i, v in enumerate(top_products[f'{selected_period} 매출']):
                            ax.text(i, v + max(top_products[f'{selected_period} 매출']) * 0.01, 
                                   f'{v:,.0f}', ha='center', va='bottom', fontsize=8)
                        
                        ax.set_ylabel('매출 (원)')
                        ax.set_title(f'{selected_period} 매출 상위 10개 상품')
                        
                        # 한글 폰트 적용
                        set_korean_font(ax)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.info("매출 데이터가 없습니다.")
                
                with tab2:
                    st.subheader(f"⚡ {selected_period} 가격대비 매출지수 분석")
                    efficiency_data = analyze_sales_efficiency(sales_df, selected_period)
                    
                    if not efficiency_data.empty:
                        st.dataframe(efficiency_data, use_container_width=True, hide_index=True)
                        
                        # 표와 그래프 사이 간격 추가
                        st.markdown("<br><br>", unsafe_allow_html=True)
                        
                        # 가격대비 매출지수 시각화
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        bars = ax.bar(range(len(efficiency_data)), 
                                     efficiency_data['가격대비매출지수'], 
                                     color='orange')
                        
                        ax.set_xticks(range(len(efficiency_data)))
                        ax.set_xticklabels([name[:15] + '...' if len(name) > 15 else name 
                                           for name in efficiency_data['상품명']], 
                                          rotation=45, ha='right')
                        
                        # 막대 위에 지수 표시
                        for i, v in enumerate(efficiency_data['가격대비매출지수']):
                            ax.text(i, v + max(efficiency_data['가격대비매출지수']) * 0.01, 
                                   f'{v:.1f}', ha='center', va='bottom', fontsize=8)
                        
                        ax.set_ylabel('가격대비매출지수 (매출÷가격)')
                        ax.set_title(f'{selected_period} 가격대비 매출지수 상위 10개 상품')
                        
                        # 한글 폰트 적용
                        set_korean_font(ax)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.info("가격대비 매출지수 분석을 위한 데이터가 부족합니다.")
                
                with tab3:
                    st.subheader(f"💰 가격대별 {selected_period} 매출 분석")
                    price_segments = analyze_price_segments(sales_df, selected_period)
                    
                    if not price_segments.empty:
                        st.dataframe(price_segments, use_container_width=True, hide_index=True)
                        
                        # 가격대별 분석 시각화
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                        
                        # 가격대별 상품수
                        ax1.bar(price_segments['가격대'], price_segments['상품수'], color='lightblue')
                        ax1.set_title('가격대별 상품 수')
                        ax1.set_ylabel('상품 수')
                        ax1.tick_params(axis='x', rotation=45)
                        
                        # 가격대별 평균매출
                        ax2.bar(price_segments['가격대'], price_segments['평균매출'], color='lightgreen')
                        ax2.set_title('가격대별 평균 매출')
                        ax2.set_ylabel('평균 매출 (원)')
                        ax2.tick_params(axis='x', rotation=45)
                        
                        # 한글 폰트 적용
                        set_korean_font(ax1)
                        set_korean_font(ax2)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.info("가격대별 분석을 위한 데이터가 부족합니다.")
                    
                    with tab4:
                        st.subheader(f"⭐ 리뷰 점수와 {selected_period} 매출 상관관계")
                        correlation, review_analysis = analyze_review_sales_correlation(sales_df, selected_period)
                        
                        if correlation is not None:
                            st.info(f"**상관계수: {correlation:.3f}**")
                            
                            if not review_analysis.empty:
                                st.dataframe(review_analysis, use_container_width=True, hide_index=True)
                                
                                # 리뷰 점수별 평균 매출 시각화
                                fig, ax = plt.subplots(figsize=(10, 6))
                                
                                bars = ax.bar(review_analysis['리뷰점수구간'], 
                                             review_analysis['평균매출'], 
                                             color='gold')
                                
                                # 막대 위에 값 표시
                                for i, v in enumerate(review_analysis['평균매출']):
                                    ax.text(i, v + max(review_analysis['평균매출']) * 0.01, 
                                           f'{v:,.0f}', ha='center', va='bottom')
                                
                                ax.set_title('리뷰 점수 구간별 평균 매출')
                                ax.set_ylabel('평균 매출 (원)')
                                
                                # 한글 폰트 적용
                                set_korean_font(ax)
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                        else:
                            st.info("리뷰-매출 상관관계 분석을 위한 데이터가 부족합니다.")
        
    except Exception as e:
        st.error(f"샘플 데이터를 불러오는 중 오류가 발생했습니다: {e}")
        st.info("홈으로 돌아가서 파일을 직접 업로드해주세요.")

else:
    # 데이터 로드
    try:
        # 파일 업로드 시 데이터 로드
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        # 파일 유형 감지
        file_type = detect_file_type(df)
        
        if file_type == "review":
            review_df = check_review_columns(df)
            st.sidebar.success("리뷰 파일이 업로드되었습니다.")
        elif file_type == "option":
            option_df = check_option_columns(df)
            st.sidebar.success("옵션 비율 파일이 업로드되었습니다.")
        else:  # sales
            sales_df = df
            st.sidebar.success("판매 현황 파일이 업로드되었습니다.")
        
        # 파일 정보 표시
        if st.session_state.analysis_option == "홈":
            st.subheader("업로드된 파일 정보")
            st.write(f"파일명: {uploaded_file.name}")
            st.write(f"파일 유형: {'리뷰 분석 파일' if file_type == 'review' else '옵션 비율 파일' if file_type == 'option' else '판매 현황 파일'}")
            st.dataframe(df.head(3))
        
        # 분석 유형에 따른 처리
        if st.session_state.analysis_option == "리뷰 분석 - 워드클라우드":
            if file_type == "review":
                st.header("리뷰 워드클라우드 분석")
                
                # 불용어 관리 UI 표시
                render_stopwords_ui()
                
                # 섹션 구분을 위한 간격과 구분선
                st.markdown("---")
                st.markdown("<br>", unsafe_allow_html=True)
                st.subheader("📊 워드클라우드 분석 결과")
                st.markdown("<br>", unsafe_allow_html=True)
                
                with st.spinner("워드클라우드 생성 중..."):
                    word_count, top_words = generate_wordcloud_data(review_df, 'review_content')
                    
                    # 워드클라우드 생성
                    if word_count:
                        wc = create_wordcloud(word_count)
                        
                        # 워드클라우드와 상위 20개 단어를 좌우로 배치
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            # 워드클라우드 제목 추가 (중앙 정렬)
                            st.markdown("<h3 style='text-align: center;'>워드클라우드</h3>", unsafe_allow_html=True)
                            
                            # 워드클라우드 표시
                            fig1, ax = plt.subplots(figsize=(8, 8))
                            ax.imshow(wc, interpolation='bilinear')
                            ax.axis('off')
                            plt.tight_layout(pad=0)
                            st.pyplot(fig1, use_container_width=True)
                            plt.close(fig1)  # 메모리 정리
                        
                        with col2:
                            # 상위 20개 단어 표시 (중앙 정렬)
                            st.markdown("<h3 style='text-align: center;'>상위 20개 단어</h3>", unsafe_allow_html=True)
                            
                            # 상위 단어 막대 그래프
                            top_words_df = pd.DataFrame({
                                '단어': list(top_words.keys()),
                                '언급 횟수': list(top_words.values())
                            })
                            
                            # 리뷰수 기준 내림차순 정렬 (높은 수가 위쪽에)
                            top_words_df = top_words_df.sort_values('언급 횟수', ascending=True)
                            
                            # 상위 20개 단어 차트 생성
                            fig2, ax = plt.subplots(figsize=(8, 8))
                            bars = ax.barh(top_words_df['단어'], top_words_df['언급 횟수'], color='steelblue')
                            
                            # 리뷰 수 표시
                            for i, bar in enumerate(bars):
                                width = bar.get_width()
                                ax.text(width + width*0.02, bar.get_y() + bar.get_height()/2, 
                                        f'{int(width):,}', 
                                        va='center', fontsize=10)
                        
                            # x축 범위 조정 (여백 줄이기)
                            if len(top_words) > 0:
                                max_count = max(top_words.values())
                                ax.set_xlim(0, max_count * 1.15)  # 텍스트 위한 여유 공간
                        
                            # y축 레이블 폰트 크기 조정
                            ax.tick_params(axis='y', labelsize=10)
                            ax.tick_params(axis='x', labelsize=10)
                        
                            # 한글 폰트 적용
                            set_korean_font(ax)
                        
                            # 그래프 제목 및 레이아웃 조정
                            ax.set_title('')
                            plt.tight_layout(pad=0)
                            st.pyplot(fig2, use_container_width=True)
                            plt.close(fig2)  # 메모리 정리
                    else:
                        st.warning("리뷰 분석을 위해 리뷰 파일을 업로드해주세요.")
                
        elif st.session_state.analysis_option == "리뷰 분석 - 감정분석":
            if file_type == "review":
                st.header("리뷰 감정분석")
                
                with st.spinner("감정 분석 중..."):
                    # 감정 분석 수행
                    df_sentiment, sentiment_counts = simple_sentiment_analysis(review_df, 'review_content')
                    
                    # 감정 분석 결과 표시
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # 감정별 리뷰 수 막대 그래프
                        fig, ax = plt.subplots(figsize=(6, 4))
                        
                        # 감정별 색상 매핑
                        emotion_colors = {'긍정': '#28a745', '중립': '#ffa500', '부정': '#dc3545'}
                        colors = [emotion_colors[emotion] for emotion in sentiment_counts['감정']]
                        
                        ax.bar(sentiment_counts['감정'], sentiment_counts['리뷰 수'], color=colors)
                        ax.set_title('감정별 리뷰 수', pad=20)
                        ax.set_ylabel('리뷰 수')
                        for i, v in enumerate(sentiment_counts['리뷰 수']):
                            ax.text(i, v + max(sentiment_counts['리뷰 수']) * 0.01, str(v), ha='center', va='bottom')
                        
                        # y축 범위 조정 (위쪽 여백 확보)
                        max_val = max(sentiment_counts['리뷰 수'])
                        ax.set_ylim(0, max_val * 1.15)
                        
                        # 한글 폰트 적용
                        set_korean_font(ax)
                        
                        st.pyplot(fig)
                    
                    with col2:
                        # 감정 비율 파이 차트
                        fig, ax = plt.subplots(figsize=(6, 4))
                        
                        # 감정별 색상 매핑
                        emotion_colors = {'긍정': '#28a745', '중립': '#ffa500', '부정': '#dc3545'}
                        colors = [emotion_colors[emotion] for emotion in sentiment_counts['감정']]
                        
                        ax.pie(sentiment_counts['리뷰 수'], labels=sentiment_counts['감정'], 
                               autopct='%1.1f%%', colors=colors, startangle=90)
                        ax.set_title('감정 분포 비율', pad=20)
                        ax.axis('equal')
                        
                        # 한글 폰트 적용
                        set_korean_font(ax)
                        
                        st.pyplot(fig)
                    
                    # 섹션 구분
                    st.markdown("---")
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # 감정별 리뷰 분석
                    st.subheader("감정별 리뷰 카테고리 분석")
                    
                    # 탭 폰트 크기 강제 적용
                    st.markdown("""
                    <style>
                    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
                        font-size: 24px !important;
                        font-weight: 600 !important;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    # 탭 생성  
                    tab1, tab2, tab3 = st.tabs(["긍정 리뷰", "중립 리뷰", "부정 리뷰"])
                    
                    with tab1:
                        # 긍정 리뷰 카테고리 분석
                        st.markdown("### 📊 긍정 리뷰 카테고리 분석")
                        with st.spinner("긍정 리뷰 카테고리 분석 중..."):
                            positive_category_analysis = analyze_positive_review_categories(df_sentiment, 'review_content')
                            
                            if not positive_category_analysis.empty:
                                st.dataframe(positive_category_analysis, use_container_width=True, hide_index=True)
                                
                                st.markdown("<br>", unsafe_allow_html=True)
                                
                                # 카테고리별 리뷰 수 시각화
                                if len(positive_category_analysis) > 0:
                                    fig, ax = plt.subplots(figsize=(8, 4))
                                    ax.bar(positive_category_analysis['카테고리'], positive_category_analysis['리뷰 수'], color='#28a745')
                                    ax.set_title('긍정 리뷰 카테고리별 언급 빈도')
                                    ax.set_ylabel('리뷰 수')
                                    ax.tick_params(axis='x', rotation=45)
                                    
                                    # 한글 폰트 적용
                                    set_korean_font(ax)
                                    
                                    plt.tight_layout()
                                    st.pyplot(fig)
                            else:
                                st.info("긍정 리뷰에서 분석 가능한 카테고리를 찾을 수 없습니다.")
                    
                    with tab2:
                        # 중립 리뷰 카테고리 분석
                        st.markdown("### 📊 중립 리뷰 카테고리 분석")
                        with st.spinner("중립 리뷰 카테고리 분석 중..."):
                            neutral_category_analysis = analyze_neutral_review_categories(df_sentiment, 'review_content')
                            
                            if not neutral_category_analysis.empty:
                                st.dataframe(neutral_category_analysis, use_container_width=True, hide_index=True)
                                
                                st.markdown("<br>", unsafe_allow_html=True)
                                
                                # 카테고리별 리뷰 수 시각화
                                if len(neutral_category_analysis) > 0:
                                    fig, ax = plt.subplots(figsize=(8, 4))
                                    
                                    # 막대 너비 설정 (카테고리 수에 따라 조정)
                                    bar_width = max(0.3, min(0.6, 2.0 / len(neutral_category_analysis)))
                                    
                                    bars = ax.bar(range(len(neutral_category_analysis)), 
                                                neutral_category_analysis['리뷰 수'], 
                                                width=bar_width, 
                                                color='#ffa500')
                                    
                                    # 막대 위에 숫자 표시
                                    for i, v in enumerate(neutral_category_analysis['리뷰 수']):
                                        ax.text(i, v + max(neutral_category_analysis['리뷰 수']) * 0.02, 
                                               str(v), ha='center', va='bottom')
                                    
                                    # y축 범위 조정 (위쪽 여백 확보)
                                    max_val = max(neutral_category_analysis['리뷰 수'])
                                    ax.set_ylim(0, max_val * 1.15)
                                    
                                    # x축 설정
                                    ax.set_xticks(range(len(neutral_category_analysis)))
                                    ax.set_xticklabels(neutral_category_analysis['카테고리'], rotation=45)
                                    
                                    ax.set_title('중립 리뷰 카테고리별 언급 빈도')
                                    ax.set_ylabel('리뷰 수')
                                    
                                    # 한글 폰트 적용
                                    set_korean_font(ax)
                                    
                                    plt.tight_layout()
                                    st.pyplot(fig)
                            else:
                                st.info("중립 리뷰에서 분석 가능한 카테고리를 찾을 수 없습니다.")
                    
                    with tab3:
                        # 부정 리뷰 카테고리 분석
                        st.markdown("### 📊 부정 리뷰 카테고리 분석")
                        with st.spinner("부정 리뷰 카테고리 분석 중..."):
                            negative_category_analysis = analyze_negative_review_categories(df_sentiment, 'review_content')
                            
                            if not negative_category_analysis.empty:
                                st.dataframe(negative_category_analysis, use_container_width=True, hide_index=True)
                                
                                st.markdown("<br>", unsafe_allow_html=True)
                                
                                # 카테고리별 리뷰 수 시각화
                                if len(negative_category_analysis) > 0:
                                    fig, ax = plt.subplots(figsize=(8, 4))
                                    
                                    # 막대 너비 설정 (카테고리 수에 따라 조정)
                                    bar_width = max(0.3, min(0.6, 2.0 / len(negative_category_analysis)))
                                    
                                    bars = ax.bar(range(len(negative_category_analysis)), 
                                                negative_category_analysis['리뷰 수'], 
                                                width=bar_width, 
                                                color='#dc3545')
                                    
                                    # 막대 위에 숫자 표시
                                    for i, v in enumerate(negative_category_analysis['리뷰 수']):
                                        ax.text(i, v + max(negative_category_analysis['리뷰 수']) * 0.02, 
                                               str(v), ha='center', va='bottom')
                                    
                                    # y축 범위 조정 (위쪽 여백 확보)
                                    max_val = max(negative_category_analysis['리뷰 수'])
                                    ax.set_ylim(0, max_val * 1.15)
                                    
                                    # x축 설정
                                    ax.set_xticks(range(len(negative_category_analysis)))
                                    ax.set_xticklabels(negative_category_analysis['카테고리'], rotation=45)
                                    
                                    ax.set_title('부정 리뷰 카테고리별 언급 빈도')
                                    ax.set_ylabel('리뷰 수')
                                    
                                    # 한글 폰트 적용
                                    set_korean_font(ax)
                                    
                                    plt.tight_layout()
                                    st.pyplot(fig)
                            else:
                                st.info("부정 리뷰에서 분석 가능한 카테고리를 찾을 수 없습니다.")
            else:
                st.warning("리뷰 분석을 위해 리뷰 파일을 업로드해주세요.")
            
        elif st.session_state.analysis_option == "옵션 분석":
            if file_type == "option":
                st.header("옵션 분석")
                
                with st.spinner("옵션 분석 중..."):
                    # 옵션 분석 수행
                    top_options = analyze_options(option_df, 'option_info', 'count')
                    
                    # 상위 10개 옵션 표시
                    st.subheader("상위 10개 옵션")
                    
                    # 표를 적절한 크기로 표시 (떨림 방지)
                    st.markdown("""
                    <style>
                    div[data-testid="stDataFrame"] {
                        width: 800px !important;
                        max-width: 800px !important;
                        overflow: visible !important;
                        margin-left: 50px !important;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    st.dataframe(top_options)
                    
                    # 간격 추가
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    
                    # 상위 10개 옵션 막대 그래프
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # 막대 그래프 생성 (인덱스를 X축 위치로 사용)
                    x_positions = range(len(top_options))
                    bars = ax.bar(x_positions, top_options['count'], color='steelblue')
                    
                    # X축 레이블 설정 (옵션명)
                    ax.set_xticks(x_positions)
                    ax.set_xticklabels(top_options['option_info'], rotation=45, ha='right')
                    
                    # 막대 위에 판매량 표시
                    for i, v in enumerate(top_options['count']):
                        ax.text(i, v + max(top_options['count']) * 0.01, 
                               f'{v:,}', ha='center', va='bottom')
                    
                    # Y축 범위 조정 (위쪽 여백 확보)
                    max_val = max(top_options['count'])
                    ax.set_ylim(0, max_val * 1.15)
                    
                    ax.set_title('상위 10개 옵션 판매량')
                    ax.set_ylabel('판매량')
                    
                    # 한글 폰트 적용
                    set_korean_font(ax)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
            else:
                st.warning("옵션 분석을 위해 옵션 비율 파일을 업로드해주세요.")
        
        elif st.session_state.analysis_option == "스토어 전체 판매현황":
            if sales_df is not None:
                st.header("🏪 스토어 전체 판매현황 분석")
                
                # 사용 가능한 기간 가져오기
                available_periods = get_sales_periods(sales_df)
                
                if len(available_periods) == 0:
                    st.error("매출 데이터를 찾을 수 없습니다.")
                else:
                    # 기간 선택 필터
                    st.subheader("📅 분석 기간 선택")
                    
                    # selectbox 커서 스타일 추가
                    st.markdown("""
                    <style>
                    div[data-baseweb="select"] {
                        cursor: pointer !important;
                    }
                    div[data-baseweb="select"] > div {
                        cursor: pointer !important;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    selected_period = st.selectbox(
                        "매출 분석 기간을 선택하세요:",
                        available_periods,
                        index=len(available_periods) - 1 if '1년' in available_periods else 0
                    )
                    
                    # 매출 요약 통계
                    st.subheader("📊 매출 요약 통계")
                    summary_stats = get_sales_summary_stats(sales_df, selected_period)
                    
                    if summary_stats:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("총 매출", f"{summary_stats['총매출']:,}원")
                        with col2:
                            st.metric("평균 매출", f"{summary_stats['평균매출']:,}원")
                        with col3:
                            st.metric("상품 수", f"{summary_stats['상품수']:,}개")
                        with col4:
                            st.metric("최대 매출", f"{summary_stats['최대매출']:,}원")
                    
                    # 분석 탭 생성
                    # 탭 폰트 크기 강제 적용
                    st.markdown("""
                    <style>
                    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
                        font-size: 24px !important;
                        font-weight: 600 !important;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    tab1, tab2, tab3, tab4 = st.tabs(["매출 랭킹", "매출 효율성", "가격대별 분석", "리뷰-매출 상관관계"])
                    
                    with tab1:
                        st.subheader(f"🏆 {selected_period} 매출 상위 10개 상품")
                        top_products = analyze_top_products_by_period(sales_df, selected_period, 10)
                        
                        if not top_products.empty:
                            st.dataframe(top_products, use_container_width=True, hide_index=True)
                            
                            # 표와 그래프 사이 간격 추가
                            st.markdown("<br><br>", unsafe_allow_html=True)
                            
                            # 매출 랭킹 시각화
                            fig, ax = plt.subplots(figsize=(12, 6))
                            
                            # 막대 그래프 생성
                            bars = ax.bar(range(len(top_products)), 
                                         top_products[f'{selected_period} 매출'], 
                                         color='steelblue')
                            
                            # 상품명을 x축 레이블로 설정 (회전)
                            ax.set_xticks(range(len(top_products)))
                            ax.set_xticklabels([name[:15] + '...' if len(name) > 15 else name 
                                               for name in top_products['상품명']], 
                                              rotation=45, ha='right')
                            
                            # 막대 위에 매출 표시
                            for i, v in enumerate(top_products[f'{selected_period} 매출']):
                                ax.text(i, v + max(top_products[f'{selected_period} 매출']) * 0.01, 
                                       f'{v:,.0f}', ha='center', va='bottom', fontsize=8)
                            
                            ax.set_ylabel('매출 (원)')
                            ax.set_title(f'{selected_period} 매출 상위 10개 상품')
                            
                            # 한글 폰트 적용
                            set_korean_font(ax)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                        else:
                            st.info("매출 데이터가 없습니다.")
                    
                    with tab2:
                        st.subheader(f"⚡ {selected_period} 매출 효율성 분석")
                        efficiency_data = analyze_sales_efficiency(sales_df, selected_period)
                        
                        if not efficiency_data.empty:
                            st.dataframe(efficiency_data, use_container_width=True, hide_index=True)
                            
                            # 효율성 시각화
                            fig, ax = plt.subplots(figsize=(12, 6))
                            
                            bars = ax.bar(range(len(efficiency_data)), 
                                         efficiency_data['매출효율성'], 
                                         color='orange')
                            
                            ax.set_xticks(range(len(efficiency_data)))
                            ax.set_xticklabels([name[:15] + '...' if len(name) > 15 else name 
                                               for name in efficiency_data['상품명']], 
                                              rotation=45, ha='right')
                            
                            # 막대 위에 효율성 표시
                            for i, v in enumerate(efficiency_data['매출효율성']):
                                ax.text(i, v + max(efficiency_data['매출효율성']) * 0.01, 
                                       f'{v:.1f}', ha='center', va='bottom', fontsize=8)
                            
                            ax.set_ylabel('매출효율성 (매출/가격)')
                            ax.set_title(f'{selected_period} 가격 대비 매출 효율성')
                            
                            # 한글 폰트 적용
                            set_korean_font(ax)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                        else:
                            st.info("매출 효율성 분석을 위한 데이터가 부족합니다.")
                        
                        with tab3:
                            st.subheader(f"💰 가격대별 {selected_period} 매출 분석")
                            price_segments = analyze_price_segments(sales_df, selected_period)
                            
                            if not price_segments.empty:
                                st.dataframe(price_segments, use_container_width=True, hide_index=True)
                                
                                # 가격대별 분석 시각화
                                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                                
                                # 가격대별 상품수
                                ax1.bar(price_segments['가격대'], price_segments['상품수'], color='lightblue')
                                ax1.set_title('가격대별 상품 수')
                                ax1.set_ylabel('상품 수')
                                ax1.tick_params(axis='x', rotation=45)
                                
                                # 가격대별 평균매출
                                ax2.bar(price_segments['가격대'], price_segments['평균매출'], color='lightgreen')
                                ax2.set_title('가격대별 평균 매출')
                                ax2.set_ylabel('평균 매출 (원)')
                                ax2.tick_params(axis='x', rotation=45)
                                
                                # 한글 폰트 적용
                                set_korean_font(ax1)
                                set_korean_font(ax2)
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                            else:
                                st.info("가격대별 분석을 위한 데이터가 부족합니다.")
                        
                        with tab4:
                            st.subheader(f"⭐ 리뷰 점수와 {selected_period} 매출 상관관계")
                            correlation, review_analysis = analyze_review_sales_correlation(sales_df, selected_period)
                            
                            if correlation is not None:
                                st.info(f"**상관계수: {correlation:.3f}**")
                                
                                if not review_analysis.empty:
                                    st.dataframe(review_analysis, use_container_width=True, hide_index=True)
                                    
                                    # 리뷰 점수별 평균 매출 시각화
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    
                                    bars = ax.bar(review_analysis['리뷰점수구간'], 
                                                 review_analysis['평균매출'], 
                                                 color='gold')
                                    
                                    # 막대 위에 값 표시
                                    for i, v in enumerate(review_analysis['평균매출']):
                                        ax.text(i, v + max(review_analysis['평균매출']) * 0.01, 
                                               f'{v:,.0f}', ha='center', va='bottom')
                                    
                                    ax.set_title('리뷰 점수 구간별 평균 매출')
                                    ax.set_ylabel('평균 매출 (원)')
                                    
                                    # 한글 폰트 적용
                                    set_korean_font(ax)
                                    
                                    plt.tight_layout()
                                    st.pyplot(fig)
                            else:
                                st.info("리뷰-매출 상관관계 분석을 위한 데이터가 부족합니다.")
                else:
                    st.warning("스토어 전체 판매현황 분석을 위해 판매현황 파일을 업로드해주세요.")
            
    except Exception as e:
        st.error(f"데이터 처리 중 오류가 발생했습니다: {e}")