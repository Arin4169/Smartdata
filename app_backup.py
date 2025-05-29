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
    get_sales_summary_stats,
    analyze_review_efficiency,
    analyze_hidden_gems,
    analyze_underperforming_products,
    analyze_review_needed_products,
    analyze_value_products
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
    
    /* 홈페이지 카드 버튼 스타일링 - 더 구체적인 선택자 사용 */
    .stButton > button {
        padding: 1rem !important;
        border-radius: 0.5rem !important;
        background: white !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
        height: 140px !important;
        margin-bottom: 1.5rem !important;
        max-width: 500px !important;
        margin-left: auto !important;
        margin-right: auto !important;
        cursor: pointer !important;
        transition: transform 0.2s, box-shadow 0.2s !important;
        border: 1px solid #e5e7eb !important;
        text-align: left !important;
        white-space: pre-line !important;
        font-size: 0.9rem !important;
        line-height: 1.4 !important;
        color: #4B5563 !important;
        width: 100% !important;
        justify-content: flex-start !important;
        align-items: flex-start !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 12px -1px rgba(0, 0, 0, 0.15) !important;
        border-color: #d1d5db !important;
    }
    
    .stButton > button:active {
        transform: translateY(0px) !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
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
    
    /* 불용어 관리 섹션의 버튼들만 작게 만들기 */
    .stButton button[title*="삭제"] {
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
    
    /* 불용어 추가/초기화 버튼 */
    form[data-testid="stForm"] .stButton > button {
        font-size: 0.8rem !important;
        padding: 0.25rem 0.5rem !important;
        height: auto !important;
        min-height: auto !important;
        width: 100% !important;
        max-width: none !important;
        margin: 0 !important;
        border-radius: 0.25rem !important;
        box-shadow: none !important;
        transform: none !important;
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
        if st.button("📊 리뷰 워드클라우드 분석\n\n• 고객 리뷰에서 자주 등장하는 키워드를 시각화\n• 불용어 관리로 분석 정확도 향상\n• 직관적인 워드클라우드와 Top 20 키워드 차트", 
                     key="card1", use_container_width=True):
            st.session_state.analysis_option = "리뷰 분석 - 워드클라우드"
            st.rerun()
    
    with col2:
        # 리뷰 감정 분석 카드
        if st.button("😊 리뷰 감정 분석\n\n• 고객 리뷰의 감정별 세부 카테고리 분석\n• 감정 분포 시각화\n• 고객 만족도 트렌드 파악", 
                     key="card2", use_container_width=True):
            st.session_state.analysis_option = "리뷰 분석 - 감정분석"
            st.rerun()
    
    # 두 번째 행
    col3, col4 = st.columns([1, 1], gap="small")
    
    with col3:
        # 옵션 분석 카드
        if st.button("🎯 옵션 분석\n\n• 상품 옵션별 판매 수량 분석\n• 인기 옵션 Top 10 시각화\n• 재고 관리 및 마케팅 전략 수립 지원", 
                     key="card3", use_container_width=True):
            st.session_state.analysis_option = "옵션 분석"
            st.rerun()
    
    with col4:
        # 스토어 전체 판매 현황 카드
        if st.button("📈 스토어 전체 판매 현황\n\n• 기간별 매출 랭킹 및 가격대비 매출지수 분석\n• 매출 및 주문 데이터 시각화\n• 동적 가격대별 분석 및 리뷰-매출 인사이트", 
                     key="card4", use_container_width=True):
            st.session_state.analysis_option = "스토어 전체 판매현황"
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 시작하기 섹션
    st.markdown("""
    <div class="start-section">
        <h2 style="color: #1E3A8A;">시작하기</h2>
        <p style="color: #6B7280;">
            👆 위의 카드를 클릭하여 원하는 분석을 시작하거나<br>
            👈 왼쪽 사이드바에서 파일을 업로드하고 분석 메뉴를 선택하세요.<br>
            파일 없이도 샘플 데이터로 각 분석 기능을 체험할 수 있습니다.
        </p>
    </div>
    """, unsafe_allow_html=True)

elif uploaded_file is None and st.session_state.analysis_option not in ["홈"]:
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
            st.header("📊 리뷰 워드클라우드 분석")
            
            # 분석 가이드 추가
            with st.expander("📖 워드클라우드 분석 가이드", expanded=False):
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
            st.header("😊 리뷰 감정분석")
            
            # 분석 가이드 추가
            with st.expander("📖 감정분석 가이드", expanded=False):
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
            st.header("🎯 옵션 분석")
            
            # 분석 가이드 추가
            with st.expander("📖 옵션 분석 가이드", expanded=False):
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
            if sales_df is not None:
                st.header("🏪 스토어 전체 판매현황 분석")
                
                # 분석 가이드 추가
                with st.expander("📖 스토어 전체 판매현황 분석 가이드", expanded=False):
                    st.markdown("""
                    ### 🏪 스토어 전체 판매현황 분석이란?
                    스토어의 **전체 상품 판매 데이터**를 다각도로 분석하여 매출 성과와 트렌드를 파악하는 종합 분석입니다.
                    
                    ### 📊 제공하는 4가지 분석
                    
                    #### 1. 🏆 매출 랭킹
                    - 기간별 매출 상위 10개 상품 순위
                    
                    #### 2. ⚡ 가격대비 매출지수 분석
                    **가격대비 매출지수 = 해당 기간 매출 ÷ 기본판매가격**
                    - 상품 가격 대비 얼마나 효율적으로 매출을 올렸는지 측정
                    - 높은 지수 = 가격 대비 매출 효율성이 좋음
                    
                    #### 3. 💰 가격대별 분석
                    - **동적 가격대 설정**: 업로드된 데이터의 가격 분포를 분석하여 사분위수 기반으로 4개 구간 자동 설정
                    - 저가(하위 25%), 중저가(25%-50%), 중가(50%-75%), 고가(상위 25%)
                    - 각 가격대별 상품 수와 평균 매출 분석
                    
                    #### 4. 💡 리뷰-매출 인사이트
                    **실용적인 리뷰-매출 분석:**
                    - **리뷰 효율성**: 리뷰 1건당 매출이 높은 상품
                    - **숨겨진 보석**: 매출은 낮지만 리뷰 점수가 높은 상품 (리뷰 점수 4.5 이상 & 매출 하위 50%에 속하는 상품)
                    - **잠재력 미달**: 리뷰는 좋은데 매출이 예상보다 낮은 상품 (리뷰 점수 4.0 이상 & 매출 상위 75%에 못미치는 상품)
                    - **리뷰 확보 필요**: 매출은 높은데 리뷰가 적은 상품 (매출 상위 50% & 리뷰수 하위 50%에 속하는 상품)
                    - **가성비 인증**: 저렴한 가격 + 높은 리뷰 점수 상품 (가격 하위 50% & 리뷰 점수 4.0 이상)
                    """)
                
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
                    
                    tab1, tab2, tab3, tab4 = st.tabs(["매출 랭킹", "매출 효율성", "가격대별 분석", "리뷰-매출 인사이트"])
                    
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
                                         efficiency_data['가격대비매출지수'], 
                                         color='orange')
                            
                            ax.set_xticks(range(len(efficiency_data)))
                            ax.set_xticklabels([name[:15] + '...' if len(name) > 15 else name 
                                               for name in efficiency_data['상품명']], 
                                              rotation=45, ha='right')
                            
                            # 막대 위에 효율성 표시
                            for i, v in enumerate(efficiency_data['가격대비매출지수']):
                                ax.text(i, v + max(efficiency_data['가격대비매출지수']) * 0.01, 
                                       f'{v:.1f}', ha='center', va='bottom', fontsize=8)
                            
                            ax.set_ylabel('가격대비매출지수 (매출/가격)')
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
                        st.subheader(f"💡 {selected_period} 리뷰-매출 인사이트")
                        
                        # 리뷰-매출 인사이트 탭 생성
                        insight_tab1, insight_tab2, insight_tab3, insight_tab4, insight_tab5 = st.tabs([
                            "⚡ 리뷰 효율성", "💎 숨겨진 보석", "📈 잠재력 미달", "🔥 리뷰 확보 필요", "💰 가성비 인증"
                        ])
                        
                        with insight_tab1:
                            st.markdown("### ⚡ 리뷰 효율성 랭킹")
                            st.info("💡 **리뷰 1건당 매출이 가장 높은 상품은?** - 리뷰 대비 매출 효율성이 높은 상품을 찾아보세요!")
                            
                            efficiency_data = analyze_review_efficiency(sales_df, selected_period)
                            
                            if not efficiency_data.empty:
                                st.dataframe(efficiency_data, use_container_width=True, hide_index=True)
                                
                                # 리뷰 효율성 시각화
                                fig, ax = plt.subplots(figsize=(12, 6))
                                
                                bars = ax.bar(range(len(efficiency_data)), 
                                             efficiency_data['리뷰1건당매출'], 
                                             color='lightcoral')
                                
                                ax.set_xticks(range(len(efficiency_data)))
                                ax.set_xticklabels([name[:15] + '...' if len(name) > 15 else name 
                                                   for name in efficiency_data['상품명']], 
                                                  rotation=45, ha='right')
                                
                                # 막대 위에 값 표시
                                for i, v in enumerate(efficiency_data['리뷰1건당매출']):
                                    ax.text(i, v + max(efficiency_data['리뷰1건당매출']) * 0.01, 
                                           f'{v:,.0f}', ha='center', va='bottom', fontsize=8)
                                
                                ax.set_title('리뷰 1건당 매출 상위 10개 상품')
                                ax.set_ylabel('리뷰 1건당 매출 (원)')
                                
                                # 한글 폰트 적용
                                set_korean_font(ax)
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                            else:
                                st.info("리뷰 효율성 분석을 위한 데이터가 부족합니다.")
                        
                        with insight_tab2:
                            st.markdown("### 💎 숨겨진 보석 상품")
                            st.info("💡 **매출은 낮은데 리뷰 점수가 높은 상품은?** - 마케팅 강화로 매출을 늘릴 수 있는 상품을 찾아보세요!")
                            
                            hidden_gems_data = analyze_hidden_gems(sales_df, selected_period)
                            
                            if not hidden_gems_data.empty:
                                st.dataframe(hidden_gems_data, use_container_width=True, hide_index=True)
                                
                                # 숨겨진 보석 시각화
                                fig, ax = plt.subplots(figsize=(12, 6))
                                
                                bars = ax.bar(range(len(hidden_gems_data)), 
                                             hidden_gems_data['리뷰점수'], 
                                             color='mediumseagreen')
                                
                                ax.set_xticks(range(len(hidden_gems_data)))
                                ax.set_xticklabels([name[:15] + '...' if len(name) > 15 else name 
                                                   for name in hidden_gems_data['상품명']], 
                                                  rotation=45, ha='right')
                                
                                # 막대 위에 값 표시
                                for i, v in enumerate(hidden_gems_data['리뷰점수']):
                                    ax.text(i, v + 0.05, f'{v:.1f}', ha='center', va='bottom', fontsize=8)
                                
                                ax.set_title('숨겨진 보석 상품 (높은 리뷰점수 + 낮은 매출)')
                                ax.set_ylabel('리뷰 점수')
                                ax.set_ylim(0, 5.5)
                                
                                # 한글 폰트 적용
                                set_korean_font(ax)
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                            else:
                                st.info("숨겨진 보석 상품 분석을 위한 데이터가 부족합니다.")
                        
                        with insight_tab3:
                            st.markdown("### 📈 잠재력 미달 상품")
                            st.info("💡 **리뷰는 좋은데 매출이 예상보다 낮은 상품은?** - 프로모션이나 노출 개선이 필요한 상품을 찾아보세요!")
                            
                            underperforming_data = analyze_underperforming_products(sales_df, selected_period)
                            
                            if not underperforming_data.empty:
                                st.dataframe(underperforming_data, use_container_width=True, hide_index=True)
                                
                                # 잠재력 미달 시각화
                                fig, ax = plt.subplots(figsize=(12, 6))
                                
                                bars = ax.bar(range(len(underperforming_data)), 
                                             underperforming_data[f'{selected_period} 매출'], 
                                             color='orange')
                                
                                ax.set_xticks(range(len(underperforming_data)))
                                ax.set_xticklabels([name[:15] + '...' if len(name) > 15 else name 
                                                   for name in underperforming_data['상품명']], 
                                                  rotation=45, ha='right')
                                
                                # 막대 위에 값 표시
                                for i, v in enumerate(underperforming_data[f'{selected_period} 매출']):
                                    ax.text(i, v + max(underperforming_data[f'{selected_period} 매출']) * 0.01, 
                                           f'{v:,.0f}', ha='center', va='bottom', fontsize=8)
                                
                                ax.set_title('잠재력 미달 상품 (좋은 리뷰 + 저조한 매출)')
                                ax.set_ylabel('매출 (원)')
                                
                                # 한글 폰트 적용
                                set_korean_font(ax)
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                            else:
                                st.info("잠재력 미달 상품 분석을 위한 데이터가 부족합니다.")
                        
                        with insight_tab4:
                            st.markdown("### 🔥 리뷰 확보 필요 상품")
                            st.info("💡 **매출은 높은데 리뷰가 적은 상품은?** - 리뷰 수집 캠페인이 필요한 상품을 찾아보세요!")
                            
                            review_needed_data = analyze_review_needed_products(sales_df, selected_period)
                            
                            if not review_needed_data.empty:
                                st.dataframe(review_needed_data, use_container_width=True, hide_index=True)
                                
                                # 리뷰 확보 필요 시각화
                                fig, ax = plt.subplots(figsize=(12, 6))
                                
                                bars = ax.bar(range(len(review_needed_data)), 
                                             review_needed_data['매출대비리뷰부족도'], 
                                             color='tomato')
                                
                                ax.set_xticks(range(len(review_needed_data)))
                                ax.set_xticklabels([name[:15] + '...' if len(name) > 15 else name 
                                                   for name in review_needed_data['상품명']], 
                                                  rotation=45, ha='right')
                                
                                # 막대 위에 값 표시
                                for i, v in enumerate(review_needed_data['매출대비리뷰부족도']):
                                    ax.text(i, v + max(review_needed_data['매출대비리뷰부족도']) * 0.01, 
                                           f'{v:,.0f}', ha='center', va='bottom', fontsize=8)
                                
                                ax.set_title('리뷰 확보 필요 상품 (높은 매출 + 적은 리뷰)')
                                ax.set_ylabel('매출대비 리뷰 부족도')
                                
                                # 한글 폰트 적용
                                set_korean_font(ax)
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                            else:
                                st.info("리뷰 확보 필요 상품 분석을 위한 데이터가 부족합니다.")
                        
                        with insight_tab5:
                            st.markdown("### 💰 가성비 인증 상품")
                            st.info("💡 **저렴한 가격 + 높은 리뷰 점수 상품은?** - 가성비 마케팅에 활용할 수 있는 상품을 찾아보세요!")
                            
                            value_products_data = analyze_value_products(sales_df, selected_period)
                            
                            if not value_products_data.empty:
                                st.dataframe(value_products_data, use_container_width=True, hide_index=True)
                                
                                # 가성비 인증 시각화
                                fig, ax = plt.subplots(figsize=(12, 6))
                                
                                bars = ax.bar(range(len(value_products_data)), 
                                             value_products_data['가성비점수'], 
                                             color='gold')
                                
                                ax.set_xticks(range(len(value_products_data)))
                                ax.set_xticklabels([name[:15] + '...' if len(name) > 15 else name 
                                                   for name in value_products_data['상품명']], 
                                                  rotation=45, ha='right')
                                
                                # 막대 위에 값 표시
                                for i, v in enumerate(value_products_data['가성비점수']):
                                    ax.text(i, v + max(value_products_data['가성비점수']) * 0.01, 
                                           f'{v:.1f}', ha='center', va='bottom', fontsize=8)
                                
                                ax.set_title('가성비 인증 상품 (저렴한 가격 + 높은 리뷰점수)')
                                ax.set_ylabel('가성비 점수')
                                
                                # 한글 폰트 적용
                                set_korean_font(ax)
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                            else:
                                st.info("가성비 인증 상품 분석을 위한 데이터가 부족합니다.")
            else:
                st.warning("스토어 전체 판매현황 분석을 위해 판매현황 파일을 업로드해주세요.")
            
    except Exception as e:
        st.error(f"데이터 처리 중 오류가 발생했습니다: {e}")