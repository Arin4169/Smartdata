import streamlit as st

# 페이지 기본 설정 (반드시 첫 번째 Streamlit 명령어여야 함)
st.set_page_config(
    page_title="스마트스토어 데이터 분석",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"  # 사이드바를 항상 펼쳐진 상태로 시작
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

# 한글 폰트 설정 함수를 캐시된 리소스로 생성
@st.cache_resource
def setup_korean_font():
    """한글 폰트를 설정하고 폰트 속성을 반환합니다."""
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
    return korean_font_prop

# 폰트 설정 실행 (캐시됨)
korean_font_prop = setup_korean_font()

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
        color: #7C3AED !important;
        font-size: 3.5rem;
        font-weight: 700;
    }
    .subtitle {
        text-align: center;
        color: #6B7280;
        margin-bottom: 1rem;
    }
    .brand-message {
        text-align: center;
        color: #4B5563;
        font-size: 1.0rem;
        font-weight: 400;
        margin: 1.5rem 0 2.5rem 0;
        padding: 0.8rem 1.5rem;
        background: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        max-width: 600px;
        margin-left: auto;
        margin-right: auto;
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

# 제목
st.markdown("<h1 class='main-title'>Smart Data Assistant</h1>", unsafe_allow_html=True)

# 라디오 버튼 값을 먼저 가져와야 브랜드 메시지 표시 로직이 정확함

# 함수: 불용어 관리 UI 생성
def render_stopwords_ui():
    """불용어 관리 UI를 표시합니다."""
   
    st.subheader("🔧 불용어 관리")
    
    # 현재 불용어 목록 가져오기
    current_stopwords = get_stopwords()
    
    # 기본 불용어와 추가 불용어 구분
    from utils import DEFAULT_STOPWORDS
    basic_words = [word for word in current_stopwords if word in DEFAULT_STOPWORDS]
    added_words = [word for word in current_stopwords if word not in DEFAULT_STOPWORDS]
    
    # 기본 불용어는 정렬, 추가 불용어는 입력 순서 유지
    sorted_basic = sorted(basic_words)
    display_order = sorted_basic + added_words
    current_text = ", ".join(display_order) if display_order else ""
    
    # 불용어 업데이트 콜백 함수
    def update_stopwords():
        if st.session_state.stopwords_input:
            new_stopwords = [word.strip() for word in st.session_state.stopwords_input.split(',') if word.strip()]
            st.session_state.stopwords = new_stopwords
    
    # 불용어 설정 (엔터키로 적용)
    custom_stopwords = st.text_input("불용어 (쉼표로 구분)", current_text, 
                                     key="stopwords_input", on_change=update_stopwords)
    
    # 불용어 추가 방법 안내
    st.markdown("💡 새로운 불용어를 추가하려면 기존 목록 뒤에 **', 새단어'** 를 입력하고 Enter를 누르세요.")

# 함수: 불용어 목록 저장 (세션 기반으로 수정)
def save_stopwords_list(stopwords_list):
    """불용어 목록을 세션에 저장합니다."""
    try:
        st.session_state.stopwords = stopwords_list
    except Exception as e:
        st.error(f"불용어 저장 중 오류가 발생했습니다: {e}")

# 함수: 파일 유형 자동 감지
def detect_file_type(df, filename=""):
    """업로드된 파일의 유형을 자동으로 감지합니다"""
    
    # 1. 파일명 기반 감지 (우선순위 최고)
    filename_lower = filename.lower()
    # 파일명에서 괄호와 숫자 제거 (예: "reviewcontents (4).xlsx" → "reviewcontents.xlsx")
    import re
    cleaned_filename = re.sub(r'\s*\(\d+\)', '', filename_lower)
    
    print(f"[DEBUG] 파일명: {filename} -> {filename_lower} -> 정리됨: {cleaned_filename}")
    
    if 'reviewcontent' in cleaned_filename or 'review' in cleaned_filename:
        print(f"[DEBUG] 파일명 기준으로 review 감지: {filename}")
        return "review"
    elif '옵션' in cleaned_filename or 'option' in cleaned_filename:
        print(f"[DEBUG] 파일명 기준으로 option 감지: {filename}")
        return "option"
    elif '판매현황' in cleaned_filename or '스토어' in cleaned_filename or 'sales' in cleaned_filename:
        print(f"[DEBUG] 파일명 기준으로 sales 감지: {filename}")
        return "sales"
    
    # 2. 컬럼명 기반 감지
    columns_lower = [col.lower() for col in df.columns]
    columns_str = ' '.join(columns_lower)
    print(f"[DEBUG] 컬럼명들(소문자): {columns_lower}")
    
    # 리뷰 파일 감지 - 더 구체적인 키워드 사용
    review_keywords = ['review_content', '리뷰내용', '리뷰', 'content', '내용', '후기', '평가', '댓글', 'review']
    matched_review = [kw for kw in review_keywords if kw in columns_str]
    if matched_review:
        print(f"[DEBUG] 컬럼명 기준으로 review 감지. 매칭된 키워드: {matched_review}")
        return "review"
    
    # 옵션 비율 파일 감지 (옵션 + 수량 모두 있어야 함)
    option_keywords = ['option', '옵션', 'option_info', '옵션정보', '옵션명']
    count_keywords = ['count', '수량', '판매량', '판매수량', '개수', 'quantity']
    
    matched_option = [kw for kw in option_keywords if kw in columns_str]
    matched_count = [kw for kw in count_keywords if kw in columns_str]
    
    has_option = len(matched_option) > 0
    has_count = len(matched_count) > 0
    
    print(f"[DEBUG] 옵션 키워드 매칭: {matched_option}, 수량 키워드 매칭: {matched_count}")
    
    if has_option and has_count:
        print(f"[DEBUG] 컬럼명 기준으로 option 감지")
        return "option"
    
    # 스토어 전체 판매현황 파일 감지
    sales_keywords = ['상품명', '매출', '판매건수', '기본판매가격', 'product', 'sales']
    matched_sales = [kw for kw in sales_keywords if kw in columns_str]
    if matched_sales:
        print(f"[DEBUG] 컬럼명 기준으로 sales 감지. 매칭된 키워드: {matched_sales}")
        return "sales"
    
    # 기본값은 sales로 간주
    print(f"[DEBUG] 기본값으로 sales 반환")
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
    uploaded_files = st.file_uploader("스마트스토어 데이터 파일", type=["xlsx", "csv"], accept_multiple_files=True, help="리뷰 분석, 옵션 비율, 판매 현황 등의 파일을 업로드하세요. (최대 3개 파일)")
    
    # 파일 타입 설명
    with st.expander("📁 파일 타입 설명"):
        st.markdown("""
        • <span style="font-size: 1.2em; font-weight: bold;">reviewcontents</span>: 리뷰 내용 컬럼을 포함한 파일  
        • **옵션비율**: 옵션 정보와 판매량/수량 컬럼을 포함한 파일  
        • **스토어전체판매현황**: 기타 판매 관련 파일  
        
        파일 유형은 자동으로 감지됩니다.
        """, unsafe_allow_html=True)
    
    st.header("분석 메뉴")
    

    
    # 초기값 설정 (한 번만)
    if "analysis_option" not in st.session_state:
        st.session_state.analysis_option = "홈"
    
    # 라디오 버튼
    options = ["홈", "데이터 분석 사용안내", "리뷰 분석 - 워드클라우드", "리뷰 분석 - 감정분석", "옵션 분석", "스토어 전체 판매현황"]
    current_option = st.session_state.get("analysis_option", "홈")
    
    # 안전한 index 계산
    try:
        current_index = options.index(current_option)
    except ValueError:
        current_index = 0  # 찾을 수 없으면 홈으로
    
    analysis_option = st.radio(
        "분석 유형 선택",
        options,
        index=current_index,
        label_visibility="collapsed"
    )
    
    # 데이터 저장 변수
    review_df = None
    option_df = None
    sales_df = None

# 업로드된 파일들 처리
if uploaded_files:
    try:
        for uploaded_file in uploaded_files:
            # 파일 읽기
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
            else:
                df = pd.read_excel(uploaded_file)
            
            # 파일 타입 감지 및 데이터 할당
            file_type = detect_file_type(df, uploaded_file.name)
            
            if file_type == "review":
                review_df = check_review_columns(df)
            elif file_type == "option":
                option_df = check_option_columns(df)
            elif file_type == "sales":
                sales_df = df
            
    except Exception as e:
        st.sidebar.error(f"파일 처리 중 오류가 발생했습니다: {e}")
        st.sidebar.write(f"오류 상세: {type(e).__name__}: {str(e)}")

# 브랜드 메시지 표시 로직 - 라디오 버튼 값 기준
if analysis_option != "홈":
    # 분석 화면: 추가 여백만 표시
    st.markdown("<div style='margin-bottom: 2rem;'></div>", unsafe_allow_html=True)
else:
    # 홈 화면: 브랜드 메시지 표시
    st.markdown("<div class='brand-message'><strong>Smart Data Assistant</strong>는 당신의 스마트스토어 데이터를 자동 분석해 핵심 인사이트를 도출해주는 서비스입니다.</div>", unsafe_allow_html=True)

# 메인 화면
if analysis_option == "홈":
    st.markdown('<div class="container">', unsafe_allow_html=True)
    
    # 첫 번째 행
    col1, col2 = st.columns([1, 1], gap="small")
    
    with col1:
        # 리뷰 워드클라우드 분석 카드
        if st.button("**📊 리뷰 워드클라우드 분석**\n\n• 고객 리뷰에서 자주 등장하는 키워드를 시각화\n• 불용어 관리로 분석 정확도 향상\n• 직관적인 워드클라우드와 Top 20 키워드 차트", 
                     key="card1", use_container_width=True):
            st.session_state.analysis_option = "리뷰 분석 - 워드클라우드"
            st.rerun()
    
    with col2:
        # 리뷰 감정 분석 카드
        sentiment_clicked = st.button("**😊 리뷰 감정 분석**\n\n• 고객 리뷰의 감정별 세부 카테고리 분석\n• 감정 분포 시각화\n• 고객 만족도 트렌드 파악", 
                                     key="sentiment_btn_v4", use_container_width=True)
        
        if sentiment_clicked:
            # 감정분석 전 완전한 상태 정리
            keys_to_clear = []
            for key in st.session_state.keys():
                # 감정분석 관련 키워드가 포함된 모든 키 제거
                if any(word in key.lower() for word in ['sentiment', 'emotion', 'analysis', 'category', 'positive', 'negative', 'neutral']):
                    keys_to_clear.append(key)
            
            # 키 삭제
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            
            # 캐시는 유지하고 session state만 정리 (대용량 데이터 재분석 성능 보장)
            # st.cache_data.clear()  # 의도적으로 주석 처리 - 빠른 재분석을 위해 캐시 유지
            
            # matplotlib 관련 정리
            try:
                import matplotlib.pyplot as plt
                plt.close('all')  # 모든 matplotlib figure 닫기
            except:
                pass
            
            st.session_state.analysis_option = "리뷰 분석 - 감정분석"
            st.rerun()
    
    # 두 번째 행
    col3, col4 = st.columns([1, 1], gap="small")
    
    with col3:
        # 옵션 분석 카드
        if st.button("**🎯 옵션 분석**\n\n• 상품 옵션별 판매 수량 분석\n• 인기 옵션 Top 10 시각화\n• 재고 관리 및 마케팅 전략 수립 지원", 
                     key="card3", use_container_width=True):
            st.session_state.analysis_option = "옵션 분석"
            st.rerun()
    
    with col4:
        # 스토어 전체 판매 현황 카드
        if st.button("**📈 스토어 전체 판매 현황**\n\n• 기간별 매출 랭킹 및 가격대비 매출지수 분석\n• 매출 및 주문 데이터 시각화\n• 동적 가격대별 분석 및 리뷰-매출 인사이트", 
                     key="card4", use_container_width=True):
            st.session_state.analysis_option = "스토어 전체 판매현황"
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 시작하기 섹션
    st.markdown("""
    <div class="start-section">
        <h2 style="color: #1E3A8A;">시작하기 전에...</h2>
        <p style="color: #6B7280;">
            💡 <strong>'데이터 분석 사용안내'를 먼저 확인해 주세요.</strong><br><br>
            👆 위의 카드를 클릭하여 원하는 분석을 시작하거나<br>
            👈 왼쪽 사이드바에서 파일을 업로드하고 분석 메뉴를 선택하세요.<br>
            📊 파일 없이도 샘플 데이터로 각 분석 기능을 체험할 수 있습니다.
        </p>
    </div>
    """, unsafe_allow_html=True)

elif analysis_option == "데이터 분석 사용안내":
    st.header("📖 데이터 분석 사용안내")
    
    # 서비스 사용 흐름
    st.subheader("🔄 서비스 사용 흐름")
    
    with st.expander("**1단계: Smart Data 확장 프로그램 설치**", expanded=True):
        st.markdown("""
        ### 📱 Smart Data 확장 프로그램 설치 방법
        
        #### 🌐 Chrome 웹스토어에서 설치
        1. **Chrome 웹스토어 접속**: [Smart Data 확장 프로그램](https://chromewebstore.google.com/detail/smart-data/mamdlaceocpnoajindmlhppbmjckmjcl?hl=ko)
        2. **'Chrome에 추가' 버튼 클릭**
        3. **'확장 프로그램 추가' 확인**
        
        #### 📊 데이터 다운로드 방법
        1. **스마트 스토어 상품 페이지 접속**
        2. **Smart Data 확장 프로그램 아이콘 클릭**
        3. **원하는 데이터 타입 선택 후 다운로드**:
           - 📝 **reviewcontents**: 리뷰 분석용
           - 📊 **옵션비율**: 옵션 분석용  
           - 💰 **스토어전체판매현황**: 매출 분석용
        """)
        
        # Smart Data 확장 프로그램 인터페이스 이미지 표시
        try:
            st.image("data/image1.png", caption="Smart Data 확장 프로그램 인터페이스", width=500)
        except Exception as e:
            st.warning("⚠️ 이미지 파일을 찾을 수 없습니다.")
            st.info("💡 Smart Data 확장 프로그램을 설치하면 위와 같은 인터페이스를 확인할 수 있습니다.")
        
        st.markdown("""
        
        #### ⚠️ 중요사항
        - **데이터 다운로드 시 해당 페이지를 벗어나지 마세요.**
        - **다운로드 파일명**: 확장 프로그램에서 다운로드되는 파일명을 그대로 사용하세요
        """)
    
    with st.expander("**2단계: 데이터 파일 업로드**"):
        st.markdown("""
        ### 📁 파일 업로드 방법
        1. **좌측 사이드바**의 '데이터 업로드' 섹션으로 이동
        2. **'스마트 스토어 데이터 파일' 업로드 버튼** 클릭
        3. **다운로드받은 파일 선택** (xlsx, csv 지원)
        
        ### 📋 지원되는 파일 타입
        - **reviewcontents**: 리뷰 데이터 (워드클라우드, 감정분석용)
        - **옵션비율**: 옵션 판매 데이터 (옵션분석용)
        - **스토어전체판매현황**: 매출 데이터 (매출분석용)
        
        ### 🔍 자동 파일 인식
        - 업로드된 파일의 컬럼을 분석하여 **자동으로 파일 타입을 감지**합니다
        - 별도의 설정 없이 바로 분석이 가능합니다
        """)
    
    with st.expander("**3단계: 분석 메뉴 선택**"):
        st.markdown("""
        ### 📊 분석 메뉴 가이드
        
        #### 📝 리뷰 분석 - 워드클라우드
        - **목적**: 고객 리뷰에서 자주 언급되는 키워드 파악
        - **결과**: 워드클라우드 시각화, 상위 20개 키워드 차트
        - **활용**: 고객 관심사 파악, 마케팅 포인트 도출
        
        #### 😊 리뷰 분석 - 감정분석  
        - **목적**: 리뷰의 감정(긍정/중립/부정) 및 세부 카테고리 분석
        - **결과**: 감정 분포, 카테고리별 상세 분석
        - **활용**: 고객 만족도 측정, 개선점 발견
        
        #### 🎯 옵션 분석
        - **목적**: 상품 옵션별 판매량 분석  
        - **결과**: 인기 옵션 Top 10, 판매량 시각화
        - **활용**: 재고 관리, 인기 옵션 파악
        
        #### 📈 스토어 전체 판매현황
        - **목적**: 종합적인 매출 성과 분석
        - **결과**: 매출 랭킹, 효율성 분석, 가격대별 분석, 리뷰-매출 인사이트
        - **활용**: 매출 전략 수립, 상품 포트폴리오 최적화
        """)
    
    with st.expander("**4단계: 결과 확인 및 비즈니스 인사이트 도출**"):
        st.markdown("""
        ### 📊 분석 결과 활용 방법
        
        #### 📈 성과 개선
        - **워드클라우드**: 고객이 중요하게 생각하는 키워드를 마케팅에 활용
        - **감정분석**: 부정 리뷰 카테고리를 통해 개선점 우선순위 설정
        - **옵션분석**: 인기 옵션 재고 확충, 비인기 옵션 정리
        
        #### 💡 전략 수립
        - **매출분석**: 효율성 높은 상품 확대, 잠재력 미달 상품 개선
        - **가격분석**: 가격대별 전략 차별화
        - **리뷰-매출 인사이트**: 숨겨진 보석 상품 발굴 및 집중 마케팅
        
        #### 📊 지속적 모니터링  
        - **정기적 분석**: 월 1회 이상 정기적으로 데이터 분석 수행
        - **트렌드 파악**: 시간에 따른 고객 반응 변화 모니터링
        - **전략 조정**: 분석 결과에 따른 마케팅 전략 지속적 개선
        """)
    
    # 도움말 섹션
    st.subheader("🆘 도움말")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### ❓ 자주 묻는 질문
        
        **Q: 파일을 업로드하지 않아도 분석이 가능한가요?**  
        A: 네! 샘플 데이터로 모든 분석 기능을 체험할 수 있습니다.
        
        **Q: 어떤 파일 형식을 지원하나요?**  
        A: Excel(.xlsx)과 CSV(.csv) 파일을 지원합니다.
        
        **Q: 데이터가 안전하게 처리되나요?**  
        A: 업로드된 데이터는 분석 목적으로만 사용되며 별도 저장되지 않습니다.
        """)
    
    with col2:
        st.markdown("""
        ### 🔧 문제 해결
        
        **파일 업로드가 안 될 때:**  
        - 파일 크기가 200MB 이하인지 확인
        - 파일 형식이 xlsx, csv인지 확인
        - 브라우저 새로고침 후 재시도
        
        **분석 결과가 이상할 때:**  
        - 데이터에 한글이 포함되어 있는지 확인
        - 필수 컬럼이 존재하는지 확인
        - 샘플 데이터로 먼저 테스트
        """)

elif analysis_option not in ["홈", "데이터 분석 사용안내"]:
    # 분석이 선택된 경우 - 업로드된 파일이 있으면 사용하고, 없으면 샘플 데이터 사용
    try:
        # 업로드된 파일이 없는 경우에만 샘플 데이터 로드
        if not uploaded_files:
            if analysis_option in ["리뷰 분석 - 워드클라우드", "리뷰 분석 - 감정분석"]:
                try:
                    review_df = pd.read_excel("data/reviewcontents.xlsx")
                    review_df = check_review_columns(review_df)
                except FileNotFoundError:
                    st.warning("⚠️ 샘플 리뷰 데이터 파일을 찾을 수 없습니다. 좌측 사이드바에서 리뷰 데이터 파일을 업로드해주세요.")
                    st.stop()
            elif analysis_option == "스토어 전체 판매현황":
                try:
                    sales_df = pd.read_excel("data/스토어전체판매현황.xlsx")
                except FileNotFoundError:
                    st.warning("⚠️ 샘플 판매현황 데이터 파일을 찾을 수 없습니다. 좌측 사이드바에서 판매현황 데이터 파일을 업로드해주세요.")
                    st.stop()
            
            if analysis_option == "옵션 분석":
                try:
                    option_df = pd.read_excel("data/옵션비율.xlsx")
                    option_df = check_option_columns(option_df)
                except FileNotFoundError:
                    st.warning("⚠️ 샘플 옵션 데이터 파일을 찾을 수 없습니다. 좌측 사이드바에서 옵션 데이터 파일을 업로드해주세요.")
                    st.stop()
        
        # 분석 실행
        if analysis_option == "리뷰 분석 - 워드클라우드":
            st.header("📊 리뷰 워드클라우드 분석")
            
            # 데이터 확인
            if review_df is None or review_df.empty:
                st.error("⚠️ 리뷰 데이터가 없습니다. 리뷰 컨텐츠 파일을 업로드해주세요.")
                st.info("💡 업로드된 파일에서 다음 컬럼 중 하나가 포함되어야 합니다: REVIEW_CONTENT, review_content, 리뷰내용, 내용, CONTENT")
                st.stop()

            if 'review_content' not in review_df.columns:
                st.error("⚠️ 리뷰 내용 컬럼을 찾을 수 없습니다.")
                st.info(f"현재 컬럼: {list(review_df.columns)}")
                st.stop()
            
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
                        ax.set_aspect('equal')  # 정사각형 비율 강제 적용
                        plt.tight_layout(pad=0)
                        st.pyplot(fig1)
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
        
        elif analysis_option == "리뷰 분석 - 감정분석":
            st.header("😊 리뷰 감정분석")
            
            # 데이터 확인
            if review_df is None or review_df.empty:
                st.error("⚠️ 리뷰 데이터가 없습니다. 리뷰 컨텐츠 파일을 업로드해주세요.")
                st.info("💡 업로드된 파일에서 다음 컬럼 중 하나가 포함되어야 합니다: REVIEW_CONTENT, review_content, 리뷰내용, 내용, CONTENT")
                st.stop()

            if 'review_content' not in review_df.columns:
                st.error("⚠️ 리뷰 내용 컬럼을 찾을 수 없습니다.")
                st.info(f"현재 컬럼: {list(review_df.columns)}")
                st.stop()
            
            # 분석 가이드 추가
            with st.expander("📖 감정분석 가이드", expanded=False):
                st.markdown("""
                ### 🎯 감정분석이란?
                고객 리뷰의 **감정을 자동으로 분류**하여 긍정, 중립, 부정으로 나누는 분석입니다.
                
                ### 🏷️ 세부 카테고리 분석
                - **긍정**: 상품의 장점을 나타내는 여러 카테고리로 자동 분류 (품질, 가격, 배송, 서비스 등)
                - **중립**: 특별한 감정 없이 중성적인 표현으로 분류 (보통, 무난, 평범 등)
                - **부정**: 상품의 문제점이나 불만사항으로 분류 (품질 문제, 배송 지연, 가격 불만 등)
                
                *※ 카테고리는 업로드하신 리뷰 데이터에 따라 동적으로 생성됩니다*
                
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

                # 수치 해석 안내 추가
                st.info("""
                📊 **수치 해석 안내**  
                
                • **상단 차트**: 전체 리뷰를 감정별로 분류한 결과입니다.  
                • **카테고리별 분석**: 키워드가 포함된 리뷰만 집계하며, 한 리뷰가 여러 카테고리에 중복 포함될 수 있습니다.  
                • **예시**: "맛있고 포장도 좋아요" → 맛 카테고리 + 배송 카테고리에 각각 1개씩 카운팅  
                • 따라서 카테고리별 합계가 상단 차트 수치와 다를 수 있습니다.
                """)

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
        
        elif analysis_option == "옵션 분석":
            st.header("🎯 옵션 분석")
            
            # 데이터 확인
            if option_df is None or option_df.empty:
                st.error("⚠️ 옵션 데이터가 없습니다. 옵션 비율 파일을 업로드해주세요.")
                st.info("💡 업로드된 파일에서 옵션 정보와 수량 컬럼이 모두 포함되어야 합니다:")
                st.info("- 옵션 컬럼: OPTION_INFO, option_info, 옵션정보, 옵션명, 상품옵션")
                st.info("- 수량 컬럼: COUNT, count, 수량, 판매량, 판매수량")
                st.stop()
            
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
        
        elif analysis_option == "스토어 전체 판매현황":
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
                    
                    #### 2. ⚡ 가격대비 매출효율성 분석
                    **매출효율성 = 해당 기간 매출 ÷ 기본판매가격**
                    - 상품 가격 대비 얼마나 효율적으로 매출을 올렸는지 측정
                    - 높은 효율성 = 가격 대비 매출 성과가 좋음 (예: 2.5 = 가격의 2.5배 매출)
                    
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
                    
                    # 탭 폰트 크기 강제 적용 (감정분석과 동일한 크기)
                    st.markdown("""
                    <style>
                    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
                        font-size: 24px !important;
                        font-weight: 600 !important;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    # 분석 탭 생성
                    tab1, tab2, tab3, tab4 = st.tabs(["매출 랭킹", "매출 효율성", "가격대별 분석", "리뷰-매출 인사이트"])
                    
                    with tab1:
                        st.subheader(f"🏆 {selected_period} 매출 상위 10개 상품")
                        top_products = analyze_top_products_by_period(sales_df, selected_period, 10)
                        
                        if not top_products.empty:
                            st.dataframe(top_products, use_container_width=True, hide_index=True)
                            
                            # 매출 랭킹 시각화
                            fig, ax = plt.subplots(figsize=(12, 6))
                            
                            bars = ax.bar(range(len(top_products)), 
                                         top_products[f'{selected_period} 매출'], 
                                         color='steelblue')
                            
                            ax.set_xticks(range(len(top_products)))
                            ax.set_xticklabels([name[:15] + '...' if len(name) > 15 else name 
                                               for name in top_products['상품명']], 
                                              rotation=45, ha='right')
                            
                            for i, v in enumerate(top_products[f'{selected_period} 매출']):
                                ax.text(i, v + max(top_products[f'{selected_period} 매출']) * 0.01, 
                                       f'{v:,.0f}', ha='center', va='bottom', fontsize=8)
                            
                            ax.set_ylabel('매출 (원)')
                            ax.set_title(f'{selected_period} 매출 상위 10개 상품')
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
                        else:
                            st.info("매출 효율성 분석을 위한 데이터가 부족합니다.")
                    
                    with tab3:
                        st.subheader(f"💰 가격대별 {selected_period} 매출 분석")
                        price_segments = analyze_price_segments(sales_df, selected_period)
                        
                        if not price_segments.empty:
                            st.dataframe(price_segments, use_container_width=True, hide_index=True)
                        else:
                            st.info("가격대별 분석을 위한 데이터가 부족합니다.")
                    
                    with tab4:
                        st.subheader(f"💡 {selected_period} 리뷰-매출 인사이트")
                        
                        # 리뷰 데이터 컬럼 확인
                        has_review_score = '리뷰점수' in sales_df.columns
                        has_review_count = '리뷰수' in sales_df.columns
                        has_price = '기본판매가격' in sales_df.columns
                        
                        if not has_review_score and not has_review_count:
                            st.info("💡 리뷰-매출 인사이트 분석을 위해서는 리뷰 점수 또는 리뷰수 데이터가 필요합니다.")
                            st.info("📋 필요한 컬럼: '리뷰점수', '리뷰수', '기본판매가격' (선택사항)")
                        else:
                            # 리뷰 효율성 분석
                            st.markdown("#### 📈 리뷰 효율성 분석")
                            if has_review_count:
                                st.info("💡 리뷰 1건당 매출이 높은 상품 분석")
                                
                                with st.spinner("리뷰 효율성 분석 중..."):
                                    efficiency_result = analyze_review_efficiency(sales_df, selected_period)
                                    
                                    if not efficiency_result.empty:
                                        st.dataframe(efficiency_result, use_container_width=True, hide_index=True)
                                    else:
                                        st.info("분석 가능한 데이터가 부족합니다.")
                            else:
                                st.info("리뷰 효율성 분석을 위해서는 '리뷰수' 컬럼이 필요합니다.")
                            
                            st.divider()
                            
                            # 숨겨진 보석 상품
                            st.markdown("#### 💎 숨겨진 보석 상품")
                            if has_review_score:
                                st.info("💡 매출은 낮지만 리뷰 점수가 높은 상품 (리뷰 점수 4.5+ & 매출 하위 50%)")
                                
                                with st.spinner("숨겨진 보석 분석 중..."):
                                    gems_result = analyze_hidden_gems(sales_df, selected_period)
                                    
                                    if not gems_result.empty:
                                        st.dataframe(gems_result, use_container_width=True, hide_index=True)
                                    else:
                                        st.info("조건에 맞는 숨겨진 보석 상품이 없습니다.")
                            else:
                                st.info("숨겨진 보석 분석을 위해서는 '리뷰점수' 컬럼이 필요합니다.")
                            
                            st.divider()
                            
                            # 잠재력 미달 상품
                            st.markdown("#### ⚠️ 잠재력 미달 상품")
                            if has_review_score:
                                st.info("💡 리뷰는 좋은데 매출이 예상보다 낮은 상품 (리뷰 점수 4.0+ & 매출 상위 75% 미달)")
                                
                                with st.spinner("잠재력 미달 분석 중..."):
                                    underperform_result = analyze_underperforming_products(sales_df, selected_period)
                                    
                                    if not underperform_result.empty:
                                        st.dataframe(underperform_result, use_container_width=True, hide_index=True)
                                    else:
                                        st.info("조건에 맞는 잠재력 미달 상품이 없습니다.")
                            else:
                                st.info("잠재력 미달 분석을 위해서는 '리뷰점수' 컬럼이 필요합니다.")
                            
                            st.divider()
                            
                            # 리뷰 확보 필요 상품
                            st.markdown("#### 📝 리뷰 확보 필요 상품")
                            if has_review_count:
                                st.info("💡 매출은 높은데 리뷰가 적은 상품 (매출 상위 50% & 리뷰수 하위 50%)")
                                
                                with st.spinner("리뷰 확보 필요 분석 중..."):
                                    review_needed_result = analyze_review_needed_products(sales_df, selected_period)
                                    
                                    if not review_needed_result.empty:
                                        st.dataframe(review_needed_result, use_container_width=True, hide_index=True)
                                    else:
                                        st.info("조건에 맞는 리뷰 확보 필요 상품이 없습니다.")
                            else:
                                st.info("리뷰 확보 필요 분석을 위해서는 '리뷰수' 컬럼이 필요합니다.")
                            
                            st.divider()
                            
                            # 가성비 인증 상품
                            st.markdown("#### 💰 가성비 인증 상품")
                            if has_review_score and has_price:
                                st.info("💡 저렴한 가격 + 높은 리뷰 점수 상품 (가격 하위 50% & 리뷰 점수 4.0+)")
                                
                                with st.spinner("가성비 인증 분석 중..."):
                                    value_result = analyze_value_products(sales_df, selected_period)
                                    
                                    if not value_result.empty:
                                        st.dataframe(value_result, use_container_width=True, hide_index=True)
                                    else:
                                        st.info("조건에 맞는 가성비 인증 상품이 없습니다.")
                            else:
                                st.info("가성비 인증 분석을 위해서는 '리뷰점수'와 '기본판매가격' 컬럼이 필요합니다.")
                        
            else:
                st.error("⚠️ 판매현황 데이터가 없습니다. 스토어 전체 판매현황 파일을 업로드해주세요.")
                st.info("💡 업로드된 파일에서 '상품명' 컬럼과 '매출' 관련 컬럼이 포함되어야 합니다.")
                st.stop()

    except Exception as e:
        st.error(f"데이터 처리 중 오류가 발생했습니다: {e}")
