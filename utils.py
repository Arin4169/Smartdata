import pandas as pd
import numpy as np
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from konlpy.tag import Okt
import streamlit as st
import matplotlib.font_manager as fm
import os
import platform

# 한글 자연어 처리를 위한 Okt 객체 초기화
okt = Okt()

# 기본 불용어 목록 (필요에 따라 추가 가능)
DEFAULT_STOPWORDS = ['이', '가', '은', '는', '을', '를', '에', '의', '과', '와', '에서', '로', '으로', '하다', '있다', '되다', '것']

# 불용어를 관리하는 함수
def get_stopwords():
    """세션에 저장된 불용어 목록을 반환하거나 기본 불용어 목록을 초기화합니다."""
    if 'stopwords' not in st.session_state:
        st.session_state.stopwords = DEFAULT_STOPWORDS.copy()
    return st.session_state.stopwords

def add_stopword(word):
    """불용어 목록에 새 단어를 추가합니다."""
    if 'stopwords' not in st.session_state:
        st.session_state.stopwords = DEFAULT_STOPWORDS.copy()
    
    # 공백으로 구분된 여러 단어를 처리
    words = [w.strip() for w in word.split() if w.strip()]
    for w in words:
        if w not in st.session_state.stopwords:
            st.session_state.stopwords.append(w)
    
    return st.session_state.stopwords

def reset_stopwords():
    """불용어 목록을 기본값으로 초기화합니다."""
    st.session_state.stopwords = DEFAULT_STOPWORDS.copy()
    return st.session_state.stopwords

def remove_stopword(word):
    """불용어 목록에서 단어를 제거합니다."""
    if 'stopwords' not in st.session_state:
        st.session_state.stopwords = DEFAULT_STOPWORDS.copy()
    
    if word in st.session_state.stopwords:
        st.session_state.stopwords.remove(word)
    
    return st.session_state.stopwords

# 한글 폰트 경로 찾기
def get_font_path():
    system_platform = platform.system()
    
    # 윈도우의 경우
    if system_platform == 'Windows':
        font_candidates = [
            'C:/Windows/Fonts/malgun.ttf',  # 맑은 고딕
            'C:/Windows/Fonts/gulim.ttc',   # 굴림
            'C:/Windows/Fonts/batang.ttc',  # 바탕
            'C:/Windows/Fonts/gothic.ttf'   # 고딕
        ]
    # macOS의 경우
    elif system_platform == 'Darwin':
        font_candidates = [
            '/Library/Fonts/AppleGothic.ttf',
            '/Library/Fonts/AppleSDGothicNeo.ttc',
            '/System/Library/Fonts/AppleSDGothicNeo.ttc'
        ]
    # 리눅스의 경우 (Streamlit Cloud 포함)
    else:
        font_candidates = [
            '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
            '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf',
            '/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf',
            '/usr/share/fonts/truetype/nanum/NanumMyeongjo.ttf'
        ]
    
    # 존재하는 폰트 경로 반환
    for font_path in font_candidates:
        if os.path.exists(font_path):
            return font_path
    
    # 시스템에 설치된 폰트 중 한글 폰트 찾기
    try:
        korean_fonts = [f for f in fm.findSystemFonts() if any(name in f.lower() for name in ['gothic', 'gulim', 'batang', 'malgun', 'nanum', 'gungsuh'])]
        
        if korean_fonts:
            return korean_fonts[0]
    except Exception as e:
        print(f"폰트 검색 중 오류: {e}")
    
    # 한글 폰트가 없는 경우
    print("한글 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")
    return None

# 시스템에서 사용 가능한 한글 폰트 경로 찾기
KOREAN_FONT_PATH = get_font_path()

def clean_text(text):
    """텍스트 전처리 함수"""
    if not isinstance(text, str):
        return ""
    
    # 특수문자 및 숫자 제거
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'[0-9]', ' ', text)
    # 여러 공백을 하나로 치환
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_nouns(text):
    """명사 추출 함수"""
    clean = clean_text(text)
    
    if not clean:
        return []
    
    # 명사 추출
    nouns = okt.nouns(clean)
    
    # 현재 세션의 불용어 목록 가져오기
    stopwords = get_stopwords()
    
    # 불용어 및 한 글자 단어 제거
    nouns = [word for word in nouns if word not in stopwords and len(word) > 1]
    
    return nouns

def generate_wordcloud_data(df, column_name='review_content'):
    """워드클라우드 생성 데이터 준비 함수"""
    
    # 모든 리뷰 텍스트 결합
    all_reviews = ' '.join(df[column_name].dropna().astype(str))
    
    # 명사 추출
    all_nouns = extract_nouns(all_reviews)
    
    # 빈도수 계산
    word_count = Counter(all_nouns)
    
    # 상위 단어 추출
    top_words = dict(word_count.most_common(20))
    
    return word_count, top_words

def create_wordcloud(word_count, width=1200, height=800):
    """워드클라우드 시각화 함수"""
    
    # 워드클라우드 생성
    wc_params = {
        'width': width, 
        'height': height, 
        'background_color': 'white',
        'max_words': 100,
        'prefer_horizontal': 0.9
    }
    
    # 한글 폰트 경로가 있으면 추가
    if KOREAN_FONT_PATH:
        wc_params['font_path'] = KOREAN_FONT_PATH
    
    wc = WordCloud(**wc_params)
    
    # 단어 빈도수 데이터로 워드클라우드 생성
    wc.generate_from_frequencies(word_count)
    
    return wc

def simple_sentiment_analysis(df, column_name='review_content'):
    """간단한 감정 분석 함수"""
    
    # 긍정/부정 키워드 (실제로는 더 많은 단어와 더 정교한 방법 사용 필요)
    positive_words = ['좋다', '좋은', '좋아요', '만족', '최고', '추천', '맛있다', '편리하다', '빠르다', '친절하다']
    negative_words = ['나쁘다', '별로', '실망', '불만', '최악', '싫다', '아쉽다', '느리다', '불친절하다']
    
    # 감정 점수 계산 함수
    def get_sentiment_score(text):
        if not isinstance(text, str):
            return 0
        
        clean = clean_text(text)
        if not clean:
            return 0
        
        # 형태소 분석
        morphs = okt.morphs(clean)
        
        # 긍정/부정 점수 계산
        positive_score = sum(1 for word in morphs if word in positive_words)
        negative_score = sum(1 for word in morphs if word in negative_words)
        
        # 최종 점수 (-1: 매우 부정, 1: 매우 긍정)
        return (positive_score - negative_score) / (positive_score + negative_score + 0.001)
    
    # 리뷰별 감정 점수 계산
    df['sentiment_score'] = df[column_name].apply(get_sentiment_score)
    
    # 긍정/중립/부정 분류
    df['sentiment'] = df['sentiment_score'].apply(
        lambda x: '긍정' if x > 0.3 else ('부정' if x < -0.3 else '중립')
    )
    
    # 감정별 카운트
    sentiment_counts = df['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['감정', '리뷰 수']
    
    return df, sentiment_counts

def analyze_options(df, option_column='option_info', count_column='count'):
    """옵션 분석 함수"""
    
    # 상위 10개 옵션 추출
    top_options = df.sort_values(by=count_column, ascending=False).head(10)
    
    # 인덱스를 1부터 10까지 재설정
    top_options = top_options.reset_index(drop=True)
    top_options.index = top_options.index + 1
    
    return top_options

def analyze_positive_review_categories(df, review_column):
    """긍정 리뷰를 카테고리별로 분석합니다."""
    
    # 카테고리별 키워드 정의 (확장 가능)
    category_keywords = {
        '맛': ['맛있', '달콤', '고소', '진한', '부드러', '깔끔', '신선', '풍미', '향', '달달', '짭짤', '매콤', '시원', '담백', '진짜맛있', '존맛'],
        '식감': ['쫄깃', '바삭', '촉촉', '부드러', '탱탱', '씹히', '질감', '식감', '텍스처', '크런치', '쫀득', '말랑', '단단'],
        '배송': ['배송', '포장', '빠른', '신속', '안전', '포장상태', '배달', '택배', '도착', '빨리', '신속배송', '당일배송'],
        '가격': ['저렴', '합리적', '가성비', '할인', '싼', '경제적', '가격', '비용', '돈', '가격대비', '세일', '특가'],
        '서비스': ['친절', '응답', '문의', '교환', '환불', '고객서비스', '직원', '상담', '대응', '서비스', '응대'],
        '품질': ['품질', '만족', '좋은', '훌륭', '우수', '최고', '완벽', '정성', '고급', '퀄리티'],
        '외관': ['예쁜', '깔끔', '포장', '디자인', '색깔', '모양', '보기좋', '깨끗', '이쁜', '예뻐', '디자인이쁜'],
        '양': ['많이', '푸짐', '양많', '충분', '넉넉', '가득', '풍성', '듬뿍']
    }
    
    return _analyze_review_categories_by_sentiment(df, review_column, '긍정', category_keywords)

def analyze_neutral_review_categories(df, review_column):
    """중립 리뷰를 카테고리별로 분석합니다."""
    
    # 중립 리뷰 카테고리별 키워드 정의
    category_keywords = {
        '일반적': ['그냥', '보통', '평범', '무난', '일반적', '나쁘지않', '그럭저럭', '평균'],
        '애매한 맛': ['그저그런', '평범한맛', '특별하지않', '무난한맛', '그런대로'],
        '보통 품질': ['보통품질', '평균적', '무난한품질', '그럭저럭품질'],
        '가격 무난': ['적당', '그럭저럭가격', '무난한가격', '평균가격'],
        '배송 보통': ['보통배송', '평균배송', '무난한배송'],
        '애매한 평가': ['모르겠', '애매', '그냥그래', '특별한감정없', '딱히'],
        '기대와 다름': ['기대보다', '생각보다', '예상과달라', '기대와달라']
    }
    
    return _analyze_review_categories_by_sentiment(df, review_column, '중립', category_keywords)

def analyze_negative_review_categories(df, review_column):
    """부정 리뷰를 카테고리별로 분석합니다."""
    
    # 부정 리뷰 카테고리별 키워드 정의
    category_keywords = {
        '맛 문제': ['맛없', '별로', '짜다', '달다', '시다', '쓰다', '비린내', '냄새', '맛이이상', '맛이없어'],
        '품질 문제': ['품질나쁘', '조잡', '싸구려', '부실', '불량', '하자', '망가져', '깨져'],
        '배송 문제': ['배송늦', '포장불량', '배송문제', '늦게도착', '파손', '포장상태나쁘', '배송오류'],
        '가격 불만': ['비싸', '비쌈', '가격부담', '가성비나쁘', '돈아까워', '가격대비별로'],
        '서비스 불만': ['불친절', '응답없', '문의무시', '서비스나쁘', '대응늦', '무례'],
        '크기/양 부족': ['작다', '적어', '양적어', '크기작아', '부족', '양부족'],
        '기대 실망': ['실망', '기대이하', '후회', '별로야', '최악', '다시안사'],
        '기타 불만': ['불편', '문제', '고장', '작동안됨', '사용법복잡']
    }
    
    return _analyze_review_categories_by_sentiment(df, review_column, '부정', category_keywords)

def _analyze_review_categories_by_sentiment(df, review_column, sentiment_type, category_keywords):
    """특정 감정의 리뷰를 카테고리별로 분석하는 공통 함수"""
    
    # 해당 감정 리뷰만 필터링
    sentiment_reviews = df[df['sentiment'] == sentiment_type].copy()
    
    if len(sentiment_reviews) == 0:
        return pd.DataFrame(columns=['카테고리', '리뷰 수', '비율(%)', '주요 키워드'])
    
    # 각 카테고리별 분석
    category_results = []
    total_sentiment = len(sentiment_reviews)
    
    for category, keywords in category_keywords.items():
        # 해당 카테고리 키워드가 포함된 리뷰 필터링
        pattern = '|'.join(keywords)
        category_mask = sentiment_reviews[review_column].str.contains(pattern, na=False, case=False)
        category_reviews = sentiment_reviews[category_mask]
        
        if len(category_reviews) > 0:
            # 실제 언급된 키워드와 빈도 계산
            mentioned_keywords = []
            for keyword in keywords:
                keyword_count = sentiment_reviews[review_column].str.contains(keyword, na=False, case=False).sum()
                if keyword_count > 0:
                    mentioned_keywords.append((keyword, keyword_count))
            
            # 빈도순으로 정렬하고 상위 10개 선택
            mentioned_keywords.sort(key=lambda x: x[1], reverse=True)
            top_keywords = [f"{kw[0]}({kw[1]})" for kw in mentioned_keywords[:10]]
            
            # 비율 계산
            percentage = round((len(category_reviews) / total_sentiment) * 100, 1)
            
            category_results.append({
                '카테고리': category,
                '리뷰 수': len(category_reviews),
                '비율(%)': percentage,
                '주요 키워드': ', '.join(top_keywords)
            })
    
    # 리뷰 수 기준으로 정렬
    result_df = pd.DataFrame(category_results)
    if len(result_df) > 0:
        result_df = result_df.sort_values('리뷰 수', ascending=False).reset_index(drop=True)
    
    return result_df 


# 스토어 전체 판매현황 분석 함수들
def check_sales_columns(df):
    """스토어 전체 판매현황 파일의 컬럼을 확인하고 검증합니다."""
    required_columns = ['상품명']
    sales_periods = ['7일', '1개월', '3개월', '6개월', '1년', '2년']
    
    # 기본 컬럼 확인
    if not all(col in df.columns for col in required_columns):
        return False, f"필수 컬럼이 누락되었습니다: {required_columns}"
    
    # 매출 관련 컬럼 확인
    sales_columns = [f'{period}매출' for period in sales_periods]
    existing_sales_cols = [col for col in sales_columns if col in df.columns]
    
    if len(existing_sales_cols) < 2:
        return False, "최소 2개 이상의 매출 기간 컬럼이 필요합니다"
    
    return True, f"확인된 매출 컬럼: {existing_sales_cols}"


def get_sales_periods(df):
    """데이터프레임에서 사용 가능한 매출 기간들을 반환합니다."""
    periods = ['7일', '1개월', '3개월', '6개월', '1년', '2년']
    available_periods = []
    
    for period in periods:
        if f'{period}매출' in df.columns:
            available_periods.append(period)
    
    return available_periods


def analyze_top_products_by_period(df, period='1년', top_n=10):
    """선택된 기간의 상위 N개 상품 분석"""
    sales_col = f'{period}매출'
    
    if sales_col not in df.columns:
        return pd.DataFrame()
    
    # 매출이 0보다 큰 상품들만 필터링하고, "토탈" 항목 제외
    filtered_df = df[df[sales_col] > 0].copy()
    
    # "토탈", "TOTAL", "합계" 등의 항목 제외
    total_keywords = ['토탈', 'TOTAL', 'Total', '합계', '전체', '총계']
    for keyword in total_keywords:
        filtered_df = filtered_df[~filtered_df['상품명'].str.contains(keyword, na=False, case=False)]
    
    if filtered_df.empty:
        return pd.DataFrame()
    
    # 상위 N개 상품 추출
    top_products = filtered_df.nlargest(top_n, sales_col)
    
    # 결과 DataFrame 생성
    result_data = {
        '순위': range(1, len(top_products) + 1),
        '상품명': top_products['상품명'].values,
        f'{period} 매출': top_products[sales_col].values
    }
    
    # 기본판매가격 컬럼이 있으면 추가
    if '기본판매가격' in top_products.columns:
        result_data['기본판매가격'] = top_products['기본판매가격'].values
    
    # 판매건수 컬럼이 있으면 추가 (여러 가능한 컬럼명 확인)
    sales_count_cols = ['판매건수', f'{period}판매건수', '주문건수', f'{period}주문건수']
    for col in sales_count_cols:
        if col in top_products.columns:
            result_data['판매건수'] = top_products[col].values
            break
    
    # 컬럼 순서 조정: 순위, 상품명, 기본판매가격, 판매건수, 매출
    ordered_columns = ['순위', '상품명']
    if '기본판매가격' in result_data:
        ordered_columns.append('기본판매가격')
    if '판매건수' in result_data:
        ordered_columns.append('판매건수')
    ordered_columns.append(f'{period} 매출')
    
    result = pd.DataFrame(result_data)[ordered_columns]
    
    return result


def analyze_sales_efficiency(df, period='1년'):
    """가격 대비 매출 효율성 분석"""
    sales_col = f'{period}매출'
    price_col = '기본판매가격'
    
    if sales_col not in df.columns or price_col not in df.columns:
        return pd.DataFrame()
    
    # 가격과 매출 정보가 있는 상품들만 필터링
    filtered_df = df.dropna(subset=[price_col, sales_col]).copy()
    filtered_df = filtered_df[(filtered_df[price_col] > 0) & (filtered_df[sales_col] > 0)]
    
    # "토탈", "TOTAL", "합계" 등의 항목 제외
    total_keywords = ['토탈', 'TOTAL', 'Total', '합계', '전체', '총계']
    for keyword in total_keywords:
        filtered_df = filtered_df[~filtered_df['상품명'].str.contains(keyword, na=False, case=False)]
    
    if filtered_df.empty:
        return pd.DataFrame()
    
    # 가격대비 매출지수 계산 (매출/가격)
    filtered_df['가격대비매출지수'] = filtered_df[sales_col] / filtered_df[price_col]
    
    # 상위 10개 가격대비 매출지수 상품
    top_efficiency = filtered_df.nlargest(10, '가격대비매출지수')
    
    result = pd.DataFrame({
        '순위': range(1, len(top_efficiency) + 1),
        '상품명': top_efficiency['상품명'].values,
        '기본판매가격': top_efficiency[price_col].values,
        f'{period} 매출': top_efficiency[sales_col].values,
        '가격대비매출지수': top_efficiency['가격대비매출지수'].values
    })
    
    return result


def analyze_price_segments(df, period='1년'):
    """가격대별 매출 분석"""
    sales_col = f'{period}매출'
    price_col = '기본판매가격'
    
    if sales_col not in df.columns or price_col not in df.columns:
        return pd.DataFrame()
    
    # 가격과 매출 정보가 있는 상품들만 필터링
    filtered_df = df.dropna(subset=[price_col, sales_col]).copy()
    
    # "토탈", "TOTAL", "합계" 등의 항목 제외
    total_keywords = ['토탈', 'TOTAL', 'Total', '합계', '전체', '총계']
    for keyword in total_keywords:
        filtered_df = filtered_df[~filtered_df['상품명'].str.contains(keyword, na=False, case=False)]
    
    if filtered_df.empty:
        return pd.DataFrame()
    
    # 가격대별 구간 설정
    price_bins = [0, 10000, 30000, 50000, 100000, float('inf')]
    price_labels = ['1만원 이하', '1-3만원', '3-5만원', '5-10만원', '10만원 이상']
    
    filtered_df['가격대'] = pd.cut(filtered_df[price_col], bins=price_bins, labels=price_labels, right=False)
    
    # 가격대별 집계
    price_analysis = filtered_df.groupby('가격대', observed=False).agg({
        '상품명': 'count',
        sales_col: ['mean', 'sum', 'count']
    }).round(0)
    
    # 컬럼명 평탄화
    price_analysis.columns = ['상품수', '평균매출', '총매출', '매출상품수']
    price_analysis = price_analysis.reset_index()
    
    return price_analysis


def analyze_review_sales_correlation(df, period='1년'):
    """리뷰 점수와 매출의 상관관계 분석"""
    sales_col = f'{period}매출'
    review_score_col = '리뷰점수'
    review_count_col = '리뷰수'
    
    required_cols = [sales_col, review_score_col]
    if not all(col in df.columns for col in required_cols):
        return None, pd.DataFrame()
    
    # 필요한 데이터가 있는 상품들만 필터링
    filtered_df = df.dropna(subset=required_cols).copy()
    filtered_df = filtered_df[filtered_df[sales_col] > 0]
    
    # "토탈", "TOTAL", "합계" 등의 항목 제외
    total_keywords = ['토탈', 'TOTAL', 'Total', '합계', '전체', '총계']
    for keyword in total_keywords:
        filtered_df = filtered_df[~filtered_df['상품명'].str.contains(keyword, na=False, case=False)]
    
    if filtered_df.empty:
        return None, pd.DataFrame()
    
    # 상관계수 계산
    correlation = filtered_df[review_score_col].corr(filtered_df[sales_col])
    
    # 리뷰 점수 구간별 평균 매출
    score_bins = [0, 3.0, 4.0, 4.5, 5.0]
    score_labels = ['3.0 미만', '3.0-4.0', '4.0-4.5', '4.5-5.0']
    
    filtered_df['리뷰점수구간'] = pd.cut(filtered_df[review_score_col], bins=score_bins, labels=score_labels, right=False)
    
    # 구간별 집계
    review_analysis = filtered_df.groupby('리뷰점수구간', observed=False).agg({
        '상품명': 'count',
        sales_col: 'mean',
        review_count_col: 'mean' if review_count_col in filtered_df.columns else None
    }).round(0)
    
    if review_count_col in filtered_df.columns:
        review_analysis.columns = ['상품수', '평균매출', '평균리뷰수']
    else:
        review_analysis.columns = ['상품수', '평균매출']
    
    review_analysis = review_analysis.reset_index()
    
    return correlation, review_analysis


def calculate_sales_growth_pattern(df):
    """기간별 매출 성장 패턴 분석"""
    periods = ['7일', '1개월', '3개월', '6개월', '1년', '2년']
    available_periods = [p for p in periods if f'{p}매출' in df.columns]
    
    if len(available_periods) < 2:
        return pd.DataFrame()
    
    # 각 상품별 기간별 매출 추출
    growth_data = []
    
    for _, row in df.iterrows():
        product_name = row['상품명']
        sales_data = []
        
        for period in available_periods:
            sales_col = f'{period}매출'
            sales_data.append(row[sales_col] if pd.notna(row[sales_col]) else 0)
        
        # 단기(7일-1개월) vs 장기(1년-2년) 매출 비교
        if '7일' in available_periods and '1년' in available_periods:
            short_term = row['7일매출'] if '7일매출' in df.columns else 0
            long_term = row['1년매출'] if '1년매출' in df.columns else 0
            
            if long_term > 0:
                growth_ratio = short_term / long_term * 365 / 7  # 연간 환산
                growth_data.append({
                    '상품명': product_name,
                    '단기매출(7일)': short_term,
                    '장기매출(1년)': long_term,
                    '성장패턴': '안정형' if 0.8 <= growth_ratio <= 1.2 else ('성장형' if growth_ratio > 1.2 else '감소형')
                })
    
    return pd.DataFrame(growth_data)


def get_sales_summary_stats(df, period='1년'):
    """매출 요약 통계"""
    sales_col = f'{period}매출'
    
    if sales_col not in df.columns:
        return {}
    
    # 토탈 항목을 제외한 실제 상품들만 필터링
    filtered_df = df[df[sales_col] > 0].copy()
    
    # "토탈", "TOTAL", "합계" 등의 항목 제외
    total_keywords = ['토탈', 'TOTAL', 'Total', '합계', '전체', '총계']
    for keyword in total_keywords:
        filtered_df = filtered_df[~filtered_df['상품명'].str.contains(keyword, na=False, case=False)]
    
    sales_data = filtered_df[sales_col]
    
    if sales_data.empty:
        return {}
    
    summary = {
        '총매출': int(sales_data.sum()),
        '평균매출': int(sales_data.mean()),
        '중간값매출': int(sales_data.median()),
        '최대매출': int(sales_data.max()),
        '최소매출': int(sales_data.min()),
        '상품수': len(sales_data),
        '매출상위10%기준': int(sales_data.quantile(0.9))
    }
    
    return summary