import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def vis_data(df, x, y, title=None, color_palette=None):
    grouped_data = df.groupby(x)[y].mean().reset_index()

    # 색상 팔레트 설정
    if color_palette is None:
        color_palette = sns.color_palette("tab10", len(grouped_data[x].unique()))
    colors = dict(zip(grouped_data[x].unique(), color_palette))

    # 바 차트 생성
    plt.figure(figsize=(8, 6))
    bars = plt.bar(grouped_data[x], grouped_data[y], color=[colors[x] for x in grouped_data[x]])

    # 범례 추가
    plt.legend(bars, grouped_data[x].unique())  # 바 객체와 고유한 x 값을 범례로 표시

    # 제목 설정 (선택사항)
    if title:
        plt.title(title)

    plt.xlabel(x)  # X축 레이블 설정
    plt.ylabel(y)  # Y축 레이블 설정
    plt.xticks(rotation=45)  # X 축 틱 라벨 45도 회전
    plt.tight_layout()  # 레이아웃 조정
    plt.show()

def plot_histogram(df, column, title=None, bins=10):
    plt.figure(figsize=(8, 6))
    plt.hist(df[column], bins=bins, edgecolor='black')

    if title:
        plt.title(title)

    plt.xlabel(column)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()