import pandas as pd
import matplotlib.pyplot as plt
import math

arr_genre = [
    'pop',
    'rock',
    'indie',
    'hippop',
    'jazz',
    'metal',
    'punk',
    'country',
    'soul',
    'house',
    'folk',
    'funk',
    'christian',
    'rnb',
    'dance',
    'latin',
    'adult_standards',
    'mellow_gold',
    'brill_building_pop',
    'easy_listening',
    'motown'
]

feature_name = 'mode'


def read_csv(genre_name):
    filepath = "C:/00.Python/files_project_1/files_project_1_3_mode/"
    filename = f'{feature_name}_{genre_name}.csv'
    file = filepath + filename
    # print(file)
    data = pd.read_csv(file, header=0, encoding='ANSI', sep=',')
    return data


# 장르별로 차트 그리기
for genre in arr_genre:
    print('--------------------', genre, '--------------------')
    # 파일 읽어오기 & 리스트로 담기
    df = read_csv(genre)
    df.columns = [genre]
    list_df = df[genre].values
    list_df = [x for x in list_df if math.isnan(x) == False]
    # print(list_df)

    # 전체 개수
    cnt_all = len(list_df)
    print('전체 개수 :', cnt_all)

    # 고유값
    u_values = df[genre].unique()
    # 고유값별 백분율
    ratio = []
    for value in u_values:
        cnt = list_df.count(value)
        percent = round(cnt/cnt_all * 100, 1)
        print(value, '>>> 개수 :', cnt, '/', percent, '%')
        ratio.append(percent)
    print('ratio=', ratio)

    explode = [0.05, 0]
    textprops = {'fontsize' : 11}
    plt.pie(ratio, autopct='%.1f%%', startangle=90, explode=explode, textprops=textprops)

    plt.title(genre.capitalize(), size=15)
    plt.xlabel(feature_name.capitalize(), size=13)
    plt.legend(labels=u_values, loc='upper right')

    plt.show()
