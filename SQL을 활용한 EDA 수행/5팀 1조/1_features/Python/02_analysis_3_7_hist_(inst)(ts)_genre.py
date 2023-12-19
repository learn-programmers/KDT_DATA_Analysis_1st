import pandas as pd
import matplotlib.pyplot as plt


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
# print(len(arr_genre))


def read_csv(genre_name):
    filepath = 'C:/00.Python/files_project_1/files_project_1_3_instrumentalness/'
    filename = f'inst_not_0_{genre_name}.csv'
    # filepath = "C:/00.Python/files_project_1/files_project_1_3_time_signature/"
    # filename = f'ts_not_4_{genre_name}.csv'
    file = filepath + filename
    # print(file)
    # 방법1
    data = pd.read_csv(file, header=0, encoding='ANSI', sep=',')
    return data


# 장르별로 차트 그리기
for genre in arr_genre:
    print('--------------------', genre, '--------------------')
    # 파일 읽어오기 & 리스트로 담기
    df = read_csv(genre)
    # print(df)
    # breakpoint()

    # df 값 개수 카운트
    cnt = pd.DataFrame()
    for i in df.columns:
        cnt_new = df[i].value_counts()
        if cnt is None:
            cnt = cnt_new
        else:
            cnt = pd.concat([cnt, cnt_new], axis=1)
    # 컬럼별로 카운트한 개수 합치기
    cnt_sum = cnt[cnt.columns].sum(axis=1)

    # 컬럼명 변경
    cnt_sum = cnt_sum.reset_index()
    cnt_sum.columns = ['value', 'count']

    # pd.set_option('display.max_rows', None)
    # print(cnt_sum.columns)
    print(cnt_sum)
    # breakpoint()

    # ------------------------------ hist ---------------------------------------
    # 기본 스타일 설정
    plt.style.use('default')
    plt.rcParams['font.size'] = 11

    # 데이터 읽어오기
    arr_values = cnt_sum['value'].values.tolist()

    # 그래프 그리기
    plt.hist(arr_values, bins=50, range=[0, 1], density=True, color='dodgerblue', edgecolor='white')
    # plt.hist(arr_values, density=True, color='skyblue', edgecolor='white')

    plt.title('Instrumentalness', size = 15)
    # plt.title('Time Signature', size=15)
    plt.ylabel('density', size=13)
    plt.xlabel(genre, size=13)
    plt.ylim([0, 21])
    # plt.ylim([0, 11])

    plt.show()