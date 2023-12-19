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
    # filepath = 'C:/00.Python/files_project_1/files_project_1_3_instrumentalness/'
    # filename = f'inst_not_0_{genre_name}.csv'
    filepath = "C:/00.Python/files_project_1/files_project_1_3_time_signature/"
    filename = f'ts_not_4_{genre_name}.csv'
    file = filepath + filename
    # print(file)
    # 방법1
    data = pd.read_csv(file, header=0, encoding='ANSI', sep=',')
    return data


# 데이터 프레임 만들기
df = pd.DataFrame()
for i in arr_genre:
    df_new = read_csv(i)
    if df is None:
        df = df_new
    else:
        df = pd.concat([df, df_new], axis=1)
# 컬럼명 변경
df.columns = arr_genre

# pd.set_option('display.max_columns', None)
# print(df.head(5))
# print(df.columns)
# breakpoint()


# df 값 개수 카운트
cnt = pd.DataFrame()
for i in arr_genre:
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

pd.set_option('display.max_rows', None)
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
# plt.hist(arr_values, bins=50, edgecolor='white')
# plt.title('Instrumentalness', size = 15)
plt.hist(arr_values, edgecolor='white')
plt.title('Time Signature', size = 15)
plt.ylabel('density', size=13)

plt.show()