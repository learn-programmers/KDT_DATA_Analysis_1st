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
# print(len(arr_genre))


feature_name = 'valence'


def read_csv(genre_name):
    filepath = "C:/00.Python/files_project_1/files_project_1_3/"
    filename = f'{feature_name}_{genre_name}.csv'
    file = filepath + filename
    # print(file)
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


# ------------------------------ boxplot ---------------------------------------
# 기본 스타일 설정
plt.style.use('default')
plt.rcParams['figure.figsize'] = (18, 8)
plt.rcParams['font.size'] = 11

# 데이터 읽어오기
arr_data = []
for i in arr_genre:
    col_df = df[i].values.tolist()
    col_df = [x for x in col_df if math.isnan(x) == False] # 빈 값 제거
    arr_data.append(col_df)


# 그래프 그리기
plt.boxplot(arr_data, notch=True)
plt.title('Genre', size = 15)
plt.ylabel('Valence', size = 15)

# danceability / energy / valence
plt.ylim(-0.2, 1.2)
# key
# plt.ylim(-17, 13)
# loudness
# plt.ylim(-30, 4)
# speechiness / acousticness / liveness
# plt.ylim(-0.05, 1.05)
# tempo
# plt.ylim(-0.2, 220)


plt.xticks(range(1,len(arr_genre)+1), arr_genre)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()