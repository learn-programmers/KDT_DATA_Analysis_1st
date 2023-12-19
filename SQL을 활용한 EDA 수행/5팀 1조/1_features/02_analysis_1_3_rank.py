import pandas as pd
import csv
from ast import literal_eval

# << 파일 경로 >>
filepath = "C:/00.Python/files_project_1/files_project_1_1/rank/"

filename_danceability_high = "danceability_high.csv"
filename_danceability_low = "danceability_low.csv"
filename_energy_high = "energy_high.csv"
filename_energy_low = "energy_low.csv"
filename_key_high = "key_high.csv"
filename_key_low = "key_low.csv"
filename_loudness_high = "loudness_high.csv"
filename_loudness_low = "loudness_low.csv"
filename_speechiness_high = "speechiness_high.csv"
filename_speechiness_low = "speechiness_low.csv"
filename_acousticness_high = "acousticness_high.csv"
filename_acousticness_low = "acousticness_low.csv"
filename_liveness_high = "liveness_high.csv"
filename_liveness_low = "liveness_low.csv"
filename_valence_high = "valence_high.csv"
filename_valence_low = "valence_low.csv"
filename_tempo_high = "tempo_high.csv"
filename_tempo_low = "tempo_low.csv"

# << 파일 데이터 읽어오기 >>
file = filepath + filename_valence_low
print(file)

# 방법 1
df = pd.read_csv(file, header=0, encoding='ANSI', sep=',')
# 방법 1로 파일 읽어올 때 에러 발생하는 경우 있음
# pandas.errors.ParserError: Error tokenizing data.

# 방법 2 -> 방법 1을 못 쓸 때만 사용
# filename_speechiness_low / filename_acousticness_low / filename_liveness_low / filename_valence_high / filename_tempo_high
# f = open(file, encoding='ANSI')
# reader = csv.reader(f)
# csv_list = []
# for row in reader:
#     csv_list.append(row)
# f.close()
# df = pd.DataFrame(csv_list)
#
# col1 = str(csv_list[0][0])
# col2 = str(csv_list[0][1])
# df.rename(columns={0:col1, 1:col2}, inplace=True)
#
# df.drop(2, axis=1, inplace=True)
# df.drop(3, axis=1, inplace=True)
# df.drop([0], axis=0, inplace=True)


# << 'spotify_genre' 컬럼 타입 변경 >>
# 문자열을 리스트로 변환
def str_to_list(x):
    try:
        if type(x) == str:
            return literal_eval(x)
        elif type(x) == list:
            return x
    except:
        print('오류 데이터 >>>', x)
        return None
        # 오류 데이터 있음
        # [australian children's music"]"
        # [children's music"]"


df['spotify_genre'] = df['spotify_genre'].apply(lambda x: str_to_list(x))


# << 결측값이 들어간 행 삭제하기 >>
print('---- 결측값 ----')
print(df.isnull().sum())
df = df.dropna()
# 값이 None인 행은 cnt_keyword 함수에서 오류 발생


# << 장르 키워드 카운트 >>
def cnt_keyword(series):
    return pd.Series([x for _list in series for x in _list])


cnt = cnt_keyword(df['spotify_genre']).value_counts().head(30)
print('---- 장르 키워드 카운트 ----')
print(cnt)

print('==============================================================================')
print('==============================================================================')


# ============================ 2차 확인 ============================
# 컬럼명 설정 후 DataFrame으로 변경
cnt = cnt.rename_axis('genre')
cnt = cnt.reset_index(name='count')
df2 = pd.DataFrame(cnt)


def split_str_to_list(x):
    try:
        if type(x) == str:
            return x.split(' ') # mellow gold -> ['mellow', 'gold']
        elif type(x) == list:
            return x
    except:
        print('오류 데이터 >>>', x)
        return None
        # 해당 값이 null값이거나 오류가 있을 때, None을 return 하기


df2['genre'] = df2['genre'].apply(lambda x: split_str_to_list(x))


# << 장르에 사용된 단어 카운트 >>
cnt2 = cnt_keyword(df2['genre']).value_counts()
pd.set_option('display.max_rows', None)
print('---- 장르에 사용된 단어 카운트 ----')
print(cnt2)
