import pandas as pd
import csv
import matplotlib.pyplot as plt


# << 파일 데이터 읽어오기 >>
file = "C:/00.Python/files_project_1/files_project_1_2/filterdata_where_energy_lowerthan1.csv"
# 방법 1 에러 발생 -> 방법 2 사용
# df = pd.read_csv(file, header=0, encoding='ANSI', sep=',')
f = open(file, encoding='ANSI')
reader = csv.reader(f)
csv_list = []
for row in reader:
    csv_list.append(row)
f.close()
df = pd.DataFrame(csv_list)
# << 컬럼명 변경 >>
for i in range(0, len(df.columns)-1):
    col = str(csv_list[0][i])
    # print(col)
    df.rename(columns={i:col}, inplace=True)

df.drop('index0', axis=1, inplace=True)
df.rename(columns={'key0':'key', 'mode0':'mode'}, inplace=True)

# print(len(df.columns))
# print(df.columns)
if len(df.columns) == 14:
    df.drop(14, axis=1, inplace=True)


# 컬럼 타입 변경
from ast import literal_eval


def str_to_list(x):
    try:
        if type(x) == str:
            return literal_eval(x)
        elif type(x) == list:
            return x
    except:
        print('오류 데이터 >>>', x)
        return None


# 문자열을 float로 변환
def str_to_float(x):
    try:
        if type(x) == str:
            return float(x)
        elif type(x) == float:
            return x
    except:
        return None


# 문자열을 int로 변환
def str_to_int(x):
    try:
        if type(x) == str:
            return int(x)
        elif type(x) == int:
            return x
    except:
        return None


# 컬럼 타입 변경
for i in range(0, len(df.columns)-1):
    col = df.columns[i]
    if col == 'spotify_genre':
        df[col] = df[col].apply(lambda x: str_to_list(x))
    elif col == 'key' or col == 'mode' or col == 'time_signature':
        df[col] = df[col].apply(lambda x: str_to_int(x))
    else:
        df[col] = df[col].apply(lambda x: str_to_float(x))

# pd.set_option('display.max_columns', None)
# print(df.head(10))
# print(df.dtypes)
# breakpoint()

# << 결측값이 들어간 행 삭제하기 >>
# print('---- 결측값 ----')
# print(df.isnull().sum())
df = df.dropna()


# ------------------------------ scatter ---------------------------------------
# 특징 간의 상관관계 확인
energy = df['energy'].values.tolist()
loud = df['loudness'].values.tolist()
dance = df['danceability'].values.tolist()
acou = df['acousticness'].values.tolist()

# Positive
plt.scatter(loud, energy, alpha=0.3)
plt.xlabel('loudness', size = 10)
plt.ylabel('energy', size = 10)
plt.show()

plt.scatter(loud, dance, alpha=0.3)
plt.xlabel('loudness', size = 10)
plt.ylabel('danceability', size = 10)
plt.show()

plt.scatter(energy, dance, alpha=0.3)
plt.xlabel('energy', size = 10)
plt.ylabel('danceability', size = 10)
plt.show()

# Negative
plt.scatter(acou, energy, alpha=0.3)
plt.xlabel('acousticness', size = 10)
plt.ylabel('energy', size = 10)
plt.show()

plt.scatter(acou, loud, alpha=0.3)
plt.xlabel('acousticness', size = 10)
plt.ylabel('loudness', size = 10)
plt.show()

plt.scatter(acou, dance, alpha=0.3)
plt.xlabel('acousticness', size = 10)
plt.ylabel('danceability', size = 10)
plt.show()



