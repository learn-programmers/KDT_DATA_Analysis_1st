import pandas as pd
import csv

# << 파일 데이터 읽어오기 >>
file = "C:/00.Python/files_project_1/billboard_db.filterdata.csv"
# 방법 1 에러 발생 -> 방법 2 사용
# df = pd.read_csv(file, header=0, encoding='ANSI', sep=',')
f = open(file, encoding='ANSI')
reader = csv.reader(f)
csv_list = []
for row in reader:
    csv_list.append(row)
f.close()
df = pd.DataFrame(csv_list)
# print(df.head(5))
# print(len(df.columns))

# << 컬럼명 변경 >>
for i in range(0, len(df.columns)-1):
    col = str(csv_list[0][i])
    # print(col)
    df.rename(columns={i:col}, inplace=True)
# print(df.head(5))
# print(df.columns)
df.drop('index0', axis=1, inplace=True)
df.drop('spotify_genre', axis=1, inplace=True)
df.drop(14, axis=1, inplace=True)
df.drop([0], axis=0, inplace=True)


# << 최빈값 구하기 >>
df_mode = df.mode()
df_mode.drop([1], axis=0, inplace=True)
pd.set_option('display.max_columns', None)
print(df_mode)
