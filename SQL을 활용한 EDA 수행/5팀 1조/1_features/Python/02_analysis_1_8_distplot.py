import pandas as pd
import csv
import matplotlib.pyplot as plt
import seaborn as sns


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
# << 컬럼명 변경 >>
for i in range(0, len(df.columns)-1):
    col = str(csv_list[0][i])
    # print(col)
    df.rename(columns={i:col}, inplace=True)
df.drop('index0', axis=1, inplace=True)
df.drop('spotify_genre', axis=1, inplace=True)
df.drop(14, axis=1, inplace=True)
df.drop([0], axis=0, inplace=True)
df.rename(columns={'key0':'key', 'mode0':'mode'}, inplace=True)


# 문자열을 숫자로 변환
def str_to_float(x):
    try:
        if type(x) == str:
            return float(x)
        elif type(x) == float:
            return x
    except:
        return None


for i in range(0, len(df.columns)):
    col = df.columns[i]
    df[col] = df[col].apply(lambda x: str_to_float(x))


def show_plot(dataframe):
    arr_cols = df.columns
    try:
        for i in range(0, len(dataframe.columns) - 1):
            col_name = str(arr_cols[i])
            print("%s >>> Skewness: %f, Kurtosis: %f" % (col_name, dataframe[col_name].skew(), dataframe[col_name].kurt()))
            sns.distplot(dataframe[col_name], bins=60)
            # plt.xlim([-0.5, 1.5]) # energy
            # plt.xlim([-1, 2]) # valence
            plt.show()
    except IndexError:
        print('columns count is odd')


show_plot(df)