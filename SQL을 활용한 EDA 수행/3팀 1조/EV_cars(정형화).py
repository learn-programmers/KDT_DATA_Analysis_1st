import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 데이터 불러오기
df = pd.read_csv('ev.csv', encoding='utf-8')

# MinMaxScaler 객체 생성
scaler = MinMaxScaler()

# 데이터 정규화
normalized_data = scaler.fit_transform(df)

# 정규화된 데이터 출력
normalized_df = pd.DataFrame(normalized_data, columns=df.columns)
print(normalized_df)

# 파일 만들기
normalized_df.to_csv('EV_Normalized.csv')