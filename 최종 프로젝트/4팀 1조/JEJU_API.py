import requests
import json
import pandas as pd

import pymysql
from sqlalchemy import create_engine


def JEJU_API(table_name,base_url,api_type):
    # MySQL 연결 설정
    host = 'da-4-1-db.c1widrnxppoz.ap-northeast-2.rds.amazonaws.com'  # 여기에 MySQL 서버의 호스트 주소 또는 IP 주소를 입력하세요
    port = 3306
    user = 'root'   
    password = '9juvtDV9gG&[!'
    database = 'JEJU_DB'

    your_key = '3tcj26trje2c3r2_p8tp8299jt8t2483'  
    params = {"number": 1, "limit": 100}
    # base_url="https://open.jejudatahub.net/api/proxy/Daaa1t3at3tt8a8DD3t55538t35Dab1t"
    startDate="20210101"
    endDate="20210301"
    if api_type==0:
        url = f"{base_url}/{your_key}"
    elif api_type==1:
        url = f"{base_url}/{your_key}?startDate={startDate}&endDate={endDate}"
    else:
        url = f"{base_url}/{your_key}?baseDate={startDate}"
    try:
        res = requests.get(url, params=params)
        res.raise_for_status()  # 네트워크 요청에서 예외 발생 시 예외 처리
        res_json = res.json()  # JSON 형식의 응답을 파이썬 객체로 변환
        totalCount = res_json['totCnt']
        items = res_json['data']

        while len(items) < totalCount:
            params['number'] += 1  # number 값을 1씩 증가시켜 추가 데이터를 요청
            tmp_res = requests.get(url, params=params)
            tmp_res.raise_for_status()  # 네트워크 요청에서 예외 발생 시 예외 처리
            tmp_json = tmp_res.json()  # JSON 형식의 응답을 파이썬 객체로 변환
            items += tmp_json['data']  # 새로운 데이터를 기존 리스트에 추가

        print(f"{totalCount} 중에 {len(items)} 만이 수집되었습니다.")
        
        ## 2. sql 서버로 데이터 업로드 및 데이터 수집 Double Check
        if totalCount==len(items):
            df=pd.DataFrame(items)
            
            # MySQL 연결
            conn = pymysql.connect(host=host, port=port, user=user, password=password, database=database)

            # 연결이 제대로 되었는지 확인
            if conn:
                print("MySQL에 성공적으로 연결되었습니다.")
                engine = create_engine(f'mysql+pymysql://{user}:{password}@{host}:{port}/{database}')
                df.to_sql(table_name, con=engine, if_exists='append', index=False)
                # 연결 종료 
                conn.close()
                print("Create Table Success.")
            else:
                print("MySQL 연결에 실패하였습니다.")

    except requests.exceptions.RequestException as e:
        print("네트워크 요청 중 오류가 발생했습니다:", e)
    except json.decoder.JSONDecodeError as e:
        print("JSON 디코딩 오류가 발생했습니다:", e)


table_name='일일_노선별_버스_이용자_정보'
base_url="https://open.jejudatahub.net/api/proxy/1t4tbb16b4t1t11tt11tDt1ab6114a1t"
JEJU_API(table_name,base_url,1) #api_type=1 : Date type On