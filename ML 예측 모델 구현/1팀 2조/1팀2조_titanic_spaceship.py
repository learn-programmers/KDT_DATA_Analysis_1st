#catboost 설치가 되어있지 않은 경우 실행
#!pip install catboost

#shap 설치가 되어있지 않은 경우 실행
#!pip install shap

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
from catboost import CatBoostClassifier

"""이 대회에서 여러분의 임무는 타이타닉호가 시공간 이상과 충돌하는 동안 승객이 다른 차원으로 이동했는지 여부를 예측하는 것입니다. 이러한 예측을 돕기 위해 함선의 손상된 컴퓨터 시스템에서 복구된 개인 기록 세트가 제공됩니다.

## **0. csv 입력 및 데이터 컬럼 확인**
"""

df = pd.read_csv('train.csv', delimiter=',')
df.head(5)

df.info()

"""**train.csv**- 약 2/3 (~8700명)의 승객에 대한 개인 기록으로, 훈련 데이터로 사용

1. **PassengerId** - 각 승객에 대한 고유 ID. 각 ID는 gggg_pp 형태를 가지며, 여기서 gggg는 승객이 함께 여행하는 그룹을 나타냄. pp는 그룹 내에서의 번호이며, 그룹에 있는 사람들은 대부분 가족으로 구성

2. **HomePlanet** - 승객이 출발한 행성으로, 일반적으로 그들의 영구 거주 행성

3. **CryoSleep** - 승객이 여행 기간 동안 중단된 애니메이션 상태에 놓이기로 선택했는지 여부를 나타냄.  CryoSleep 상태의 승객들은 객실에 제한

4. **Cabin** - 승객이 머무르는 객실 번호. deck/num/side 형태를 가지며, 여기서 side는 Port를 나타내는 P 또는 Starboard를 나타내는 S가 됨

5. **Destination** - 승객이 하차할 행성

6. **Age** - 승객의 나이

7. **VIP** - 승객이 여행 동안 특별한 VIP 서비스를 이용했는지 여부

8. **RoomService, FoodCourt, ShoppingMall, Spa, VRDeck** - 승객이 타이타닉 우주선의 다양한 고급 편의 시설에서 청구한 금액

13. **Name**- 승객의 이름

14. **Transported** - 승객이 다른 차원으로 이동했는지 여부. 이것이 목표로, 예측하려는 열

**test.csv** - 나머지 1/3 (~4300명)의 승객에 대한 개인 기록으로, 테스트 데이터로 사용되며 이 세트의 승객에 대해 Transported 값을 예측하는 것이 작업

**sample_submission.csv** - 올바른 형식의 제출 파일

**PassengerId** - 테스트 세트의 각 승객에 대한 ID

**Transported** - 목표로 각 데이터셋 설명

총 14개의 데이터 걸럼이 존재 마지막 컬럼 1개는 결과값 컬럼

info 결과 데이터셋 열의 수는 8693개

모든 컬럼에서 결측치가 발생하는 것을 확인
"""

df.describe()

df.describe(exclude=['float']).round(1)

"""## **1. 결측치 확인 및 제거**"""

missing_values = df.isnull().sum()

missing_values

df.dropna(inplace=True)

missing_values = df.isnull().sum()

missing_values

df.info()

"""## **2. 예측값 분포도**"""

# 그림 크기 설정
plt.figure(figsize=(6, 6))

# 원형 그래프
sns.set(style="whitegrid")
plt.pie(df['Transported'].value_counts(), labels=df['Transported'].value_counts().index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'), textprops={'fontsize': 15})
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title("Transported Distribution")
plt.show()

"""## **3. 각 컬럼별 이진 값 상관관계 분석**"""

plt.figure(figsize=(10, 6))
sns.set_theme()

sns.countplot(data=df, x='HomePlanet', hue='Transported', palette='pastel')

plt.title('Transported Ratio for HomePlanet')
plt.xlabel('HomePlanet')
plt.ylabel('Count')
plt.show()

"""False와 True의 비율을 비교했을 때, 가장 True가 많은 행성은 Europa이며, 가장 False가 많은 행성은 Earth입니다.

Mars는 True와 False의 비율이 크게 차이나지 않았습니다.
"""

plt.figure(figsize=(10, 6))
sns.set_theme()

sns.countplot(data=df, x='CryoSleep', hue='Transported', palette='pastel')

plt.title('Transported Ratio for CryoSleep')
plt.xlabel('CryoSleep')
plt.ylabel('Count')
plt.show()

"""**인사이트:**

승객이 CryoSleep, 즉 냉동수면 여부에 따른 결과가 매우 유의미했습니다.

냉동수면인 경우 True의 비율이 월등히 높았으며, 반면 냉동수면이 아닌 경우 False의 비율이 월등히 높았습니다.
"""

# A부터 G까지의 Cabin에 대한 따로운 그래프 그리기
cabin_categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

plt.figure(figsize=(15, 10))
sns.set_theme()

for idx, cabin_category in enumerate(cabin_categories, 1):
    # 'Cabin'이 앞자리가 현재 cabin_category인 경우에 대한 데이터프레임 생성
    cabin_category_df = df[df['Cabin'].str[0] == cabin_category]

    # subplot 설정
    plt.subplot(2, 4, idx)

    # 'Transported'에 대한 바 그래프 그리기
    sns.countplot(data=cabin_category_df, x='Transported', palette='pastel')

    plt.title(f'Transported Counts for Cabin {cabin_category}')
    plt.xlabel('Transported')
    plt.ylabel('Count')

plt.tight_layout()
plt.show()

"""**인사이트:**

객실 클래스의 앞자리에 따른 그래프를 분석해봤습니다.

True의 비율이 가장 높은 객실 클래스는 B와 C이며, 반면에 False의 비율이 가장 높은 객실 클래스는 D와 E입니다.
"""

# 'Cabin'의 맨 뒷자리가 'P' 또는 'S'인 경우에 대한 데이터프레임 생성
cabin_p_df = df[df['Cabin'].str[-1] == 'P']
cabin_s_df = df[df['Cabin'].str[-1] == 'S']

# 그림 크기 설정
plt.figure(figsize=(15, 6))

# 'Cabin'의 맨 뒷자리가 'P'인 경우에 대한 각 Transported의 비율을 나타내는 바 그래프 그리기
plt.subplot(1, 2, 1)
sns.countplot(data=cabin_p_df, x='Transported', palette='pastel')
plt.title('Distribution of Transported for Cabin Categories with Last Character P')
plt.xlabel('Transported')
plt.ylabel('Count')

# 'Cabin'의 맨 뒷자리가 'S'인 경우에 대한 각 Transported의 비율을 나타내는 바 그래프 그리기
plt.subplot(1, 2, 2)
sns.countplot(data=cabin_s_df, x='Transported', palette='pastel')
plt.title('Distribution of Transported for Cabin Categories with Last Character S')
plt.xlabel('Transported')
plt.ylabel('Count')

plt.tight_layout()
plt.show()

"""**인사이트:**

객실 클래스의 앞뒷자리에 따른 그래프를 분석해봤습니다.

뒷자리가 P인 경우 False의 비율이 높았으며 뒷자리가 S인 경우 True의 비율이 높았습니다.
"""

plt.figure(figsize=(10, 6))
sns.set_theme()

sns.countplot(data=df, x='Destination', hue='Transported', palette='pastel')

plt.title('Transported Ratio for Destination')
plt.xlabel('Destination')
plt.ylabel('Count')
plt.show()

"""**인사이트:**

도착지에 따른 Transported 결과를 살펴보았습니다.

TRAPPIST-1e로 향하는 경우 False의 비율이 높았으며, 반면 55 Cancri e로 향하는 경우 True의 비율이 가장 높았습니다.
"""

# Figure size
plt.figure(figsize=(10,4))

# Histogram
sns.histplot(data=df, x = 'Age', hue='Transported', binwidth=1, kde=True, palette='pastel')

# Aesthetics
plt.title('Age distribution')
plt.xlabel('Age (years)')

# 나이를 10단위로 범주화하여 새로운 열 'AgeGroup' 추가
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80], labels=['0s','10s','20s', '30s', '40s', '50s', '60s', '70s'])

plt.figure(figsize=(10, 6))
sns.set_theme()

sns.countplot(data=df, x='AgeGroup', hue='Transported', palette='pastel')

plt.title('Transported Ratio for Age Groups')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.show()

"""**인사이트:**

나이가 0-20대인 경우 True의 비율이 더 높았으며, 20-30대인 경우 False의 비율이 더 높았습니다.

40-70대는 True와 False의 비율에 큰 차이가 없었습니다.
"""

plt.figure(figsize=(10, 6))
sns.set_theme()

sns.countplot(data=df, x='VIP', hue='Transported', palette='pastel')

plt.title('Transported Ratio for VIP')
plt.xlabel('VIP')
plt.ylabel('Count')
plt.show()

"""**인사이트:**

VIP인 경우 False의 비율이 더 높았으며, VIP가 아닌 경우 True의 비율이 더 높았습니다.
"""

average_room_service = df['RoomService'].mean().round(1)

# 'RoomServiceAboveAverage' 열 생성: RoomService 값이 평균보다 높으면 True, 아니면 False
df['RoomServiceAverage'] = df['RoomService'] > average_room_service

plt.figure(figsize=(8, 6))
sns.set_theme()

sns.countplot(data=df, x='RoomServiceAverage', hue='Transported', palette='pastel')

plt.title('Transported Ratio for RoomService')
plt.xlabel('RoomService Average')
plt.ylabel('Count')
plt.show()

"""**인사이트:**

room service의 평균을 계산하여 비교를 하였습니다.

평균보다 높은 경우 False의 비율이 더 높았으며, 평균보다 낮은 경우 True의 비율이 더 높은 그래프를 그렸습니다.
"""

average_food_count = df['FoodCourt'].mean().round(1)

df['FoodCourtAverage'] = df['FoodCourt'] > average_food_count

plt.figure(figsize=(8, 6))
sns.set_theme()

sns.countplot(data=df, x='FoodCourtAverage', hue='Transported', palette='pastel')

plt.title('Transported Ratio for FoodCourt Average')
plt.xlabel('FoodCourt Average')
plt.ylabel('Count')
plt.show()

"""**인사이트:**

Food court의 평균보다 높은 경우 Flase의 비율이 근소하게 높았으며 낮은 경우 True의 비율이 근소하게 낮았습니다.

Food court는 크게 관여하지 않는 것으로 판단됩니다.
"""

average_shoppingmall = df['ShoppingMall'].mean().round(1)

df['ShoppingMallAverage'] = df['ShoppingMall'] > average_shoppingmall

plt.figure(figsize=(8, 6))
sns.set_theme()

sns.countplot(data=df, x='ShoppingMallAverage', hue='Transported', palette='pastel')

plt.title('Transported Ratio for ShoppingMall Average')
plt.xlabel('ShoppingMall Average')
plt.ylabel('Count')
plt.show()

"""**인사이트:**
shoppingmall의 평균에서는 평균보다 높은 경우 False의 비율이 높았으며, 평균보다 낮은 경우 True의 비율이 높았습니다.
"""

average_spa = df['Spa'].mean().round(1)

df['SpaAverage'] = df['Spa'] > average_spa

plt.figure(figsize=(8, 6))
sns.set_theme()

sns.countplot(data=df, x='SpaAverage', hue='Transported', palette='pastel')

plt.title('Transported Ratio for Spa Average')
plt.xlabel('Spa Average')
plt.ylabel('Count')
plt.show()

"""**인사이트:**
spa의 평균에서는 평균보다 높은 경우 False의 비율이 높았으며, 평균보다 낮은 경우 True의 비율이 높았습니다.
"""

average_vrdeck = df['VRDeck'].mean().round(1)

df['VRDeckAverage'] = df['VRDeck'] > average_vrdeck

plt.figure(figsize=(8, 6))
sns.set_theme()

sns.countplot(data=df, x='VRDeckAverage', hue='Transported', palette='pastel')

plt.title('Transported Ratio for VRDeck Average')
plt.xlabel('VRDeck Average')
plt.ylabel('Count')
plt.show()

"""**인사이트:**
VRDeck의 평균에서는 평균보다 높은 경우 False의 비율이 높았으며, 평균보다 낮은 경우 True의 비율이 높았습니다.
"""

# Expenditure features
exp_feats=['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

# Plot expenditure features
fig=plt.figure(figsize=(10,20))
for i, var_name in enumerate(exp_feats):
    # Left plot
    ax=fig.add_subplot(5,2,2*i+1)
    sns.histplot(data=df, x=var_name, axes=ax, bins=30, kde=False, hue='Transported')
    ax.set_title(var_name)

    # Right plot (truncated)
    ax=fig.add_subplot(5,2,2*i+2)
    sns.histplot(data=df, x=var_name, axes=ax, bins=30, kde=True, hue='Transported')
    plt.ylim([0,100])
    ax.set_title(var_name)
fig.tight_layout()  # Improves appearance a bit
plt.show()

"""**주의사항:**

대부분의 사람들은 돈을 아무것도 소비하지 않습니다 (왼쪽에서 확인할 수 있습니다).

소비 분포는 지수적으로 감소합니다 (오른쪽에서 확인할 수 있습니다).

이상치가 몇 개 있습니다.

이상으로 이동한 사람들은 일반적으로 덜 지출하는 경향이 있습니다.

RoomService, Spa 및 VRDeck는 FoodCourt 및 ShoppingMall과는 다른 분포를 가지고 있습니다. 이를 고급 vs 필수 편의시설로 생각할 수 있습니다.

**인사이트:**

5개의 편의시설 전체에서의 총 지출을 추적하는 새로운 기능을 생성합니다.

지출이 없는 경우를 나타내는 이진 특성을 만듭니다 (즉, 총 지출이 0인 경우).

왜곡을 감소시키기 위해 로그 변환을 수행합니다.

# 특성 엔지니어링
"""

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# New features - training set
train['Age_group']=np.nan
train.loc[train['Age']<=12,'Age_group']='Age_0-12'
train.loc[(train['Age']>12) & (train['Age']<18),'Age_group']='Age_13-17'
train.loc[(train['Age']>=18) & (train['Age']<=25),'Age_group']='Age_18-25'
train.loc[(train['Age']>25) & (train['Age']<=30),'Age_group']='Age_26-30'
train.loc[(train['Age']>30) & (train['Age']<=50),'Age_group']='Age_31-50'
train.loc[train['Age']>50,'Age_group']='Age_51+'

# New features - test set
test['Age_group']=np.nan
test.loc[test['Age']<=12,'Age_group']='Age_0-12'
test.loc[(test['Age']>12) & (test['Age']<18),'Age_group']='Age_13-17'
test.loc[(test['Age']>=18) & (test['Age']<=25),'Age_group']='Age_18-25'
test.loc[(test['Age']>25) & (test['Age']<=30),'Age_group']='Age_26-30'
test.loc[(test['Age']>30) & (test['Age']<=50),'Age_group']='Age_31-50'
test.loc[test['Age']>50,'Age_group']='Age_51+'

plt.figure(figsize=(10,4))
g=sns.countplot(data=train, x='Age_group', hue='Transported', order=['Age_0-12','Age_13-17','Age_18-25','Age_26-30','Age_31-50','Age_51+'])
plt.title('Age group distribution')

# 각 연령 그룹별 Transported 열의 비율 계산
transported_ratio = train.groupby('Age_group')['Transported'].mean()

# 각 연령 그룹별 Transported 열의 False 비율 계산
transported_false_ratio = 1 - transported_ratio

# 막대 그래프 그리기
plt.figure(figsize=(10, 6))
width = 0.35
transported_ratio.plot(kind='bar', color='skyblue', width=width, position=1, label='True')
transported_false_ratio.plot(kind='bar', color='lightgreen', width=width, position=0, label='False')
plt.title('Transported Ratio by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Transported Ratio')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

"""Age Gruop 중 Age_31-50에 인원이 가장 많이 분포하며, Age_13-17 인원이 가장 적다.

그룹별 Transported Ratio 경우, Age_0-12가 True & False가 가장 두드러지며, 이를 통해 해당 그룹이 Transported 할 경우가 높다.
"""

# Expenditure features
exp_feats=['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

# New features - training set
train['Expenditure']=train[exp_feats].sum(axis=1)
train['No_spending']=(train['Expenditure']==0).astype(int)

# New features - test set
test['Expenditure']=test[exp_feats].sum(axis=1)
test['No_spending']=(test['Expenditure']==0).astype(int)

# Plot distribution of new features
fig=plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
sns.histplot(data=train, x='Expenditure', hue='Transported', bins=200)
plt.title('Total expenditure (truncated)')
plt.ylim([0,200])
plt.xlim([0,20000])

plt.subplot(1,2,2)
sns.countplot(data=train, x='No_spending', hue='Transported')
plt.title('No spending indicator')
fig.tight_layout()

""""Transported"에 대한 소비 사용 패턴을 보았을 때, 소비하지 않는 인원이 "Transported" 할 경우가 높다.

따라서 "Transported"할 경우를 줄이기 위해서는 소비 사용을 권장해야 하며, "Transported"하는 인원들의 소비 금액은 대부분 대략 500~2500사이로 추정된다.
"""

# New feature - Group
train['Group'] = train['PassengerId'].apply(lambda x: x.split('_')[0]).astype(int)
test['Group'] = test['PassengerId'].apply(lambda x: x.split('_')[0]).astype(int)

# New feature - Group size
train['Group_size']=train['Group'].map(lambda x: pd.concat([train['Group'], test['Group']]).value_counts()[x])
test['Group_size']=test['Group'].map(lambda x: pd.concat([train['Group'], test['Group']]).value_counts()[x])

# New feature
train['Solo']=(train['Group_size']==1).astype(int)
test['Solo']=(test['Group_size']==1).astype(int)

# Plot distribution of new features
plt.figure(figsize=(20,4))
plt.subplot(1,2,1)
sns.histplot(data=train, x='Group', hue='Transported', binwidth=1)
plt.title('Group')

plt.subplot(1,2,2)
sns.countplot(data=train, x='Group_size', hue='Transported')
plt.title('Group size')
fig.tight_layout()

# New feature distribution
plt.figure(figsize=(10,4))
sns.countplot(data=train, x='Solo', hue='Transported')
plt.title('Passenger travelling solo or not')
plt.ylim([0,3000])

"""Group size 작을수록 인원이 많이 분포하며, "Transported"의 경우에는 Group size가 클 수록 Transported 비율이 높아짐을 알 수 있다.

Group size를 통해 추출된 "Solo"라는 새로운 컬럼에서 우주선 탑승 시에 동승의 여부에 따른 Transported 비율을 알 수 있다.
"""

# Replace NaN's with outliers for now (so we can split feature)
train['Cabin'].fillna('Z/9999/Z', inplace=True)
test['Cabin'].fillna('Z/9999/Z', inplace=True)

# New features - training set
train['Cabin_deck'] = train['Cabin'].apply(lambda x: x.split('/')[0])
train['Cabin_number'] = train['Cabin'].apply(lambda x: x.split('/')[1]).astype(int)
train['Cabin_side'] = train['Cabin'].apply(lambda x: x.split('/')[2])

# New features - test set
test['Cabin_deck'] = test['Cabin'].apply(lambda x: x.split('/')[0])
test['Cabin_number'] = test['Cabin'].apply(lambda x: x.split('/')[1]).astype(int)
test['Cabin_side'] = test['Cabin'].apply(lambda x: x.split('/')[2])

# Put Nan's back in (we will fill these later)
train.loc[train['Cabin_deck']=='Z', 'Cabin_deck']=np.nan
train.loc[train['Cabin_number']==9999, 'Cabin_number']=np.nan
train.loc[train['Cabin_side']=='Z', 'Cabin_side']=np.nan
test.loc[test['Cabin_deck']=='Z', 'Cabin_deck']=np.nan
test.loc[test['Cabin_number']==9999, 'Cabin_number']=np.nan
test.loc[test['Cabin_side']=='Z', 'Cabin_side']=np.nan

# Drop Cabin (we don't need it anymore)
train.drop('Cabin', axis=1, inplace=True)
test.drop('Cabin', axis=1, inplace=True)

# New features - training set
train['Cabin_region1']=(train['Cabin_number']<300).astype(int)   # one-hot encoding
train['Cabin_region2']=((train['Cabin_number']>=300) & (train['Cabin_number']<600)).astype(int)
train['Cabin_region3']=((train['Cabin_number']>=600) & (train['Cabin_number']<900)).astype(int)
train['Cabin_region4']=((train['Cabin_number']>=900) & (train['Cabin_number']<1200)).astype(int)
train['Cabin_region5']=((train['Cabin_number']>=1200) & (train['Cabin_number']<1500)).astype(int)
train['Cabin_region6']=((train['Cabin_number']>=1500) & (train['Cabin_number']<1800)).astype(int)
train['Cabin_region7']=(train['Cabin_number']>=1800).astype(int)

# New features - test set
test['Cabin_region1']=(test['Cabin_number']<300).astype(int)   # one-hot encoding
test['Cabin_region2']=((test['Cabin_number']>=300) & (test['Cabin_number']<600)).astype(int)
test['Cabin_region3']=((test['Cabin_number']>=600) & (test['Cabin_number']<900)).astype(int)
test['Cabin_region4']=((test['Cabin_number']>=900) & (test['Cabin_number']<1200)).astype(int)
test['Cabin_region5']=((test['Cabin_number']>=1200) & (test['Cabin_number']<1500)).astype(int)
test['Cabin_region6']=((test['Cabin_number']>=1500) & (test['Cabin_number']<1800)).astype(int)
test['Cabin_region7']=(test['Cabin_number']>=1800).astype(int)

# Plot distribution of new features
fig=plt.figure(figsize=(10,12))
plt.subplot(3,1,1)
sns.countplot(data=train, x='Cabin_deck', hue='Transported', order=['A','B','C','D','E','F','G','T'])
plt.title('Cabin deck')

plt.subplot(3,1,2)
sns.histplot(data=train, x='Cabin_number', hue='Transported',binwidth=20)
plt.vlines(300, ymin=0, ymax=200, color='black')
plt.vlines(600, ymin=0, ymax=200, color='black')
plt.vlines(900, ymin=0, ymax=200, color='black')
plt.vlines(1200, ymin=0, ymax=200, color='black')
plt.vlines(1500, ymin=0, ymax=200, color='black')
plt.vlines(1800, ymin=0, ymax=200, color='black')
plt.title('Cabin number')
plt.xlim([0,2000])

plt.subplot(3,1,3)
sns.countplot(data=train, x='Cabin_side', hue='Transported')
plt.title('Cabin side')
fig.tight_layout()

# Plot distribution of new features
plt.figure(figsize=(10,4))
train['Cabin_regions_plot']=(train['Cabin_region1']+2*train['Cabin_region2']+3*train['Cabin_region3']+4*train['Cabin_region4']+5*train['Cabin_region5']+6*train['Cabin_region6']+7*train['Cabin_region7']).astype(int)
sns.countplot(data=train, x='Cabin_regions_plot', hue='Transported')
plt.title('Cabin regions')
train.drop('Cabin_regions_plot', axis=1, inplace=True)

"""Cabin이라는 승객이 머무르는 객실의 컬럼과 Transported 관계
- Cabin_num "300"기준으로 그룹화
- "F" & "G"에 가장 많은 승객이 있으며, 그 중에서는 Cabin_regions '1'에 많은 인원이 있다.
- "Transported" 관점으로 보았을 때 "G" 객실에서 Cabin_regions '1', 좌현에 머무르는 객실의 승객일 경우 높은 확률로 "Transported"할 것이라 추측된다.
- "Transported"를 줄이기 위해 위에 말한 조건들을 피한다면 줄일 수 있지 않을까?
"""

# Replace NaN's with outliers for now (so we can split feature)
train['Name'].fillna('Unknown Unknown', inplace=True)
test['Name'].fillna('Unknown Unknown', inplace=True)

# New feature - Surname
train['Surname']=train['Name'].str.split().str[-1]
test['Surname']=test['Name'].str.split().str[-1]

# New feature - Family size
train['Family_size']=train['Surname'].map(lambda x: pd.concat([train['Surname'],test['Surname']]).value_counts()[x])
test['Family_size']=test['Surname'].map(lambda x: pd.concat([train['Surname'],test['Surname']]).value_counts()[x])

# Put Nan's back in (we will fill these later)
train.loc[train['Surname']=='Unknown','Surname']=np.nan
train.loc[train['Family_size']>100,'Family_size']=np.nan
test.loc[test['Surname']=='Unknown','Surname']=np.nan
test.loc[test['Family_size']>100,'Family_size']=np.nan

# Drop name (we don't need it anymore)
train.drop('Name', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)

# New feature distribution
plt.figure(figsize=(12,4))
sns.countplot(data=train, x='Family_size', hue='Transported')
plt.title('Family size')

""""Name" 컬럼에서 가족일 경우 성이 동일할 거라 생각하여 "Family"라는 새로운 컬럼을 생성

이를 통한 승객의 가족 단위를 파악, 5~6인 단체 승객인원일 경우에 "Transported"할 경우가 높다는 것을 알 수 있다.
"""

# Labels and features
y=train['Transported'].copy().astype(int)
X=train.drop('Transported', axis=1).copy()

# Concatenate dataframes
data=pd.concat([X, test], axis=0).reset_index(drop=True)

# Columns with missing values
na_cols=data.columns[data.isna().any()].tolist()

# Missing values summary
mv=pd.DataFrame(data[na_cols].isna().sum(), columns=['Number_missing'])
mv['Percentage_missing']=np.round(100*mv['Number_missing']/len(data),2)
mv

# Joint distribution of Group and HomePlanet
GHP_gb=data.groupby(['Group','HomePlanet'])['HomePlanet'].size().unstack().fillna(0)
GHP_gb.head()

# Missing values before
HP_bef=data['HomePlanet'].isna().sum()

# Passengers with missing HomePlanet and in a group with known HomePlanet
GHP_index=data[data['HomePlanet'].isna()][(data[data['HomePlanet'].isna()]['Group']).isin(GHP_gb.index)].index

# Fill corresponding missing values
data.loc[GHP_index,'HomePlanet']=data.iloc[GHP_index,:]['Group'].map(lambda x: GHP_gb.idxmax(axis=1)[x])

# Print number of missing values left
print('#HomePlanet missing values before:',HP_bef)
print('#HomePlanet missing values after:',data['HomePlanet'].isna().sum())

# Joint distribution of CabinDeck and HomePlanet
CDHP_gb=data.groupby(['Cabin_deck','HomePlanet'])['HomePlanet'].size().unstack().fillna(0)

# Missing values before
HP_bef=data['HomePlanet'].isna().sum()

# Decks A, B, C or T came from Europa
data.loc[(data['HomePlanet'].isna()) & (data['Cabin_deck'].isin(['A', 'B', 'C', 'T'])), 'HomePlanet']='Europa'

# Deck G came from Earth
data.loc[(data['HomePlanet'].isna()) & (data['Cabin_deck']=='G'), 'HomePlanet']='Earth'

# Print number of missing values left
print('#HomePlanet missing values before:',HP_bef)
print('#HomePlanet missing values after:',data['HomePlanet'].isna().sum())

# Joint distribution of Surname and HomePlanet
SHP_gb=data.groupby(['Surname','HomePlanet'])['HomePlanet'].size().unstack().fillna(0)

# Missing values before
HP_bef=data['HomePlanet'].isna().sum()

# Passengers with missing HomePlanet and in a family with known HomePlanet
SHP_index=data[data['HomePlanet'].isna()][(data[data['HomePlanet'].isna()]['Surname']).isin(SHP_gb.index)].index

# Fill corresponding missing values
data.loc[SHP_index,'HomePlanet']=data.iloc[SHP_index,:]['Surname'].map(lambda x: SHP_gb.idxmax(axis=1)[x])

# Print number of missing values left
print('#HomePlanet missing values before:',HP_bef)
print('#HomePlanet missing values after:',data['HomePlanet'].isna().sum())

# Only 10 HomePlanet missing values left - let's look at them
data[data['HomePlanet'].isna()][['PassengerId','HomePlanet','Destination']]

# Joint distribution of HomePlanet and Destination
HPD_gb=data.groupby(['HomePlanet','Destination'])['Destination'].size().unstack().fillna(0)

# Missing values before
HP_bef=data['HomePlanet'].isna().sum()

# Fill remaining HomePlanet missing values with Earth (if not on deck D) or Mars (if on Deck D)
data.loc[(data['HomePlanet'].isna()) & ~(data['Cabin_deck']=='D'), 'HomePlanet']='Earth'
data.loc[(data['HomePlanet'].isna()) & (data['Cabin_deck']=='D'), 'HomePlanet']='Mars'

# Print number of missing values left
print('#HomePlanet missing values before:',HP_bef)
print('#HomePlanet missing values after:',data['HomePlanet'].isna().sum())

# Missing values before
D_bef=data['Destination'].isna().sum()

# Fill missing Destination values with mode
data.loc[(data['Destination'].isna()), 'Destination']='TRAPPIST-1e'

# Print number of missing values left
print('#Destination missing values before:',D_bef)
print('#Destination missing values after:',data['Destination'].isna().sum())

# Joint distribution of Group and Surname
GSN_gb=data[data['Group_size']>1].groupby(['Group','Surname'])['Surname'].size().unstack().fillna(0)

# Missing values before
SN_bef=data['Surname'].isna().sum()

# Passengers with missing Surname and in a group with known majority Surname
GSN_index=data[data['Surname'].isna()][(data[data['Surname'].isna()]['Group']).isin(GSN_gb.index)].index

# Fill corresponding missing values
data.loc[GSN_index,'Surname']=data.iloc[GSN_index,:]['Group'].map(lambda x: GSN_gb.idxmax(axis=1)[x])

# Print number of missing values left
print('#Surname missing values before:',SN_bef)
print('#Surname missing values after:',data['Surname'].isna().sum())

# Replace NaN's with outliers (so we can use map)
data['Surname'].fillna('Unknown', inplace=True)

# Update family size feature
data['Family_size']=data['Surname'].map(lambda x: data['Surname'].value_counts()[x])

# Put NaN's back in place of outliers
data.loc[data['Surname']=='Unknown','Surname']=np.nan

# Say unknown surname means no family
data.loc[data['Family_size']>100,'Family_size']=0

# Joint distribution of Group and Cabin features
GCD_gb=data[data['Group_size']>1].groupby(['Group','Cabin_deck'])['Cabin_deck'].size().unstack().fillna(0)
GCN_gb=data[data['Group_size']>1].groupby(['Group','Cabin_number'])['Cabin_number'].size().unstack().fillna(0)
GCS_gb=data[data['Group_size']>1].groupby(['Group','Cabin_side'])['Cabin_side'].size().unstack().fillna(0)

# Missing values before
CS_bef=data['Cabin_side'].isna().sum()

# Passengers with missing Cabin side and in a group with known Cabin side
GCS_index=data[data['Cabin_side'].isna()][(data[data['Cabin_side'].isna()]['Group']).isin(GCS_gb.index)].index

# Fill corresponding missing values
data.loc[GCS_index,'Cabin_side']=data.iloc[GCS_index,:]['Group'].map(lambda x: GCS_gb.idxmax(axis=1)[x])

# Print number of missing values left
print('#Cabin_side missing values before:',CS_bef)
print('#Cabin_side missing values after:',data['Cabin_side'].isna().sum())

# Joint distribution of Surname and Cabin side
SCS_gb=data[data['Group_size']>1].groupby(['Surname','Cabin_side'])['Cabin_side'].size().unstack().fillna(0)

# Ratio of sides
SCS_gb['Ratio']=SCS_gb['P']/(SCS_gb['P']+SCS_gb['S'])

# Missing values before
CS_bef=data['Cabin_side'].isna().sum()

# Drop ratio column
SCS_gb.drop('Ratio', axis=1, inplace=True)

# Passengers with missing Cabin side and in a family with known Cabin side
SCS_index=data[data['Cabin_side'].isna()][(data[data['Cabin_side'].isna()]['Surname']).isin(SCS_gb.index)].index

# Fill corresponding missing values
data.loc[SCS_index,'Cabin_side']=data.iloc[SCS_index,:]['Surname'].map(lambda x: SCS_gb.idxmax(axis=1)[x])

# Drop surname (we don't need it anymore)
data.drop('Surname', axis=1, inplace=True)

# Print number of missing values left
print('#Cabin_side missing values before:',CS_bef)
print('#Cabin_side missing values after:',data['Cabin_side'].isna().sum())

# Value counts
data['Cabin_side'].value_counts()

# Missing values before
CS_bef=data['Cabin_side'].isna().sum()

# Fill remaining missing values with outlier
data.loc[data['Cabin_side'].isna(),'Cabin_side']='Z'

# Print number of missing values left
print('#Cabin_side missing values before:',CS_bef)
print('#Cabin_side missing values after:',data['Cabin_side'].isna().sum())

# Missing values before
CD_bef=data['Cabin_deck'].isna().sum()

# Passengers with missing Cabin deck and in a group with known majority Cabin deck
GCD_index=data[data['Cabin_deck'].isna()][(data[data['Cabin_deck'].isna()]['Group']).isin(GCD_gb.index)].index

# Fill corresponding missing values
data.loc[GCD_index,'Cabin_deck']=data.iloc[GCD_index,:]['Group'].map(lambda x: GCD_gb.idxmax(axis=1)[x])

# Print number of missing values left
print('#Cabin_deck missing values before:',CD_bef)
print('#Cabin_deck missing values after:',data['Cabin_deck'].isna().sum())

# Joint distribution
data.groupby(['HomePlanet','Destination','Solo','Cabin_deck'])['Cabin_deck'].size().unstack().fillna(0)

# Missing values before
CD_bef=data['Cabin_deck'].isna().sum()

# Fill missing values using the mode
na_rows_CD=data.loc[data['Cabin_deck'].isna(),'Cabin_deck'].index
data.loc[data['Cabin_deck'].isna(),'Cabin_deck']=data.groupby(['HomePlanet','Destination','Solo'])['Cabin_deck'].transform(lambda x: x.fillna(pd.Series.mode(x)[0]))[na_rows_CD]

# Print number of missing values left
print('#Cabin_deck missing values before:',CD_bef)
print('#Cabin_deck missing values after:',data['Cabin_deck'].isna().sum())

# Missing values before
CN_bef=data['Cabin_number'].isna().sum()

# Extrapolate linear relationship on a deck by deck basis
for deck in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
    # Features and labels
    X_CN=data.loc[~(data['Cabin_number'].isna()) & (data['Cabin_deck']==deck),'Group']
    y_CN=data.loc[~(data['Cabin_number'].isna()) & (data['Cabin_deck']==deck),'Cabin_number']
    X_test_CN=data.loc[(data['Cabin_number'].isna()) & (data['Cabin_deck']==deck),'Group']

    # Linear regression
    model_CN=LinearRegression()
    model_CN.fit(X_CN.values.reshape(-1, 1), y_CN)
    preds_CN=model_CN.predict(X_test_CN.values.reshape(-1, 1))

    # Fill missing values with predictions
    data.loc[(data['Cabin_number'].isna()) & (data['Cabin_deck']==deck),'Cabin_number']=preds_CN.astype(int)

# Print number of missing values left
print('#Cabin_number missing values before:',CN_bef)
print('#Cabin_number missing values after:',data['Cabin_number'].isna().sum())

# One-hot encode cabin regions
data['Cabin_region1']=(data['Cabin_number']<300).astype(int)
data['Cabin_region2']=((data['Cabin_number']>=300) & (data['Cabin_number']<600)).astype(int)
data['Cabin_region3']=((data['Cabin_number']>=600) & (data['Cabin_number']<900)).astype(int)
data['Cabin_region4']=((data['Cabin_number']>=900) & (data['Cabin_number']<1200)).astype(int)
data['Cabin_region5']=((data['Cabin_number']>=1200) & (data['Cabin_number']<1500)).astype(int)
data['Cabin_region6']=((data['Cabin_number']>=1500) & (data['Cabin_number']<1800)).astype(int)
data['Cabin_region7']=(data['Cabin_number']>=1800).astype(int)

data['VIP'].value_counts()

# Missing values before
V_bef=data['VIP'].isna().sum()

# Fill missing values with mode
data.loc[data['VIP'].isna(),'VIP']=False

# Print number of missing values left
print('#VIP missing values before:',V_bef)
print('#VIP missing values after:',data['VIP'].isna().sum())

# Joint distribution
data.groupby(['HomePlanet','No_spending','Solo','Cabin_deck'])['Age'].median().unstack().fillna(0)

# Missing values before
A_bef=data[exp_feats].isna().sum().sum()

# Fill missing values using the median
na_rows_A=data.loc[data['Age'].isna(),'Age'].index
data.loc[data['Age'].isna(),'Age']=data.groupby(['HomePlanet','No_spending','Solo','Cabin_deck'])['Age'].transform(lambda x: x.fillna(x.median()))[na_rows_A]

# Print number of missing values left
print('#Age missing values before:',A_bef)
print('#Age missing values after:',data['Age'].isna().sum())

# Update age group feature
data.loc[data['Age']<=12,'Age_group']='Age_0-12'
data.loc[(data['Age']>12) & (data['Age']<18),'Age_group']='Age_13-17'
data.loc[(data['Age']>=18) & (data['Age']<=25),'Age_group']='Age_18-25'
data.loc[(data['Age']>25) & (data['Age']<=30),'Age_group']='Age_26-30'
data.loc[(data['Age']>30) & (data['Age']<=50),'Age_group']='Age_31-50'
data.loc[data['Age']>50,'Age_group']='Age_51+'

# Joint distribution
data.groupby(['No_spending','CryoSleep'])['CryoSleep'].size().unstack().fillna(0)

# Missing values before
CSL_bef=data['CryoSleep'].isna().sum()

# Fill missing values using the mode
na_rows_CSL=data.loc[data['CryoSleep'].isna(),'CryoSleep'].index
data.loc[data['CryoSleep'].isna(),'CryoSleep']=data.groupby(['No_spending'])['CryoSleep'].transform(lambda x: x.fillna(pd.Series.mode(x)[0]))[na_rows_CSL]

# Print number of missing values left
print('#CryoSleep missing values before:',CSL_bef)
print('#CryoSleep missing values after:',data['CryoSleep'].isna().sum())

print('Maximum expenditure of passengers in CryoSleep:',data.loc[data['CryoSleep']==True,exp_feats].sum(axis=1).max())

# Missing values before
E_bef=data[exp_feats].isna().sum().sum()

# CryoSleep has no expenditure
for col in exp_feats:
    data.loc[(data[col].isna()) & (data['CryoSleep']==True), col]=0

# Print number of missing values left
print('#Expenditure missing values before:',E_bef)
print('#Expenditure missing values after:',data[exp_feats].isna().sum().sum())

# Joint distribution
data.groupby(['HomePlanet','Solo','Age_group'])['Expenditure'].mean().unstack().fillna(0)

# Missing values before
E_bef=data[exp_feats].isna().sum().sum()

# Fill remaining missing values using the median
for col in exp_feats:
    na_rows=data.loc[data[col].isna(),col].index
    data.loc[data[col].isna(),col]=data.groupby(['HomePlanet','Solo','Age_group'])[col].transform(lambda x: x.fillna(x.mean()))[na_rows]

# Print number of missing values left
print('#Expenditure missing values before:',E_bef)
print('#Expenditure missing values after:',data[exp_feats].isna().sum().sum())

# Update expenditure and no_spending
data['Expenditure']=data[exp_feats].sum(axis=1)
data['No_spending']=(data['Expenditure']==0).astype(int)

data.isna().sum()

"""# 데이터 전처리

##  **1. 학습 / 훈련 데이터 세팅**
"""

# 특성 엔지니어링을 위해 합쳤던 학습 데이터와 훈련 데이터를 분리
# train과 test의 각 PassnegerId를 기준으로 각각 X와 X_test에 데이터를 할당
X=data[data['PassengerId'].isin(train['PassengerId'].values)].copy()
X_test=data[data['PassengerId'].isin(test['PassengerId'].values)].copy()

# 모델 학습에 불 필요한 컬럼을 제거
X.drop(['PassengerId', 'Group', 'Group_size', 'Age_group', 'Cabin_number'], axis=1, inplace=True)
X_test.drop(['PassengerId', 'Group', 'Group_size', 'Age_group', 'Cabin_number'], axis=1, inplace=True)

"""## **2. 데이터 스케일링**

주어진 열(컬럼) 목록에 대해 로그 변환을 통해 데이터의 분포를 조정하여 모델 학습에 도움이 되도록 조정

숫자형 데이터에 대해 평균이 0이고 분산이 1인 표준 스케일링을 수행
숫자 데이터를 조정하여 학습 알고리즘이 더 잘 작동하도록 함

범주형 데이터에 대해 OneHotEncoder를 통한 인코딩을 수행
범주형 변수를 수치형으로 변환하여 모델 학습에 사용할 수 있도록 함

"""

# 로그 변환 적용
for col in ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Expenditure']:
    X[col] = np.log(1 + X[col])
    X_test[col] = np.log(1 + X_test[col])

# 숫자 및 범주형 열 식별
numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]
categorical_cols = [cname for cname in X.columns if X[cname].dtype == "object"]

# 숫자 데이터를 평균 = 0 및 분산 = 1 로 스케일 조정
numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])

# 범주형 데이터를 OneHotEncoder를 사용한 인코딩
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse=False))])

# 전처리 결합
ct = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)],
    remainder='passthrough')

# 전처리 적용
X = ct.fit_transform(X)
X_test = ct.transform(X_test)

# 데이터를 훈련 세트와 검증 세트로 분할
X_train, X_valid, y_train, y_valid = train_test_split(X,y,stratify=y,train_size=0.8,test_size=0.2,random_state=0)

"""# 모델 학습
1. RandomForest, LogisticRegression 두 모델을 기본으로 사용
2. GridSearch 또는 RandomSearch를 통해 최적 파라미터를 탐색
3. 최적 파라미터를 튜닝해 데이터 학습 후 정확도 점수 확인
4. 정확도 점수가 높게 나온 모델을 통해 최적 파라미터 탐색 반복

추가로 Kaggle에서 공개된 Code를 참고해 성능이 가장 좋게 나타났던 CatBoostClassifier 모델 또한 사용 후 결과 확인

## **1. RandomForest**
"""

# RandomForest의 파라미터 정의
rf_param_grid = {
    'n_estimators': [300, 400, 500],
    'max_depth': [20, 30],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['auto', 'sqrt', 'log2', 0.5, None]
}

# Grid Search 객체 생성
rf_grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=0), param_grid=rf_param_grid, cv=5)

# 최적 모델 탐색을 위한 GridSearch 진행
rf_grid_search.fit(X_train, y_train)

# 최적 파라미터와 해당 파라미터의 점수 출력
rf_best_params = rf_grid_search.best_params_
rf_best_score = rf_grid_search.best_score_
print("Best parameters:", rf_best_params)
print("Best score:", rf_best_score)

# 최적의 하이퍼파라미터로 랜덤 포레스트 모델 초기화
rf_best_params = rf_grid_search.best_params_
rf_best_model = RandomForestClassifier(**rf_best_params)

# 전체 훈련 데이터에 대해 모델 학습
rf_best_model.fit(X_train, y_train)

# rf_best_result에 예측 결과 초기화
rf_best_result = rf_best_model.predict(X_test)

# 예측 결과를 Kaggle 제출 양식에 맞게 변환
rf_submission = pd.DataFrame({'PassengerId': test['PassengerId']})
rf_bs_bool = rf_best_result.astype(bool)
rf_submission['Transported'] = rf_bs_bool

# 제출 양식에 맞춘 데이터를 csv 파일로 Export
rf_submission.to_csv('RandomForest_result.csv', index=False)

"""## **2. LogisticRegression**"""

# LogisticRegression의 파라미터 정의
lr_param_grid = {
    'penalty': ['l1', 'l2'],
    'C': [0.001, 0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'saga'],
    'max_iter': [100, 200, 300],
    'class_weight': [None, 'balanced'],
    'multi_class': ['ovr', 'multinomial']
}

# Grid Search 객체 생성
lr_grid_search = GridSearchCV(estimator=LogisticRegression(), param_grid=lr_param_grid, cv=5)

# 최적 모델 탐색을 위한 GridSearch 진행
lr_grid_search.fit(X_train, y_train)

# 최적 파라미터와 해당 파라미터의 점수 출력
lr_best_params = lr_grid_search.best_params_
lr_best_score = lr_grid_search.best_score_
print("Best parameters:", lr_best_params)
print("Best score:", lr_best_score)

# 최적의 하이퍼파라미터로 랜덤 포레스트 모델 초기화
lr_best_params = lr_grid_search.best_params_
lr_best_model = LogisticRegression(**lr_best_params)

# 전체 훈련 데이터에 대해 모델 학습
lr_best_model.fit(X_train, y_train)

# lr_best_result에 예측 결과 초기화
lr_best_result = lr_best_model.predict(X_test)

# 예측 결과를 Kaggle 제출 양식에 맞게 변환
lr_submission = pd.DataFrame({'PassengerId': test['PassengerId']})
lr_bs_bool = lr_best_result.astype(bool)
lr_submission['Transported'] = lr_bs_bool

# 제출 양식에 맞춘 데이터를 csv 파일로 Export
lr_submission.to_csv('LogisticRegression_result.csv', index=False)

"""## **3. CatBoostClassifier**"""

# CatBoostClassifier의 파라미터 정의
cb_param_dist = {
    'depth': [4, 5, 6, 7, 8, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    'l2_leaf_reg': [1, 3, 5, 7, 9]
}

# Random Search 객체 생성
cb_random_search = RandomizedSearchCV(CatBoostClassifier(), param_distributions=cb_param_dist, n_iter=10, cv=5, scoring='accuracy', random_state=42)

# 최적 모델 탐색을 위한 Random Search 진행
cb_random_search.fit(X_train, y_train)

# 최적 파라미터와 해당 파라미터의 점수 출력
cb_best_params = cb_random_search.best_params_
cb_best_score = cb_random_search.best_score_
print("Best parameters:", cb_best_params)
print("Best score:", cb_best_score)

# 최적의 하이퍼파라미터로  CatBoostClassifier 모델 초기화
cb_best_params = cb_random_search.best_params_
cb_best_model = CatBoostClassifier(**cb_best_params)

# 전체 훈련 데이터에 대해 모델 학습
cb_best_model.fit(X_train, y_train)

# cb_best_result에 예측 결과 초기화
cb_best_result = cb_best_model.predict(X_test)

# 예측 결과를 Kaggle 제출 양식에 맞게 변환
cb_submission = pd.DataFrame({'PassengerId': test['PassengerId']})
cb_bs_bool = cb_best_result.astype(bool)
cb_submission['Transported'] = cb_bs_bool

# 제출 양식에 맞춘 데이터를 csv 파일로 Export
cb_submission.to_csv('CatBoostClassifier_result.csv', index=False)

"""# 모델 검증
- ROC Curve와 AUC를 계산 후 시각화
"""

# 모델 리스트 정의
models = [rf_best_model, lr_best_model, cb_best_model]

# 모델명 리스트 정의
model_names = ['RandomForest', 'LogisticRegression', 'CatBoost']

# 모델별 결과 변수 리스트 정의
results = [rf_best_result, lr_best_result, cb_best_result]

for model, name in zip(models, model_names):
    model.fit(X_train, y_train)
    y_score = model.predict_proba(X_valid)[:, 1]
    fpr, tpr, _ = roc_curve(y_valid, y_score)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.legend(loc="lower right", fontsize='large', title='Model')
plt.show()

"""# 결과

## **1. ROC Curve롤 통한 검증 결과**

![image.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAk0AAAHPCAIAAADBCHlBAAAgAElEQVR4AeydCXRURb7/UZ5/Pfz1P8+jZ57nzDi+ecd3u0MSkrCEHQZZdCAq4IAIYRFGljCsskRlkT0IgXFASAQGEJewKaugBBHQgILsskjI1ul03046pLuTdNbuP50KleIu1ZVebt/u/vXJ0bpVdbfPraovVfX7VbVwwg8IAAEgAASAQOgSaBG6rwZvBgSAABAAAkDACToHhQAIAAEgAARCmQDoXCh/XXg3IAAEgAAQAJ2DMgAEgAAQAAKhTAB0LpS/LrwbEAACQAAIgM5BGQACQAAIAIFQJgA6F8pfF94NCAABIAAEQOegDAABIAAEgEAoEwCdC+WvC+8GBIAAEAACoHNQBoAAEAACQCCUCYDOhfLXhXfzH4G9e/dy938RERHdunWbO3eu0WgU3NHhcHz11VfDhw9v165dmzZtEhIS1q1bV1FRIcjmdDq//fbbcePGxcfHR0ZGdu3aderUqVlZWeJsOKaqqmrr1q1/+9vf2rZtGxUV1a9fv0WLFuXk5OAMEAACQAARAJ2DkgAEPCGAdG79+vX79u3btWvXu+++GxER0adPn6qqKny5urq6adOmcRw3fPjwrVu3ZmRkzJo1S6vVJiQkFBcX42wOhyM5OZnjuIEDB27cuHH37t0bNmwYNGgQx3G//PILzkYGzGYzyjBhwoRt27bt2rVr5cqVPXv2jIyMJLNBGAgAAacT1v2CUgAEPCKAdO7KlSv47FWrVnEcd/jwYRyTlpbGcVxKSgqOcTqdx48f12q148aNw5GbN2/mOG7ZsmUOhwNHOp3Or7766vLly2QMDo8fP16r1R49ehTHOJ3O6upqwb3IVEq4tra2urqakgGSgEBQE4D+XFB/Pnj4gBEQ69yJEyc4jktLS0PPZLfbO3To0K9fv9raWsFTot7bxYsXnU6n3W6Pj49/6aWX6urqBNnkDi9dusRx3Lx58+QyOJ3OxIYfmWHu3Lm9evVCMTqdjuO4zZs3b926tXfv3lqt9tKlSxEREevWrSNPuXPnDsdxO3bsQJEWi2Xp0qU9evSIjIzs06dPenp6fX09mR/CQECdBEDn1Pld4KnUTkCsc59++um9CbvPP/8cPfoPP/zAcZxAOVDS2bNnOY5bs2aN0+lE2davX8/+wmvWrOE47ty5c5RTWHSuf//+vXv3Tk9P37p1q16vHzVqVP/+/clrrlu3LiIiAg2xVlZWvvzyy/Hx8WvWrPniiy/mzJmj0WiWLl1K5ocwEFAnAdA5dX4XeCq1E0A6l5WVZTabDQbD0aNHO3XqFBUVZTAY0KNv27aN47hjx46J36SsrIzjuH/84x9Op3P79u1y2cQnopjJkydzHGexWOQyMPbn2rZtazab8UUyMjI4jrt16xaO6d+//6hRo9DhRx99FBsbm5ubi1NXr14dERFRVFSEYyAABNRJAHROnd8FnkrtBJDO3be4dP2/V69ep0+fxs/90UcfcRwnaTNZW1vLcdyYMWOcTiclG76UIDB69Oh745b0cU6W/lxycjJ5ZbPZ3Lp167Vr16LIW7ducRyXkZGBDl9++eVx48aZiV9WVhbHcfv37ycvAmEgoEICoHMq/CjwSEFAAOncp59++uOPPx49evStt96KjY396aef8KO77c9Nnjw5sP058WDp2LFj+/Xrh15h7dq1rVu3xh2+Nm3akKKOw1u3bsWvDAEgoE4CoHPq/C7wVGonIJifq6urGzp0aLdu3crLy9Gjnz59Wm5+7qeffsLzcyibWHIo75+amurB/NysWbPEdiiCu6CXun79utPp7Nev39ixY3GGqKioN99880fRT6/X4zwQAALqJAA6p87vAk+ldgICnXM6nci6JD09HT16ZWVl+/btX3zxRfEA4zvvvMNxHLK3rKys7NChw1//+ldxNjkEFy5c4Dhu/vz5chmcTufkyZNfeeUVMsPw4cPd6pzFYomMjFy9evX169c5jtu7dy++Qv/+/V9//XV8CAEgEEQEQOeC6GPBo6qIgFjnnE7n3/72ty5dumBX8Q0bNnAct2rVKvK5T5w4odVqya5Seno6crMT+M/t27dPzn9u3LhxWq1WYORC+s+lpKRERUXhUccbN25otVq3Oud0OidMmNC7d+9Vq1ZFRkaSpi7r1q3jOO7UqVPku1gsFrHXBJkBwkBADQRA59TwFeAZgo+ApM4dOXLk3sQVdi2oq6ubMmUKx3EjRozYvn37zp0758yZo9VqBwwYQK6HUl9fP3v2bI7jBg0alJaWtmfPnrS0tL/97W8cx124cEESjdlsfvXVVzUazcSJE7dv375r165Vq1b16tULr4eSnZ19zytu4MCBn3766Ycffti5c+eEhAQWndu/f/89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)

ROC Curve는 왼쪽 위에 가까울수록 우수한 성능을 가진다고 할 수 있음

CatBoost, RandomForest, LogisticRegression 순서로 성능이 좋게 나타남

따라서 CatBoost가 해당 데이터에 대해 가장 우수한 성능을 보이는 모델이라고 할 수 있음

## **2. 각 모델의 최적 모델의 성능 평가 점수**

#### RandomForest의 결과물
```
Best parameters: {'max_depth': 20, 'max_features': 'log2', 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 300}
Best score: 0.8129130225654129
```

#### LogisticRegression의 결과물
```
Best parameters: {'C': 0.01, 'class_weight': None, 'max_iter': 100, 'multi_class': 'ovr', 'penalty': 'l2', 'solver': 'liblinear'}
Best score: 0.7832915608562754
```

#### CatBoost의 결과물
```
Best parameters: {'learning_rate': 0.05, 'l2_leaf_reg': 3, 'depth': 5}
Best score: 0.8129139535244558
```

## **2. Kaggle 제출을 통한 검증 결과**

![image.png](data:image/png;base64,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gCHhfQbM39lq/q7qEydQ/UQ7QSUDHfWP1XFaBE41dsKtrvUfMRoNQr9qnrD8sgZ7c1zh9IS8O3CUO75NWI9Mh5GwFfb7O4zCv7duQrHbaGmfiIiIiIiI6JeCQSlj9XRAYHwIBqQEYPQkbwQ8PwlekRfx19i3lUCKGRzfWA/HzCCoZ/giYJovYm4EYaXXHQyZshmC6jh5mJ2vbzTqZ0diquFlerpi/rqB2PvK8/KQvGlBSOvvjmHKnE/DJvrju2hvqN89hnqlbgNTfOW6zXgeQen94Ta8ywmiWrNU491Yd5St8ob780LdJk2Cv2YAFsfPge1tZIT1V0fiA7cyrPWeJNTHGxMnBeFQ3xBsmuegNyytA09OxPQelcD0ibd1T1l/jFkeBk9lr40nffC+RzkWiPNJiW0qvOOgr0bjfb1Z3POF9zJ67N9aPhuK5BMd9o9IjLvNZu7UYC+snH0VMb7CPYQ6quenw3HdAjjqhoIOHQ3P3vVwetpe2XeAc0M9ho3svsAYERERERERUXf4XZNA2SZjNTagvtEMvTuKoPxQj/oevTs+35kaDZa+A4TGq9FfuE6DWW9laFbHGurrpQm1uyonEYeo4Q7rpk+4ToPwjEbdsxMN9Q0wu+vKdDPx/Qr1Etv0jqrWVf/oDlIdgd59f2FtR0RERERERGQkZkrdiR5dBBzuNJhhyMhAk1nv2wgOmXVT3YwIlhnjFxeQEonvt+9dtFNX/aM7SHX8BbYdERERERERkZGYKfVL80MNis8DjzlYdj2UjYiIiIiIiIjoPsWgFBERERERERERmRyH7xERERERERERkckxKEVERERERERERCbHoBQREREREREREZkcg1JERERERERERGRyDEoREREREREREZHJMShFREREREREREQmx6AUERERERERERGZHINSRERERERERERkcgxKERERERERERGRyTEoRUREREREREREJsegFBERERERERERmRyDUkREREREREREZHIMShERERERERERkckxKEVERERERERERCbHoBQREREREREREZkcg1JERERERERERGRyDEoREREREREREZHJMShFREREREREREQmx6AUERERERERERGZHINSRERERERERERkcgxKERERERERERGRyTEoRUREREREREREJsegFBERERERERERmRyDUkREREREREREZHIMShERERERERERkckxKEVERERERERERCbHoBQREREREREREZkcg1JERERERERERGRyDEoREREREREREZHJMShFREREREREREQmx6AUERERERERERGZHINSRERERERERERkcgxKERERERERERGRyTEoRUREREREREREJsegFBERERERERERmRyDUkREREREREREZHIMShERERERERERkckxKEVERERERERERCbHoBQREREREREREZkcg1L0m6Stq4O2UdnRd6sOddf1zmmFfa2yTURERERERETd5ndNAmWbjFR7OgWfaXKgOVsF9LXG2Ckz4T3ZAYNUSoG7VoWM2BSUKHvNHrDC2ElT4GjVbTe6N2qKsO/zz5Hzj0JcQl/YuE2Gn1freted3oPEE7WAnRdCxlspR02j9lAgRofkQBWQhFPLnSHVqlFo8xXzsWhvKcQY1KA3NfjXCxUI+NtiZKtmITkvFC6/8GbvPrr+Z4Gxs3zhaK4cJiIiIiIiIupGP19QquEyir/6El8W5yK95gIqbwgf5dTgB57A4H7DMdHOHa4jnoFtX+XEz62xCpqFaiw6XKcc0NPPAxGfxMDvsduJXNQiI2w2NpfY4/VtkZhgoRxGESIf9UaisteaCmPfy8COF00byGnRUZ1llw7Nh2dIJtq2kDkmvLMfG3yspSBQ7aeBGLk0B3htPy6EOshFTKTuSBhGz/0cg0L3I+01O/nY4cV46vVUwNoDgZNs4DglGBP6ZyL0b/Px2cBQHNDMgn0PqehvgK7/uSEqLxEvWYqHtsNz1efA31cJbWba90VERERERES/TiYfvqet+RI7t7yKZ1dMhseB1Vj29THk6gWkRJXfn0du1X4sy5yDMZGOeDZyGT4pvqmc/bloUbhpthyQsvZClCYPZ/63GOfzMrAhwA6q65kInx2Lwtsc6nWjphQl52pxo72hZHBGRFoBzhSInzykRU7GIKEe2UtjkNFOXMxUOqqztigWAVJAyhovvafBqa+E9inOw9GNs2Dfqw4ZK2Yj+vTPPxbOfHwkznxT3ByQEmlvyf1r7JxVCAsOxgRrYaefB6IKynA+7bcUkOpAY63wzoX3fk3ZJyIiIiIiIrpLpgtK/XgZh7e9jFGxc7Ds27OtglBdqbyRjpDdY/D0u7HIvaocNLWLKVgTVwH0ckPUzki89KQFzFUqqCytoV6+G+sm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+qsj8YFbGdZ6TxLq442Jk4JwqG8INs1z6Dr55smJmN6jEpg+8bbuKeuPMcvD4KnstfGkD973KMcCcT4psU2Fdxz01Wi8rzeLe77wXkaP/VvLZ0ORfKLD/hGJcbfZzB3p7fE2wi12wkfoWwHTJiEg0wGblj8nZYv1th2FgVkLlPfrjWVfqzHfwzCUZgkPf3scnC/UT3y+wCzcfEI5JbT8sHkfYPoPW7HgFeFcRBYeGeOunAMGekUi8Ea0NDxQ6j/ivd9oO7E7ERERERERkbF+1yRQtslYjQ2obzSTs03a80M96nv07vh8Z2o0WPoOEBqvRn/hOg1mvWHWRfClob5emlC7q3IScYga7rBu+oTrNAjPaNQ9O9FQ3wCzu65MNxPfr1AvsU3vqGpd9Y+7JVy/Qbi+WXvXN6bvSc8H9O5rUEiXIaZ7p19EY3T+c62HeYr9p1G4fkfzoREREREREREZiZlSd6JHFwGHOw1mGDIy0GTW+zaCQ2bdVDcjgmXG+MUFpETi++17F+3UVf+4W8L12w1IiYzpe9LzGRZqQNk2X/hGpCC/tBRlWSlYt6EIfmPslfMKsf8wIEVERERERETdgJlSvzQ/1KD4PPCYgyWHRpHJXStNR05WOa72HAwnd3c4DjV6KnkiIiIiIiKi28KgFBERERERERERmRyH7xERERERERERkckxKEVERERERERERCbHoBQREREREREREZkcg1JERERERERERGRyDEoREREREREREZHJMShFREREREREREQmx6AUERERERERERGZHINSRERERERERERkcgxKERERERERERGRyTEoRUREREREREREJsegFBERERERERERmRyDUkREREREREREZHIMShERERERERERkckxKEVERERERERERCbHoBQREREREREREZkcg1JERERERERERGRyDEoREREREREREZHJMShFREREREREREQmx6AUERERERERERGZHINSRERERERERERkcgxKERERERERERGRyTEoRUREREREREREJsegFBERERERERERmRyDUkREREREREREZHIMShERERERERERkckxKEVERERERERERCbHoBQREREREREREZkcg1JERERERERERGRyDEoREREREREREZHJMShFREREREREREQmx6AUERERERERERGZHINSRERERERERERkcgxK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)

ROC Curve에서 본 결과와 같이 CatBoost 모델이 가장 우수한 결과를 보임\
그 뒤로 RandomForest, LogisticRegression의 순서로 성능이 좋았음

## **3. 최적 모델을 통한 데이터 해석**
SHAP(SHapley Additive exPlanations)를 통해 최적 모델의 예측을 해석
- 각 feature가 얼마나 영향을 주는지 확인
"""

# SHAP 값을 계산
explainer = shap.Explainer(cb_best_model)
shap_values = explainer.shap_values(X_train)

# SHAP 요약 플롯 생성
shap.summary_plot(shap_values, X_train, plot_type="bar", class_names=cb_best_model.classes_, feature_names=ct.get_feature_names_out(), plot_size=(10, 6), show=False)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

"""Spa, VR_Deck, RoomService, FoodCourt 와 같이 금액을 지불하는 서비스를 사용했는지의 여부가 생존에 있어 가장 중요한 feature로 나타남

Cryo Sleep의 경우 값이 True인 경우 위 4개의 데이터에 대해서 값이 모두 0으로 나타나는 특징이 있기 때문에 그 다음으로 영향이 높게 나타남

결과적으로 돈을 지불하는 유료 서비스의 사용 여부가 Transported의 값을 결정짓는 가장 중요한 요소라고 할 수 있음
"""
