# Ensemble 

# Table of Contents 
- [Ensemble](#ensemble)
- [Table of Contents](#table-of-contents)
- [이론](#이론)
- [튜토리얼](#튜토리얼)
  - [0. 기본 함수 정의](#0-기본-함수-정의)
  - [1. Bagging](#1-bagging)
  - [1.1 Bagging - 비율은 고정 시키고 모델 갯수에 따른 성능 변화](#11-bagging---비율은-고정-시키고-모델-갯수에-따른-성능-변화)
- [2. 모델 앙상블](#2-모델-앙상블)
- [3. Features 앙상블](#3-features-앙상블)

# 이론 

# 튜토리얼 
- 해당 파트에서는 앙상블과 관련된 튜토리얼 + 실험을 진행 합니다. 
- 앙상블은 각기 다른 모델 또는 데이터를 이용하여 최대한 많은 다양성을 확보하여 최종 모델의 성능을 향상시키는 것을 목적으로 합니다. 
- 모델링 방법론 측면에서의 앙상블 방법(ex 부스팅)이 있겠지만 테크닉 관점에서 앙상블을 적용할 수 있습니다. 
- 그래서 이번 튜토리얼에서는 데이터, 모델 종류, Feature 종류 를 변화해 가며 혼합하여 앙상블을 진행할 것입니다. 
- 튜토리얼의 목적은 데이터, 모델 종류, Feature 종류를 변화하며 앙상블을 진행 했을 때 성능이 어떻게 변화하는지 확인하고자 합니다. 

- 사용 데이터 : California housing price 
- 사용 모델 : [DecisionTreeRegressor,LinearRegression,SVR,LinearSVR,BayesianRidge,Ridge,TweedieRegressor,SGDRegressor]
- 사용 Feature : ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup','Latitude', 'Longitude', 'target']

## 0. 기본 함수 정의 
```python
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler
import pandas as pd 
import numpy as np 
import warnings 
from tqdm import tqdm 
import matplotlib.pyplot as plt 
warnings.filterwarnings('ignore')

data = fetch_california_housing()
df = pd.DataFrame(data.data)
df.columns = data.feature_names
df['target'] = data.target

# Train-valid-split 
indx = int(len(df)*0.1)
train_df = df.iloc[:indx*8,:] 
test_df = df.iloc[indx*8:,:]

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

def random_sampling(df,ratio,cfg):
    features = cfg['feature']
    df_y = df.sample(frac=ratio)[cfg['feature']]['target']
    df_x = df.sample(frac=ratio)[cfg['feature']].reset_index(drop=True).drop(columns='target').values
    return df_x, df_y 

def ineference(model_list,x):
    predicted = [] 
    for model in model_list:
        y_pred = model.predict(x)
        predicted.append(y_pred)
    y_pred = np.mean(predicted,axis=0)
    return y_pred 

def metric(model_list,cfg):
    train_x,train_y = random_sampling(train_df,1,cfg)
    test_x,test_y = random_sampling(test_df,1,cfg)

    train_y_pred = ineference(model_list,train_x)
    test_y_pred = ineference(model_list,test_x)
    train_loss = mean_absolute_error(train_y,train_y_pred)
    test_loss = mean_absolute_error(test_y,test_y_pred)

    #print(f"Train MAE : {train_loss}")
    #print(f"Test MAE : {test_loss}")
    return train_loss, test_loss 

def train_epoch(cfg,model_list):
    model_save = [] 
    for model_name in model_list:
        for i in range(cfg['num_iter']):
            train_x,train_y = random_sampling(train_df,cfg['ratio'],cfg)
            model = model_name()
            model.fit(train_x,train_y)
            model_save.append(model)
    return model_save

```

## 1. Bagging
- 데이터의 일정 비율 만큼 Random 으로 샘플링한 뒤 여러 번 모델을 학습 함 
- 학습한 모델을 혼합하여 최종 모델을 만들고 성능을 측정 함 
- 데이터를 Random sampling하는 비율을 바꿔가며 어떤 비율로 데이터를 샘플링 해 앙상블 하는 것이 가장 좋은 성능을 보이는 지 확인 
  
```python
from sklearn.tree import DecisionTreeRegressor
cfg = {} 
cfg['ratio'] = 0.9 
cfg['num_iter'] = 100 
cfg['model'] = [DecisionTreeRegressor]
cfg['feature'] = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup',
                'Latitude', 'Longitude', 'target']

total_train = []
total_test = [] 
for ratio in tqdm(list(np.arange(0.05,1.05,0.05))):
    cfg['ratio'] = ratio 
    model_list = train_epoch(cfg,LinearRegression)
    train_loss,test_loss = metric(model_list,cfg)

    total_train.append(train_loss)
    total_test.append(test_loss)
total_test = np.array(total_test)
total_train = np.array(total_train)                
```
**결과 확인**
```python
fig,ax = plt.subplots()
ax.plot(list(np.arange(0.05,1.05,0.05)),total_test,color='b',label='test')
ax1 = ax.twinx()
ax1.plot(list(np.arange(0.05,1.05,0.05)),total_train,color='r',label='train')
fig.legend()
plt.grid(True)

plt.show()
```

<p align='center'><img src = https://user-images.githubusercontent.com/92499881/203989121-2a4d450a-96de-497b-a27c-699a3f3c6b4c.png width='60%',hegiht='70%'>

**결과해석**
```
모든 데이터를 사용하는 Ratio=1인 경우 보다 Ratio < 1 을 선택하는 경우 대체로 더 나은 성능을 보인다. 데이터의 일부만을 샘플링 하여 여러개의 모델을 만들어 합쳤기 때문에 다양성이 증가되고 기대하던 앙상블의 효과를 얻은 것이라 생각된다. 

하지만 이 샘플링 비율을 너무 낮춘 경우 되려 더 좋지 못 한 성능을 보이게 되고 이는 샘플링 한 데이터의 양이 너무 적어 해당 Train 데이터에 오버 피팅 된 것이라 생각 될 수 있다. 이는 Ratio=0.3에서 확인할 수 있는데, Train Loss(Red)의 경우 낮은 Loss를 보여주는 반면 Test Loss는 가장 높은 Loss를 보여주며 Overfitting 이 되었다는 것을 확인할 수 있었다. 
```

## 1.1 Bagging - 비율은 고정 시키고 모델 갯수에 따른 성능 변화 
- 데이터 샘플링 하는 비율은 고정한 채 앙상블을 위해 여러개의 모델을 만드는 경우 모델의 갯수에 따른 성능 변화를 확인하고자 한다. 
- 데이터 샘플링 비율은 임의의 값으로 설정하며, Base모델로 Linear Regression을 사용 

```python
from sklearn.tree import DecisionTreeRegressor
cfg = {} 
cfg['ratio'] = 0.5
cfg['model'] = [LinearRegression]
cfg['feature'] = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup',
                'Latitude', 'Longitude', 'target']

total_train = []
total_test = [] 
Range=  list(np.arange(400,600,10))
for num_iter in tqdm(Range):
    cfg['num_iter'] = num_iter 
    model_list = train_epoch(cfg,cfg['model'])
    train_loss,test_loss = metric(model_list,cfg)

    total_train.append(train_loss)
    total_test.append(test_loss)
total_test = np.array(total_test)
total_train = np.array(total_train)

fig,ax = plt.subplots()
ax.plot(list(Range),total_test,color='b',label='test')
ax1 = ax.twinx()
ax1.plot(list(Range),total_train,color='r',label='train')
fig.legend()
plt.grid(True)

plt.show()
```

<p align='center'><img src = https://user-images.githubusercontent.com/92499881/203994103-3c698d25-e41c-4353-ac33-fb8954c1701f.png>
<center><왼쪽 축 : Train | 오른쪽 축 : Test></center>

**결과해석**
```
우선 Train보다 Test의 성능이 전체적으로 좋은 것으로 보아 Underfitting이 발생한 것으로 생각 된다. 이는 모델의 복잡도 혹은 Capacity가 너무 낮아 발생하는 현상으로 Base model로 Linear regression이 아닌 다른 모델을 사용 할 필요가 있어 보인다. 

또한 초반 모델 갯수가 적은 경우 높은 loss를 보이는 것을 제외하곤 이후에 모델 갯수에 따른 변화가 크게 있어 보이지 않는다. 이는 일정 수준의 모델 갯수가 충족 되는 경우 더 많은 모델이 필요 없음을 의미한다. 
```



# 2. 모델 앙상블 
- 데이터를 변화 시키는 것 뿐만 아니려 여러 개의 모델을 섞어 사용하는 경우 앙상블의 효과를 기대할 수 있다. 
- 그래서 이번 파트에서는 모델의 갯수에 따른 변화를 보기 위한 실험을 진행 하고자 한다. 
- 총 7개의 모델을 사용하며, 2개부터 5개까지 임의로 모델을 선택하여 모델링을 진행한 뒤 성능을 비교한다. 
- 이 경우 선택된 모델이 무엇이냐에 따라 성능이 변화할 수 있기 때문에 신뢰 가능한 표본을 뽑기 위해 동일한 절차를 50번 진행한 뒤 평균을 구해 최종 지표를 산출한 뒤 비교할 것이다. 
- 사용 모델 
  - DecisionTreeRegressor
  - LinearRegression
  - SVR
  - BayesianRidge
  - SGDRegressor

# 3. Features 앙상블 
- RandomForest나 IsolationForest는 앙상블을 위한 다양성을 확보하기 위해 데이터의 Features를 임의로 선택하여 모델링을 진행한다.
- 이러한 효과를 직접적으로 확인하기 위해 다른 하이퍼 파라미터는 고정한 채 Feature에만 변화를 주어 성능 변화를 비교하고자 한다. 
- 이 경우 모델 앙상블 파트와 유사하게 랜덤하게 Features를 추출한 뒤 모델 갯수에 따라 비교를 진행하고자 한다. 
- 모델 앙상블과 마찬가지로 Feature가 무엇을 선택하냐에 따라 성능 변화가 발생할 수 있기 때문에 50번 진행한 뒤 평균을 구해 최종 지표를 산출한 뒤 비교할 것이다. 