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
- [4. 모두 사용](#4-모두-사용)
- [결론](#결론)

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

fig,ax = plt.subplots()
ax.plot(list(np.arange(0.05,1.05,0.05)),total_test,color='b',label='test')
ax1 = ax.twinx()
ax1.plot(list(np.arange(0.05,1.05,0.05)),total_train,color='r',label='train')
fig.legend()
plt.grid(True)

plt.show()
```

**실험 결과** 

<p align='center'><img src = https://user-images.githubusercontent.com/92499881/204077962-157bfa87-6504-4aff-a345-1aa7715d026a.png width='60%',hegiht='70%'>
<center>왼쪽 축 : Test Loss | 오른쪽 축 : Train Loss</center>

**결과해석**
```
Train Loss의 경우 샘플링의 Ratio가 높을 수록 성능이 좋아지는 반면 Test Loss는 샘플링의 Ratio가 적을 수록 성능이 높아지는 경향을 보여준다. 이는 Train 데이터에 의한 Overfitting이라 생각 된다. 모델의 복잡도가 높아 데이터의 양이 많을 수록 성능은 증가하지만 그 만큼 overfitting 경향이 강해져 Test 데이터의 Loss는 증가하게 된다. 반대로 샘플링의 Ratio가 낮을 수록 복잡도는 상대적으로 낮아지고 다양성이 증가해지기 때문에 Overfitting이 완화 되어 낮은 Test Loss를 보인다고 생각 된다. 

```

## 1.1 Bagging - 비율은 고정 시키고 모델 갯수에 따른 성능 변화 
- 데이터 샘플링 하는 비율은 고정한 채 앙상블을 위해 여러개의 모델을 만드는 경우 모델의 갯수에 따른 성능 변화를 확인하고자 한다. 
- 데이터 샘플링 비율은 임의의 값으로 설정하며, Base모델로 Linear Regression을 사용 

```python
from sklearn.tree import DecisionTreeRegressor
#Hyperparameter settings 
cfg = {} 
cfg['ratio'] = 0.9
cfg['model'] = [DecisionTreeRegressor]
cfg['feature'] = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup',
                'Latitude', 'Longitude', 'target']

total_train = []
total_test = [] 
Range=  list(np.arange(10,600,10)) #Hyperparameter for number of iteration 
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

**실험 결과** 

<p align='center'><img src = https://user-images.githubusercontent.com/92499881/204090111-85e71014-3ede-40a1-9925-cf68f6c6a9d4.png>
<center><왼쪽 축 : Train | 오른쪽 축 : Test></center>

**결과 해석**
```
Train Loss 의 경우 반복 횟수가 증가함에 따라 Loss가 점차 감소하는 추세를 보여주고 있다. 하지만 일정 반복 횟수를 넘은 뒤에는 큰 편차가 발생하지 않은데, 이는 일정갯수 이상의 모델을 만들어 앙상블 하는 경우 큰 의미가 없음을 의미한다. 어느정도 모델의 갯수가 넘어간 이후에는 Variance가 잡혀 더 이상의 모델을 만들어도 변화가 없기 때문이라 생각 된다. 

하지만 Test의 경우 되려 Loss가 증가하게 되는데 이는 모델의 수가 너무 많아 Overfitting이 되었으며, Train 데이터의 Variance를 해결하다 보니 전체적인 일반화에 실패하게 된 것이 아닌가 라는 생각이 든다. 
```



# 2. 모델 앙상블 
- 데이터를 변화 시키는 것 뿐만 아니려 여러 개의 모델을 섞어 사용하는 경우 앙상블의 효과를 기대할 수 있다. 
- 그래서 이번 파트에서는 모델의 갯수에 따른 변화를 보기 위한 실험을 진행 하고자 한다. 
  - 위에서 말한 모델 갯수와 헷갈릴 수 있는데, 위에서는 동일한 모델을 반복해서 만든 경우이며, 이번 파트에서는 각기 다른 모델을 합치는 경우를 의미한다. 
- 총 6개의 모델을 사용하며, 1개부터 6개까지 임의로 모델을 선택하여 모델링을 진행한 뒤 성능을 비교한다. 
- 이 경우 선택된 모델이 무엇이냐에 따라 성능이 변화할 수 있기 때문에 신뢰 가능한 표본을 뽑기 위해 동일한 절차를 300번 진행한 뒤 평균을 구해 최종 지표를 산출한다.
- 사용 모델 
  - DecisionTreeRegressor
  - LinearRegression
  - SVR
  - BayesianRidge
  - Ridge
  - SGDRegressor

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression,BayesianRidge,Ridge,TweedieRegressor,SGDRegressor
from sklearn.svm import SVR 
from sklearn.svm import LinearSVR
#Hyperparameter Setting 
cfg = {} 
cfg['ratio'] = 1
cfg['num_iter'] = 1 
cfg['model'] = [DecisionTreeRegressor]
cfg['feature'] = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup',
                'Latitude', 'Longitude', 'target']
model_kind = [DecisionTreeRegressor,LinearRegression,SVR,BayesianRidge,Ridge,TweedieRegressor]                

#Train 
train = []
test = [] 
Range = list(np.arange(1,len(model_kind),1))
for i in range(300):
    total_train = []
    total_test = []     
    for model_num in tqdm(Range):
        #Random model choice 
        cfg['model'] = np.random.choice(model_kind,model_num,replace=False)
        #Train 
        model_save = train_epoch(cfg)
        train_loss,test_loss = metric(model_save,cfg)

        total_train.append(train_loss)
        total_test.append(test_loss)
    
    total_test = np.array(total_test)
    total_train = np.array(total_train)
    
    train.append(total_train)
    test.append(total_test)
#Average                        
train = np.mean(np.array(train),axis=0)
test = np.mean(np.array(test),axis=0)    
#Plot 
plt.plot(list(Range),train,label='train')
plt.plot(list(Range),test,label='test')
plt.legend()
plt.show()
```

**실험 결과**
<p align='center'><img src = https://user-images.githubusercontent.com/92499881/204090278-2248efab-15fa-43ff-a755-ccd2a1dcc8f0.png>
<center><X축 : 모델 갯수 | Y축 : MAE Loss ></center>


**결과 해석**
```
모델의 종류가 증가함에 따라 전체적으로 성능이 증가한 것을 확인할 수 있다. 여러번 반복 시행한 뒤 평균을 낸 결과이기 때문에 특정 모델이 샘플링 되어 성능이 좋은 것이 아닌 여러 모델을 혼합하여 최종 모델을 만들었기 떄문에 성능이 좋아졌다 라고 생각할 수 있다. 1개에서 2개로 모델의 종류가 변화하는 경우 성능 폭이 가장 컸고 그 이후로는 모델의 종류가 증가할 때 마다 크지는 않지만 계속해서 성능 향상이 일어남을 확인할 수 있었다. 

다만 이 실험 결과를 토대로 최대한 다양한 모델을 앙상블을 하게 되면 성능이 좋아질까? 라는 질문에 대해서는 의문을 품을 수 있을 것 같다. 우선 오버피팅을 무시할 수 없으며 혼합한 모델의 종류 갯수가 일정 수치를 넘어 가게되면 성능 향상이 멈추지 않을까 라는 생각이 든다. 
```

# 3. Features 앙상블 
- RandomForest나 IsolationForest는 앙상블을 위한 다양성을 확보하기 위해 데이터의 Features를 임의로 선택하여 모델링을 진행한다.
- 이러한 효과를 직접적으로 확인하기 위해 다른 하이퍼 파라미터는 고정한 채 Feature에만 변화를 주어 성능 변화를 비교하고자 한다. 
- 이 경우 모델 앙상블 파트와 유사하게 랜덤하게 Features를 추출한 뒤 모델 갯수에 따라 비교를 진행하고자 한다. 
- 모델 앙상블과 마찬가지로 Feature가 무엇을 선택하냐에 따라 성능 변화가 발생할 수 있기 때문에 50번 진행한 뒤 평균을 구해 최종 지표를 산출한 뒤 비교할 것이다. 

- 이는 Feature와 Target 간의 상관관계, 혹은 영향력을 무시하고 최대한 Feature의 갯수 또는 다양성에 따른 성능 변화를 보기 위해 Feature를 무작위 추출하며, 상관관계가 높은 특정 변수에 의한 성능 변화를 무시하기 위해 300번 반복 후 평균을 내는 방식으로 비교를 하고자 한다. 

```python
cfg = {} 
cfg['ratio'] = 1
cfg['num_iter'] = 1
cfg['model'] = [DecisionTreeRegressor]
cfg['feature'] = []
features_kind = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup',
                'Latitude', 'Longitude']               
                
train = []
test = [] 
Range = list(np.arange(1,len(features_kind),1))
for i in tqdm(range(300)):
    total_train = []
    total_test = [] 
    for features_num in Range:
        #Feature choice 
        selected_features = list(np.random.choice(features_kind,features_num,replace=False))
        selected_features.append('target')
        cfg['feature'] = selected_features
        #Train 
        model_save = train_epoch(cfg)
        train_loss,test_loss = metric(model_save,cfg)
        #Save 
        total_train.append(train_loss)
        total_test.append(test_loss)
    
    total_test = np.array(total_test)
    total_train = np.array(total_train)
    
    train.append(total_train)
    test.append(total_test)

train = np.mean(np.array(train),axis=0)
test = np.mean(np.array(test),axis=0)    

plt.plot(list(np.arange(1,8,1)),train,label='train')
plt.plot(list(np.arange(1,8,1)),test,label='test')
plt.legend()
plt.show()
```

**실험 결과** 
<p align='center'><img src =https://user-images.githubusercontent.com/92499881/204074489-ac66beb0-72a0-4012-900a-7c95043b8e4b.png>
<center><X축 : 모델 갯수 | Y축 : MAE Loss ></center>

**결과 해석**
```
기대했던 대로 Feature가 많아질 수록 성능이 향상 되는 것을 확인할 수 있었다. Test에서도 Feature와 1개,2개인 경우 큰 차이는 없었지만 3개 부터는 Feature의 갯수가 증가함에 따라 성능이 증가하였다. 
```

# 4. 모두 사용 
- 해당 파트에서는 위에서 시도 한 실험 결과를 토대로 종합하여 최적의 모델을 만들고자 하며, 적용하는 하이퍼 파라미터는 다음과 같다. 
  - 데이터 샘플링 비율 - 0.4 
  - 모델 생성 반복 횟수 - 100개 
  - 모델 종류 갯수 - 6개 
  - Feature 갯수 - 6개

```python
cfg = {} 
cfg['ratio'] = 0.4
cfg['num_iter'] = 100
cfg['model'] = [DecisionTreeRegressor]
cfg['feature'] = []
features_kind = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup',
                'Latitude', 'Longitude']               
model_kind = [DecisionTreeRegressor,LinearRegression,SVR,BayesianRidge,Ridge,TweedieRegressor]                
           
train = []
test = [] 
Feature_num = 6
Model_num = 6
for i in tqdm(range(1)):
    total_train = []
    total_test = [] 
    
    #Feature choice 
    selected_features = list(np.random.choice(features_kind,Feature_num,replace=False))
    selected_features.append('target')
    cfg['feature'] = selected_features

    #Model Choice 
    cfg['model'] = np.random.choice(model_kind,Model_num,replace=False)
    #Train 
    model_save = train_epoch(cfg)
    print('model train done')
    train_loss,test_loss = metric(model_save,cfg)
    #Save 
    total_train.append(train_loss)
    total_test.append(test_loss)

    total_test = np.array(total_test)
    total_train = np.array(total_train)
    
    train.append(total_train)
    test.append(total_test)

train = np.mean(np.array(train),axis=0)
test = np.mean(np.array(test),axis=0)    

plt.plot(Range,train,label='train')
plt.plot(Range,test,label='test')
plt.legend()
plt.show()

print(f" Train loss = {train}")
print(f" Test loss = {test}")
```
**실험 결과** 
<p align='center'><img src =https://user-images.githubusercontent.com/92499881/204090446-bf43ec5a-7a4a-4591-9ddd-260408fd60b8.png>
<center><X축 : 모델 갯수 | Y축 : MAE Loss ></center>
<p align='center'><img src =https://user-images.githubusercontent.com/92499881/204090513-e2a06915-dfc2-4c69-9234-8c822854c5d3.png>
<center><X축 : 모델 갯수 | Y축 : MAE Loss ></center>

**결과 해석**
```
모든 하이퍼 파라미터(Ratio,모델갯수,모델종류,Feature종류)를 사용한 경우  어떠한 앙상블을 사용하지 않은 경우(베이스라인 표) 보다는 좋은 성능을 보여주고 있지만 데이터만을 이용한 경우 보다 되려 성능이 떨어지는 결과를 보여준다.이런 경우 문제점이 하나 더 있는데 학습 시간과 Inference 시간이 오래 걸린다는 문제이다. 

다만 이러한 결과가 모든 경우에 통용된다 라고는 하기 어려운데, 이는 데이터의 차원, 모델의 Capacity에 따라 데이터에 의한 앙상블, 모델에 의한 앙상블이 적합한 것이 있고, 부적합한 것이 있기 때문이다. 이번 실험에서 사용한 데이터의 경우 차원이 낮기 때문에 여러 모델을 사용하는 경우 그 복잡도가 증가하여 좋지 못한 성능을 보여준다 생각이 되며, 이러한 경우 데이터를 이용항 앙상블이 좋은 효과를 보여준 것 같다. 
```

# 결론 
- 앙상블은 여러 모델을 만들어 혼합함으로써 최종 성능을 높이고자 한다. 다양성을 높이는 것이 관건인데, 역시 Feature, 모델, 데이터의 종류 갯수 등 변화를 줌으로써 다양성을 꿰하니 성능은 자연스럽게 증가하였다. 하지만 이러한 방법이 가장 좋은 것인가에 대해서는 의문이 든다. 그러한 이유는 속도 때문이다. 마지막 실험에서 Feature, 모델 종류, 갯수, 샘플링 비율 모든 하이퍼 파라미터를 적용하여 최종 모델을 만든 경우 상당히 좋은 성능을 도출할 수 있었다. 하지만 학습부터 추론까지 8분이라는 긴 시간이 소요 되었다. 

- 마지막 실험에서도 확인할 수 있다 싶이 모든 방법을 동원하여 다양성을 늘리는 것이 항상 좋은 결과를 도출하는 것은 아니며, 데이터의 차원, 모델의 복잡도 등을 고려하여 적합한 앙상블 방법을 고려 할 필요성이 있어 보인다. 
