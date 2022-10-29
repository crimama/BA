# SVM

- Support Vectors Machine(SVM)에 관한 이론적 배경과 함께 비선형 데이터를 위한 Case 3 : Kernel SVM에 대해 주로 다룹니다. 
- 관련된 이론적 배경을 설명한 뒤 실질적으로 비선형 데이터를 분류하는지 시각화를 통해 확인하고자 합니다. 
- 그 외에도 다른 비선형 분류 방법론들과 함께 비교하면서 어떤 차이점이 있는지 비교하고자 합니다. 

# Table of Contents 
  - [1.이론적 배경](#1이론적-배경)

  - [2.SVM Cases](#2-svm-cases)
  
  - [3.kernel svm](#case-3--linear--soft-margin-svm)
    - [3.1 Kernel 함수에 따른 차이 비교](#3.1-Kernel-함수에-따른-차이-비교)
    - [3.2 다른 분류 모델과의 비교](#3.2-다른-분류-모델과의-비교) 

  - [Appendix : 이론적 배경](theory.md)
    
 
$\space$ 
# 1.이론적 배경

**Support Vector Machine**은 벡터 공간에서 다른 클래스의 데이터들간 가장 잘 나눌 수 있는 초 평면, 결정 경계를 찾는 것을 목적으로 한다. 분류되지 않은 새로운 데이터가 나타났을 때 이 경계를 기준으로 어느 위치에 있는지 확인하여 분류 과제를 수행할 수 있다. 즉 SVM은 이 결정 경계를 어떻게 정의하고 계산하는 것이 매우 중요하다. 가운데 실선이 결정 경계이며, 실선으로 부터 가장 가까운 검은색 테두리의 빨간색 점 그리고 파란색 점을 지나는 점선이 존재한다. 결정경계부터 점선까지의 거리를 마진(margin)이라고 한다. SVM은 결정 경계를 만들 때 Margin을 최대화 하는 결정 경계를 찾고자 한다. 이러한 이유는 모델의 구조적 위험 때문이다.  
<p align="center"><img src="https://user-images.githubusercontent.com/92499881/195275527-08bc4c5c-aa7f-4d5d-98e7-d6a21be654f2.png"  width="30%" height="30%"/>


**구조적 위험**

분류 모형은 모형의 복잡도가 증가할 수록 주어진 학습 데이터를 더 잘 분류할 수 있게 되지만 되려 데이터가 갖고 있는 노이즈까지 학습하게 된다. 이런 경우 학습으로 사용한 Train데이터에 대해서는 좋은 성능을 보이지만 새로운 데이터가 들어왔을 때는 좋은 성능을 보이지 못한다. 따라서 분류 모델을 만들 경우 학습 과정에서 학습 오류를 낮춤과 동시에 모델의 복잡도를 줄여야 한다. 이 둘을 합쳐 구조적 위험이라 한다. 
 
  <p align='center'><img src = "https://user-images.githubusercontent.com/92499881/195276434-0294a827-6b88-47b4-89a5-e3a55a9e6dd7.png" width="40%" height='40%'/>
  <p align='center'><img src = "https://user-images.githubusercontent.com/92499881/195277956-db9be88e-fb44-4c1f-8bba-f2dc99a1bea6.png" width="50%" height='50%'/>
 
  모델의 성능과 복잡도는 Trade-off 관계를 가지며 동일한` 정확도라면 상대적으로 복잡도가 낮은 모델이 선호되어야 한다. 이는 수식적으로 풀이가 가능한데.
     
- VC dimension : 함수 H에 의해 최대로 shattered 될 수 있는 points의 수, 어떤 함수의 복잡도,Capacity를 측정하는 지표  
- 위 이미지에서 h가 VC dimension, 복잡도를 의미하며 이것이 커질 수록 R[f] term 전체가 증가하게 되고 위험도가 증가하게 된다. 
- 반대로 h가 낮아지거나 데이터의 양(n)이 많아질 수록 R[f] term 전체가 감소하고 리스크가 감소하게 된다. 
- 마진을 최대화 할 경우 데이터를 분류할 수 있는 경계면의 수가 감소하게 되고 이는 VC dimension의 감소를 의미한다. 
- 즉 마진 최대화 -> Vc dimension 감소 -> 구조적 위험도 감소로 수렴하게 된다. 

     

# 2. SVM Cases 
<figure class ='center'>     
<p align="center"><img src = "https://user-images.githubusercontent.com/92499881/195294814-c6d5a20b-e26d-4054-85a4-a5a22597d66b.png" width="50%" height='50%'/>

- SVM은 margin의 허용 범위, kernel 함수의 사용 여부에 따라 case 1부터 case3 까지 나뉠 수 있다. 
- 해당 튜토리얼에서는 Case3를 위주로 다루며, 자세한 이론적인 부분 그리고 나머지 Case 1과 Case 2에 대해서는 [Appendix : 이론적 배경](theory.md)에서 자세히 확인할 수 있다. 
- Case 3에 대한 설명과 함께 코드와 데이터를 통해 실질적으로 Kernel SVM이 비선형 데이터를 분류할 수 있는지 확인하고 다른 비선형 방법들과 비교를 해보고자 한다. 
</figure>

##  Case 3 : Linear & soft Margin SVM
  
- 만약 고차원의 데이터 이거나 클래스 간 결정 경계면이 **비선형**인 경우 앞선 Case 1과 Case 2를 사용하는데 어려움을 겪을 수 있다. 
- 그래서 Case 3는 원래 공간이 아닌 선형 분류가 가능한 더 고차원의 공간으로 보내 모델을 학습하자 라는 아이디어를 갖고 출발하며, 고차원 공간으로 옮기기 위해 매핑 함수를 사용한다. 
- 목적은 앞선 과정과 마찬가지로 마진을 최대화 하는 결정 경계면을 찾는 것이며 여기에 유연성 확보를 위한 비선형 분류 경계면 생성이 추가된다. 
 
**목적함수 및 제약식** 
  
$$
min {1\over 2}||w||^2 + C\sum^N_{i=1}\xi_i
$$

$$
s.t. y_i(w^T\Phi(x_i)+b) \geq 1 - \xi, \xi \geq 0,
$$   
  
- 이 때 $\Phi$가 x를 고차원으로 매핑해주는 매핑 함수를 의미한다. 
- 앞선 Case1과 2와 마찬가지로 라그랑주 승수법을 통해 풀면 아래의 식을 얻을 수 있다. 
  
$$
max L_D = \sum^n_{i=1}\alpha_i - {1\over 2}\sum^n_{i=1}\sum^n_{j=1}\alpha_i\alpha_jy_iy_j\Phi(x_i)\Phi(x_j)
$$
- 이 때 하나의 문제가 발생하게 되는데 고차원으로 이동시키는 매핑 함수를 찾을 수 없다는 것이다. 
- 하지만 최종 목적 함수를 잘 보면 고차원에서 항상 두 벡터 간의 내적(inner product) 형태로만 존재한다는 것을 알 수 있다. 
- 만약 저차원 데이터를 이용해 고차원 공간상에 내적 값을 결과물로 줄 수 있다면 명시적인 고차원 매핑은 필요가 없게 된다.   

**커널 트릭**
- 고차원 상에서의 목적 함수를 달성하기 위해 매핑 함수를 직접 찾는 것이 아닌 내적값을 결과물을 내는 커널 함수를 이용하기 때문에 커널 트릭이라 한다. 
- 우리가 찾고자 하는 것은 커널 함수이며, 유효한 함수는 다음의 조건을 만족해야 한다. 
  1. Symmetric해야 하며 
  2. Positive semi-definite 해야 한다. (Mercer's condition) 
  
  
**대표적 커널 함수** 
- Polynomial
- Gaussian(RBF) 
- Sigmoid 
- 커널함수에 따라 나타나는 결정 경계면의 형태가 다르며 커널 함수 별 특징이 있다. 가우시안 커널 함수(RBF)의 경우 이론적으로 무한개의 점을 shatter 할 수 있다 라는 장점이 있다. (VC dimension이 무한대가 아님) 

$\space$


# 3. Kernel SVM을 이용한 비선형 데이터 분류 
- 이번 파트 부터는 실질적 코드와 시각화를 통해 kernel SVM이 어떻게 작용하고, kernel 함수에 따라 결정 경계면과 Metrics이 어떻게 변하는지 확인하고자 한다. 
- kernel만의 비교 뿐만 아니라 다른 분류 모델과의 비교 또한 진행하고자 한다. 
  
$\space$
## 3.1 Kernel 함수에 따른 차이 비교 


### 3.1.1. 데이터 로드 및 확인 
- 실험에 사용할 데이터는 Sklearn 라이브러리에 포함되어 있는 iris데이터를 사용한다 
- Feature로는 `sepal length` 와 `sepal width`만을 사용 함
```python
def data_load():
    #데이터 로드 
    data = load_iris() 
    df = pd.DataFrame(data.data)
    df.columns = data['feature_names']
    df['class'] = data['target']
    df = df.sample(frac=1,random_state=42).reset_index(drop=True) #shuffle 
    return df 

def train_split(X,Y):
    idx = int(len(X)*0.8)
    train_x = X[:idx]
    train_y = Y[:idx]
    test_x = X[idx:]
    test_y = Y[idx:]
    return train_x, train_y, test_x, test_y


def data_preprocess(df):
    class_list = ['sepal length (cm)','sepal width (cm)']
    X = df.drop(columns='class')[class_list].to_numpy()
    Y = df['class'].to_numpy()


    train_x, train_y, test_x, test_y = train_split(X,Y)
    return (X,Y),(train_x, train_y, test_x, test_y)

df = data_load() 

(X,Y),(train_x, train_y, test_x, test_y) = data_preprocess(df)


for i in np.unique(df['class']):
    plt_x = X[np.where(Y == i )[0]] 
    plt.scatter(plt_x[:,0],plt_x[:,1],label=i)
    plt.legend()
plt.show()
```
<p align="center"><img src = https://user-images.githubusercontent.com/92499881/198553665-d2719c66-67d8-4fe6-a8c7-19d13bc4a30a.png width="35%" height='30%'/>

- 클래스 0과 나머지 사이에는 명확하게 분류가 가능하지만 1과 2는 다소 겹쳐있음을 확인할 수 있다. 
  
$\space$

### 3.1.2. 모델링 및 커널함수에 따른 차이 비교 
```python
def make_decision_boundary(X,y,classifier,kernel,resolution=0.02):
    #Decion boundary 생성 
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                          np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    return xx1,xx2,Z

def plot_decion_boundary(xx1,xx2,z,ax,X,y):
    #앞서 생성한 Decision boundary plot 
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    ax.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    ax.set_xlim(xx1.min(), xx1.max())
    ax.set_ylim(xx2.min(), xx2.max())
    ax.set_title(f'{kernel}',fontsize=32)
    # 샘플의 산점도 
    for idx, cl in enumerate(np.unique(y)):
        ax.scatter(x=X[y == cl, 0],
                   y=X[y == cl, 1],
                   alpha=0.8,
                   c=colors[idx],
                   marker=markers[idx],
                   label=cl,
                   edgecolor='black')

def model_metric(model,test_x,test_y):
    y_pred = model.predict(test_x)
    f1 = f1_score(test_y,y_pred,average='macro')
    precision = precision_score(test_y,y_pred,average='macro')
    recall  = recall_score(test_y,y_pred,average='macro')
    acc = accuracy_score(test_y,y_pred)
    return [acc,precision,recall,f1]

def model_train(kernel,data):
    (train_x, train_y, test_x, test_y) = data 

    model = SVC(C=1,kernel=kernel)
    model.fit(train_x,train_y)
    y_pred = model.predict(test_x)
    return model 


(X,Y),data = data_preprocess(df)

fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(ncols=2,nrows=2,figsize=(10, 10))

metric_list = [] 
for kernel,ax in zip(['linear','rbf','sigmoid','poly'],[ax1,ax2,ax3,ax4]):
    model = model_train(kernel,data)
    metric = model_metric(model,test_x,test_y)
    xx1,xx2,Z = make_decision_boundary(test_x,test_y,model,kernel)
    plot_decion_boundary(xx1,xx2,Z,ax,test_x,test_y)
    
    
    
```
- 제일 먼저 커널함수를 `linear`로 사용하여 모델을 만들었으며, 해당 모델의 성능과 Decision boundary plot은 아래와 같다. 
  
<p align = 'center'><img src = 'https://user-images.githubusercontent.com/92499881/198582855-02e34f99-de91-43c5-9f0e-ddeeb1223149.png'>

<figure>
  <p align = 'center'><img src = "https://user-images.githubusercontent.com/92499881/198582135-99adcedb-2782-42b2-b39f-f7102db7e4d3.png" width="70%" height='50%'/>
</figure>
 
 $\space$

### 3.1.3. 그래프 분석 
- Linear 커널을 사용한 경우 세 class 사이의 Decision boundary가 모두 선형으로 나타나 있는 것을 알 수 있다. 하지만 데이터들의 차원이 그리 크지 않고 복잡하지 않기 때문에 다른 커널과 비교했을 때 Metric에서 크게 밀리지 않는다. 
- RBF 커널과 Poly 커널함수를 사용한 경우 실제로도 Decision boundary가 비선형으로 형성된다는 것을 확인할 수 있었다. 하지만 데이터의 차원이 그리 크지 않은 탓에 Metric 면에서 크게 차이가 나지 않았다. 
- 이상하게도 sigmoid는 아예 분류를 하지 못했으며, Metric에서도 ACC : 0.233으로 매우 낮았다. 이는 Radnom 추출 보다도 못하다라 해도 무방한데, 이는 Sigmoid 함수가 Binary classification에 적합하여 Multi class인 이 케이스와는 적합하지 않기 때문이라 생각된다. 

$\space$

## 3.2 다른 분류 모델과의 비교 
- 앞선 과정에선 SVM 방식을 사용하되 Kernel Function의 종류에 따라 Decision boundary가 어떻게 바뀌나 확인했다. 
- 이번 파트에서는 RBF 커널을 사용하는 SVM과 다른 알고리즘을 사용한 모델들을 비교하고자 한다. 
- 비교하고자 하는 모델은 다음과 같다. 
  - Logistic 
  - SGD Classifier 
  - Random Forest 
  - Adaptive boost 
  - Gradient Boost 
  - Neural Network 
- 데이터는 앞선 과정과 동일하게 `iris` 데이터 사용 
### 3.2.1 모델링 및 결과 비교 
**Neural Network 학습 용 코드**
```python
import torch 
import torch.nn as nn 

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Sequential(
                                    nn.Linear(2,10),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(10)
                                    )
        self.fc1_1 = nn.Sequential(  
                                    nn.Linear(10,10),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(10))                                
        self.fc2 = nn.Linear(10,3)
    def forward(self,x):
        x = self.fc1(x)
        x = self.fc1_1(x)
        x = self.fc1_1(x)
        x = self.fc2(x)
        return x 

def nn_plot_decision_boundary(X,y,classifier,kernel,ax,resolution=0.2):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    #결정경계
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                            np.arange(x2_min, x2_max, resolution))
    input_x = np.array([xx1.ravel(), xx2.ravel()]).T
    Z = NN(torch.tensor(input_x).type(torch.float).to(device))
    y_pred = torch.argmax(Z,axis=1)
    Z = y_pred.detach().cpu().numpy().reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    plt.title('Neural Network')
    
    #샘플의 산점도 
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')
    plt.show()
fig,ax = plt.subplots(2,1)




device = 'cuda:0'
NN = Model().to(device)
optimizer = torch.optim.Adam(NN.parameters(),lr=1e-3)
criterion = nn.CrossEntropyLoss()
num_epochs = 20000

train_x, train_y = torch.tensor(train_x).type(torch.float).to(device),torch.tensor(train_y).type(torch.float).to(device)

for epoch in range(num_epochs):
    NN.train()
    optimizer.zero_grad()
    y_pred = NN(train_x)
    loss = criterion(y_pred,train_y.reshape(-1).type(torch.long))
    loss.backward()
    optimizer.step()
    
    print(loss)
    
y_pred = torch.argmax(NN(torch.tensor(test_x).to(device).type(torch.float)),axis=1).detach().cpu().numpy()    
```
**다른 모델 학습 및 Plot 코드**
```python
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier

(X,Y),data = data_preprocess(df)
(train_x, train_y, test_x, test_y) = data 
fig, ((ax1, ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(ncols=2,nrows=3,figsize=(13, 13))

metric_list = [] 

for model_func,model_name,ax in zip([LogisticRegression,SGDClassifier,RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier],['Logistic','SGD','RF','AB','GB'],[ax1,ax2,ax3,ax4,ax5]):
    model = model_func()
    model.fit(train_x,train_y)
    y_pred = model.predict(test_x)
    metric = model_metric(model,test_x,test_y)
    xx1,xx2,Z = make_decision_boundary(test_x,test_y,model,kernel)
    plot_decion_boundary(xx1,xx2,Z,ax,test_x,test_y,kernel=model_name)
    
nn_plot_decision_boundary(test_x,test_y,NN,ax6,'nn') 
        
```
<p align='center'><img src=https://user-images.githubusercontent.com/92499881/198611451-b9e4f2e8-6ac9-4ae2-bbf8-e662631cdb12.png width='80%',h`eight='50%'>`