# SVM

## 1.이론적 배경

**Support Vector Machine**은 벡터 공간에서 다른 클래스의 데이터들간 가장 잘 나눌 수 있는 초 평면, 결정 경계를 찾는 것을 목적으로 한다. 분류되지 않은 새로운 데이터가 나타났을 때 이 경계를 기준으로 어느 위치에 있는지 확인하여 분류 과제를 수행할 수 있다. 즉 SVM은 이 결정 경계를 어떻게 정의하고 계산하는 것이 매우 중요하다. 
<p align="center"><img src="https://user-images.githubusercontent.com/92499881/195275527-08bc4c5c-aa7f-4d5d-98e7-d6a21be654f2.png"  width="30%" height="30%"/>

가운데 실선이 결정 경계이며, 실선으로 부터 가장 가까운 검은색 테두리의 빨간색 점 그리고 파란색 점을 지나는 점선이 존재한다. 결정경계부터 점선까지의 거리를 마진(margin)이라고 한다. SVM은 결정 경계를 만들 때 Margin을 최대화 하는 결정 경계를 찾고자 한다. 이러한 이유는 모델의 구조적 위험 때문이다. 

**구조적 위험**
  
  분류 모형은 모형의 복잡도가 증가할 수록 주어진 학습 데이터를 더 잘 분류할 수 있게 되지만 되려 데이터가 갖고 있는 노이즈까지 학습하게 된다. 이런 경우 학습으로 사용한 Train데이터에 대해서는 좋은 성능을 보이지만 새로운 데이터가 들어왔을 때는 좋은 성능을 보이지 못한다. 따라서 분류 모델을 만들 경우 학습 과정에서 학습 오류를 낮춤과 동시에 모델의 복잡도를 줄여야 한다. 이 둘을 합쳐 구조적 위험이라 한다. 
  
 <p align="center"><img src = "https://user-images.githubusercontent.com/92499881/195276434-0294a827-6b88-47b4-89a5-e3a55a9e6dd7.png" width="30%" height='30%'/>

모델의 성능과 복잡도는 Trade-off 관계를 가지며 동일한 정확도라면 상대적으로 복잡도가 낮은 모델이 선호되어야 한다. 이는 수식적으로 풀이가 가능한데..
   <p align="center"><img src = "https://user-images.githubusercontent.com/92499881/195277956-db9be88e-fb44-4c1f-8bba-f2dc99a1bea6.png" width="30%" height='30%'/>
     
   - VC dimension : 함수 H에 의해 최대로 shattered 될 수 있는 points의 수, 어떤 함수의 복잡도,Capacity를 측정하는 지표  
   - 위 이미지에서 h가 VC dimension, 복잡도를 의미하며 이것이 커질 수록 R[f] term 전체가 증가하게 되고 위험도가 증가하게 된다. 
   - 반대로 h가 낮아지거나 데이터의 양(n)이 많아질 수록 R[f] term 전체가 감소하고 리스크가 감소하게 된다. 
   - 마진을 최대화 할 경우 데이터를 분류할 수 있는 경계면의 수가 감소하게 되고 이는 VC dimension의 감소를 의미한다. 
   - 즉 마진 최대화 -> Vc dimension 감소 -> 구조적 위험도 감소로 수렴하게 된다. 
     
## 2. SVM 도출 과정 
### 2.1 Hard margin SVM, linear case
     
**목적함수정의**   
- Emperical risk는 d차원의 데이터를 나누는 d-1차원의 hyperplane을 구하는 문제로, 아래의 식에서 w와 b를 구하는 것이 목적이다. 

$$
H = \{x -> sign(w \cdot x + b : w \in R^d, b \in R)\}
$$

- 이 수식을 통해 도출되는 w와 b의 조합이 classification boundary이며 이를 통해 나오는 결과 값이 + 또는 -의 부호를 갖게 되며 이 부호가 각 데이터 포인트들이 속하는 클래스를 의미하게 된다.      
- Classification Boundary는 하나가 아니며 여러 boundary 가운데 VC dimension을 최소화 하는 boundary를 선택하게 된다. 
- 마진과 VC Dimension 간에는 아래와 같은 관계가 성립한다. 

$$
h \leq min([{R^2 \over \delta^2}], D) +1      
$$
     
- 위 식에서 R은 hyperplane의 반지름으로 모든 데이터를 감싸는 원을 그렸을 때 반지름을 의미한다. 
- 이 식에 따르면 마진($\delta$)이 커지는 것은 곧 VC dimension(h)가 작아지는 것을 의미한다. 
- 이를 다시 목적 함수로 정의를 하면 
     
$$
Objective function: min {1 \over 2}||w||^2
$$ 
     
**제약조건, 최적화** 
     
- 이에 대하여 제약식이 존재하는데 $s.t.y_i(w^Tx_i +b) \geq 1$ 이며 이는 어떤 feature set x의 벡터가 주어졌을 때 이를 + 또는 -로 분류하는 hyperplane을 의미하며 위의 목적함수는 제약식을 만족하는 hyperplane 중 마진을 최대화 하는 최적값을 찾는 것을 의미한다. 

- 제약이 있는 최적화 문제를 풀기 위해 라그랑지앙 승수법을 사용한다. 
     
$$
minL_p(w,b,\alpha_i) = {1\over 2}||w||^2 - {\sum}^N_{i=1} \alpha_i(y_i(w^Tx_i +b)-1)
$$
     
- L을 미지수 w와 b로 각각 편미분한 값이 0이 되는 곳에서 최소값을 가지므로 이를 정리하면 w와 b를 a,x,y에 대한 식으로 정리할 수 있다. 
- 이를 목적 함수에 넣어 정리하면 a,x,y에 대한 식으로 정리가 가능하며 이 때 x와y는 이미 주어진 값이므로 SVM 문제를 미지수 $\alpha$의 이차 방정식을 푸는 문제로 정의할 수 있다. 
     
$$
max L_D(\alpha_i) = \sum^N_{i=1}\alpha_i - {1\over 2}\sum^N_{i=1}\sum^N_{j=1}\alpha_i\alpha_jy_iy_jx_i^Tx_js.t. \sum^N_{i=1}\alpha_iy_i = 0, \alpha_i \geq 0
$$
     
- 여기서 KKTcondition에 따라 아래와 같은 수식이 성립한다. 
     
$$
kkt\space condition : {\alpha L_p \over {\alpha w}} = 0 => w = \sum^N_{i=1}\alpha_i y_i x_i
$$ 

$$
\alpha_i(y_i(w^Tx_i+b)-1) =0
$$ 
- 즉 $\alpha$가 0 -> y(wx+b)-1 != 0 
- 또는 y(wx+b)-1 = 0 -> $\alpha$ != 0 
        
- 정리하면 마진을 최대화 하는 하이퍼 평면을 찾기 위해 데이터와 라그랑지앙 승수를 이용해 w와 b를 구한다. 그리고 이 때 $\alpha$가 0이 아닌 벡터들을 일컬어 Support Vector라고 한다. 
