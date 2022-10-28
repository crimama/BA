# SVM

- Support Vectors Machine(SVM)에 관한 자세한 이론은 [theory](theory.md) 에서 다루며 해당 페이지에서는 SVM에 대한 Overview + tutorial에 대해 다룹니다 

# Table of Contents 
  - [1.이론적 배경](#1이론적-배경)

  - [2.SVM Cases](#2-svm-cases)
  
  - [3.kernel svm](#case-3--linear--soft-margin-svm)

  - [Appendix : 이론적 배경](theory.md)
 
# 1.이론적 배경

**Support Vector Machine**은 벡터 공간에서 다른 클래스의 데이터들간 가장 잘 나눌 수 있는 초 평면, 결정 경계를 찾는 것을 목적으로 한다. 분류되지 않은 새로운 데이터가 나타났을 때 이 경계를 기준으로 어느 위치에 있는지 확인하여 분류 과제를 수행할 수 있다. 즉 SVM은 이 결정 경계를 어떻게 정의하고 계산하는 것이 매우 중요하다. 가운데 실선이 결정 경계이며, 실선으로 부터 가장 가까운 검은색 테두리의 빨간색 점 그리고 파란색 점을 지나는 점선이 존재한다. 결정경계부터 점선까지의 거리를 마진(margin)이라고 한다. SVM은 결정 경계를 만들 때 Margin을 최대화 하는 결정 경계를 찾고자 한다. 이러한 이유는 모델의 구조적 위험 때문이다.  
<p align="center"><img src="https://user-images.githubusercontent.com/92499881/195275527-08bc4c5c-aa7f-4d5d-98e7-d6a21be654f2.png"  width="30%" height="30%"/>


**구조적 위험**

분류 모형은 모형의 복잡도가 증가할 수록 주어진 학습 데이터를 더 잘 분류할 수 있게 되지만 되려 데이터가 갖고 있는 노이즈까지 학습하게 된다. 이런 경우 학습으로 사용한 Train데이터에 대해서는 좋은 성능을 보이지만 새로운 데이터가 들어왔을 때는 좋은 성능을 보이지 못한다. 따라서 분류 모델을 만들 경우 학습 과정에서 학습 오류를 낮춤과 동시에 모델의 복잡도를 줄여야 한다. 이 둘을 합쳐 구조적 위험이라 한다. 
 <figure class="half">
    <img src = "https://user-images.githubusercontent.com/92499881/195276434-0294a827-6b88-47b4-89a5-e3a55a9e6dd7.png" width="40%" height='40%'/>
    <img src = "https://user-images.githubusercontent.com/92499881/195277956-db9be88e-fb44-4c1f-8bba-f2dc99a1bea6.png" width="50%" height='50%'/>
 </figure>
  모델의 성능과 복잡도는 Trade-off 관계를 가지며 동일한 정확도라면 상대적으로 복잡도가 낮은 모델이 선호되어야 한다. 이는 수식적으로 풀이가 가능한데.
     
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

#  Case 3 : Linear & soft Margin SVM
  
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

## SVM 실험 