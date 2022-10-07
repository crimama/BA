# **Dimensionality Reduction**  - 이론

<aside>
💡 해당 내용은 고려대학교 강필성 교수님의 비즈니스 애널리틱스 수업 일부를 정리했음을 미리 말씀 드립니다.
  
💡 해당 페이지에서는 이론적인 부분을 간단하게 다루며 각 파트 별 example notebook을 통해 더 자세하게 알아볼 예정입니다. 
</aside>

## 1. **Dimensionality Reduction Overview**

- 데이터 분석의 일련의 과정을 살펴 볼 때 본격적인 모델링 과정에 앞서 전처리를 하게 됩니다. 전처리 과정에선 정규화, 차원축소 등의 과정을 통해 효율적인 데이터셋을 구성하게 됩니다.
- 이 때 차원축소는 고차원의 데이터를 조금 더 모델링 하기 쉽게 저차원의 데이터로 변형주는 과정입니다. 또한 고차원의 데이터는 모델의 성능을 저하시키는 노이즈를 포함할 확률이 높기 때문에 데이터의 본질적인 정보만 갖도록 저차원으로 축소시키게 됩니다.
- 차원축소의 방법은 크게 **지도 방법**과 **비지도 방법**으로 분류가 됩니다.
<img src="https://github.com/crimama/BA/blob/main/1.Dimensionality%20Reduction/source/Untitled.png"  width="50%" height="50%"/>

- 지도 학습 기반의 차원 축소는 차원 축소 후 모델로부터 평가받은 결과를 다시 반영하여 반복적으로 진행하여 최적의 차원을 찾는 방법입니다.


<img src="https://github.com/crimama/BA/blob/main/1.Dimensionality%20Reduction/source/Untitled%201.png"  width="50%" height="50%"/>

- 반면 비지도 차원 축소는 성능의 반영, 반복적인 과정 없이 일련의 계산을 통해 바로 최적의 차원을 찾게 됩니다. 이 일련의 계산에서는 원본 데이터의 분산, 거리와 같은 성질을 최대한 보존하는 좌표계를 찾는 과정이 해당하게 됩니다.
- 차원축소의 방법은 **변수 선택** or **변수 추출**과 같은 테크닉에 따라서도 분류가 가능합니다.
- 변수 선택은 원본 변수 셋에서 특정 부분 집합 변수 셋만을 선택하는 것이고
- 반대로 변수 추출의 경우 기존의 원본 데이터의 특징을 유지하면서 더 작은 차원의 변수를 생성하는 것을 의미합니다.


<img src="https://github.com/crimama/BA/blob/main/1.Dimensionality%20Reduction/source/Untitled%202.png"  width="50%" height="50%"/>

## 2. Supervised Methods : Genetic algorithm 

- [Example](https://github.com/crimama/BA/blob/main/1.Dimensionality%20Reduction/1.Genetic.ipynb)
- 여기서는 지도 학습 이며 변수 선택 방법론을 다룹니다. 그 중 유전 알고리즘에 대해 자세히 다룰 예정입니다.
- 우선 유전 알고리즘에 대해 다루기 전에 전진 선택법, 후방 선택법, 단계적 선택법에 대해 간단하게 다룬 뒤 유전 알고리즘에 대해 다루겠습니다.

## 3. Unsupervised Method

### 3.1. Linear embedding
  - PCA
  - MDS [Example](https://github.com/crimama/BA/blob/main/1.Dimensionality%20Reduction/2.MDS.ipynb)
### 3.2. Nonlinear embedding [Example](https://github.com/crimama/BA/blob/main/1.Dimensionality%20Reduction/3.non_linear.ipynb)
  - ISOMAP 
  - LLE 
  - t-SNE 
