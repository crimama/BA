Convolution Auto Encoder를 이용한 Anomaly Detection 에 관하여 몇가지 실험을 진행하고자 한다. Anomaly Detection using Convolution AutoEncoder 는 Reconstruction 방식의 Anomaly Detection 으로, 정상 데이터로만 학습 한 Auto Encoder는 test inference time에서 normal data는 복원을 잘 하고 anomaly data는 복원을 하지 못한다는 가정에 기반한다. 

Auto Encoder를 이용한 Anomaly detection 은 Reconstruction 뿐만 아니라 representation을 이용한 Machine learning을 통해서도 가능하다. AutoEncoder를 학습하게 되면 Encoder 와 Decoder 사이의 bottleneck에서는 Input image를 충분히 표현할 수 있도록 rerpresentation을 학습하게 된다. 학습이 끝난 뒤에는 Encoder를 통해서 Input image의 Representation을 출력할 수 있으며, 추출한 normal data의 representation을 이용하 여여 One class SVM과 같은 방법론을 이용해 normal training data의 boundary를 결정하게 된다. Test 데이터를 Inference할 때 normal trainin data의 boundary내에 있으면 normal, 바깥에 있으면 anomaly로 판단할 수 있게 된다. 



# 1. 실험 

## 1.1. 기본 베이스라인 

## 1.2. Augmentaation 차이 확인 
- Valid, Test는 Resize만 적용 
- Train에는 Augmentation 하나씩 적용하며 Metric 변화 확인  
- 적용되는 Augmentation은 아래와 같으며, 한 번에 한번씩 단일 augmentation 사용 
  - RandomCrop 
  - RandomAutoContrast 
  - RandomRotation
  - RandomGaussianBlur 
  - RandomSolarize
  - RandomVerticlal-horizontal flip 