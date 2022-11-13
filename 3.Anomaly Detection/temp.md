Convolution Auto Encoder를 이용한 Anomaly Detection 에 관하여 몇가지 실험을 진행하고자 한다. Anomaly Detection using Convolution AutoEncoder 는 Reconstruction 방식의 Anomaly Detection 으로, 정상 데이터로만 학습 한 Auto Encoder는 test inference time에서 normal data는 복원을 잘 하고 anomaly data는 복원을 하지 못한다는 가정에 기반한다. 

Auto Encoder를 이용한 Anomaly detection 은 Reconstruction 뿐만 아니라 representation을 이용한 Machine learning을 통해서도 가능하다. AutoEncoder를 학습하게 되면 Encoder 와 Decoder 사이의 bottleneck에서는 Input image를 충분히 표현할 수 있도록 rerpresentation을 학습하게 된다. 학습이 끝난 뒤에는 Encoder를 통해서 Input image의 Representation을 출력할 수 있으며, 추출한 normal data의 representation을 이용하 여여 One class SVM과 같은 방법론을 이용해 normal training data의 boundary를 결정하게 된다. Test 데이터를 Inference할 때 normal trainin data의 boundary내에 있으면 normal, 바깥에 있으면 anomaly로 판단할 수 있게 된다. 

Auto Encoder를 이용하여 Reconstruction 방식과 Machine learning 방식을 이용해 Augmentation이 Anomaly detection에 어떤 영향을 끼치고자 확인하고자 한다. Augmentation이란 학습 전에 이미지에 가하는 일종의 변형, 처리로 Rotate, Contrast, Color, Flip 등의 변형을 가해 적은 수의 이미지 이더라도 많은 수의 이미지가 있는 것과 유사한 효과를 내도록 하는 방식이다. 일반적으로 이미지 태스크에서 모델을 학습하기 위해선 항상 적용되는 방식으로 

- 인코더는 Convolution 과 Linear로만 구성되어 있으며, Convolution은 커널사이즈 3, strides=2로 구성하여 Downsampling 되도록 하였다. 
- 디코더는 Upsampling을 위해 Convolution Transpose를 사용하였으며 인코더에서 출력된 임베딩 벡터를 Convolution layer에 맞게 변형하도록 linear projection, unflatten을 사용하였다. 
- loss function은 l2 norm : mse loss를 사용하였으며 Input 이미지와 복원된 Output 이미지 간의 차이를 통해 loss를 계산하였다. 