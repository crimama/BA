# Anomaly Detection 
- Part 1에서는 Anomaly Detection과 관련된 이론에 대해서 다룹니다. 
- Part 2에서는 Anomaly Detection과 Augmentation에 관련된 실험을 진행합니다.

# Table of Contents 
- [Anomaly Detection](#anomaly-detection)
- [Table of Contents](#table-of-contents)
- [1. Anomaly Detection 이론](#1-anomaly-detection-이론)
  - [1.1. Anomaly Data의 정의 및 특성](#11-anomaly-data의-정의-및-특성)
  - [1.2. Autoencoder](#12-autoencoder)
  - [1.3 Autoencoder를 이용한 Anomaly Detection](#13-autoencoder를-이용한-anomaly-detection)
- [2. 실험 : Augmentation에 따른 Anomaly Detection 성능 비교](#2-실험--augmentation에-따른-anomaly-detection-성능-비교)
  - [2.0. 실험 세팅](#20-실험-세팅)
  - [2.1. 베이스라인](#21-베이스라인)
- [3. 결과](#3-결과)
  - [3.1 Preprocess : Augmentation을 모델 학습 전에 적용 시킨 경우](#31-preprocess--augmentation을-모델-학습-전에-적용-시킨-경우)
  - [3.2 Postprocess : 모델 학습 후 Test 데이터에 Augmentation을 적용하는 경우](#32-postprocess--모델-학습-후-test-데이터에-augmentation을-적용하는-경우)
  - [3.3 Mixed : Preprocess + Postprocess](#33-mixed--preprocess--postprocess)
- [4. 결론](#4-결론)
$\space$

# 1. Anomaly Detection 이론

<p align='center'><img src = 'https://user-images.githubusercontent.com/92499881/201667850-4c6a54ad-94c5-43bb-93fe-df0194236f93.png' width='40%',height='40%'>

<center>대표적인 Anomaly Detection Dataset : MVtecAD</center>  
<center>같은 Anomaly로 분류되더라도 그 세부적인 형태는 모두 다르다.</center>

$\space$

## 1.1. Anomaly Data의 정의 및 특성
- Anomaly Detection 이란 단어 그대로 전체 데이터 중 이상치(Anomaly)를 탐지하는 것을 목적으로 하는 태스크를 말한다. 이상치(Anomaly)는 정상 데이터 분포 속 매우 낮은 확률로 나타나는 데이터를 말하며, 어떻게 정의하냐에 따라 Outlier, Novelty로 불리기도 한다. 
  
- 일반적인 분류 방법을 통해 탐지를 할 수 없는데, 이는 Anomaly의 고유의 특성 때문이다. Anoamly Data의 특징은 크게 두가지를 뽑을 수 있는데, 첫번째는 Imbalanced 하다는 점이다. 일반적으로 정상 데이터 분포 속 이상치는 매우 낮은 확률로 발생하게 되고 이상치 데이터를 수집했다 하더라도 그 수는 정상 데이터 수에 비해 굉장히 낮다. 따라서 일반적인 분류 방법으로 모델을 만들 경우 이상치에 대한 정보를 모델은 학습할 수 없고 normal - anomal 간의 decision boundary를 만들 수 없게 된다. 
  
- 두번째 특징은 anomaly들은 모두 각기 다른 특징을 갖고 있다는 점이다. Anomaly Detection task에서는 데이터를 normal - anomaly로 이진 분류를 하지만 실제 Anomaly data 내에서는 각 데이터 마다 다른 특징들을 갖고 있다. Anomaly가 발생하는 이유는 굉장히 다양하며, 그 다양한 이유에서 발생되는 Anomaly Data 또한 각기 다른 특징을 갖게 된다. 
- 따라서 Anomaly Detection 방법론은 대부분 Anomaly 가 없는 Anomaly-free normal 데이터로만 학습을 하여 Normal data의 manifold를 형성한 뒤 새로운 데이터가 들어왔을 때 normal 데이터의 범주 안에 있지 않으면 Anomaly로 판별하는 식의 방법을 사용한다. 

**Anomaly Detection Method** 
- Anomaly Detection의 방법론은 크게 3개로 분류할 수 있다. 
  - Density-based 
  - Distance-based 
  - Model-based 
- 위 방법론 범주 중 여기서 다루고자 하는 것은 Model-base의 **Autoencoder** 이다. 

$\space$

## 1.2. Autoencoder

<figure class='half'>
    <p align='center'><img src = 'https://user-images.githubusercontent.com/92499881/201670009-d3e2128d-7d5f-4ab8-b412-a920c55c17a1.png' width='45%',height='50%'>
    <img src = 'https://user-images.githubusercontent.com/92499881/201671565-a7e5bf68-de12-466b-9c54-03002d93fab9.png' width='45%',height='50%'>
</figure>
<center>좌 : Autoencoder 구조 | 우 : Mnist를 이용하여 Autoencoder 학습한 결과</center>
<center>출처 : 강필성 교수님 비즈니스 애널리틱스 강의 자료</center>

$\space$

- Autoencoder란 대칭 구조를 갖고 있는 인코더와 디코더를 연결한 인공 신경망 구조로, 입력과 출력이 동일하다. 
- 인코더는 Input 데이터를 Embedding Vector를 압축하며, 디코더는 압축된 Embedding vector를 Input으로 받아 원래 데이터로 복원하는 구조로 학습이 진행된다. 
- 오토 인코더는 반드시 입력 변수의 수 보다 은닉 노드의 수가 더 적은 은닉 층이 있어야 하는데 이를 bottleneck layer 라고 한다. 
- 이러한 구조를 갖는 이유는 동일한 수의 노드를 갖게 되면 해당 Bottleneck layer는 단순히 입력을 외우게 되며 복원을 위한 Feature를 학습할 수 없게 되며, 오버피팅이 되는 것과 동일한 현상이 나타난다. 
- 기본적인 layer로는 MLP를 사용하지만 Convolution layer를 이용하여 Convolution Autoencoder를 만들 수 있다. 이 경우 디코더는 Convolution Transpose를 사용하거나 Upsampling 을 사용하여 데이터를 복원하게 된다. 

$\space$

## 1.3 Autoencoder를 이용한 Anomaly Detection 
<figure class='half'>
    <p align='center'><img src = 'https://user-images.githubusercontent.com/92499881/201673002-ee8d6909-9a0f-48f7-8d04-f5ae8d1938c8.png' width='45%',height='50%'>
    <img src = 'https://user-images.githubusercontent.com/92499881/201673015-31544a86-cc4e-403c-a6e0-fd013811a586.png' width='45%',height='50%'>
</figure>
<center>좌 : Reconstruction 방식 | 우 : Machine Learning 방식</center>
<center>출처 : 강필성 교수님 비즈니스 애널리틱스 강의 자료</center>

$\space$

- Autoencoder를 이용한 Anomaly Detection은 다양한 방법론이 있지만 학습된 Autoencoder를 어떻게 사용하냐에 따라 크게 두가지로 나눌 수 있다. 
    1. 학습된 Autoencoder를 모두 사용하는 Reconstruction 방식 
    2. 학습된 Encoder만을 이용하는 Machine Learning 방식 
```
Reconstruction Method
```
- Reconstruction 방식이란 Autoencoder를 통해 이미지를 Inference시킬 경우 복원된 이미지가 원래 이미지와의 차이를 이용해 Anomaly Detection을 하는 방식이다. Autoencoder는 Anomaly-free normal 데이터로만 학습 시키며 따라서 normal에 대한 정보만을 학습하게 된다. 이 경우 Autoencoder가 경험해 보지 못 한 Anomaly를 갖는 데이터가 input으로 들어오는 경우 정상적으로 복원 하는 것에 실패하게 된다. 
- 즉 Input 데이터와 Autoencoder 가 복원한 Output 데이터 간의 차이를 이용하여 Anomaly Detection을 수행하게 되며 l2 Norm, Cosine Similarity 등을 이용하여 Anomaly Score를 계산하게 된다. 
- 원본 이미지와 복원된 이미지 간의 차이를 Pixel 단위로 측정하여 Anomaly score를 산출할 수 있으므로 어느 특정 region, pixel이 anomaly인지 쉽게 알 수 있다는 장점이 있다. 
- 하지만 입력에 대한 약간의 변형에도 모델이 민감하게 반응하며, 최근에는 Test data 역시도 정상적으로 복원 해 내는 문제점이 두드러지고 있다. 
  - Reconstruction 방식의 경우 Normal데이터는 잘 복원하고 Test 데이터는 잘 복원하지 못 한다는 가정 하에 Anomaly Detection이 수행 되지만 Test data도 잘 복원 될 경우 이 가정이 깨지므로 정상적으로 Anoamly Detection을 수행할 수 없다. 
```
Machine learning
```
- Machine learning 방식이란 학습된 Autoencoder의 Encoder를 통해 얻어진 Embedding vectors를 이용하여 Anomaly detection을 수행하는 방식이다. 
- 앞서 Reconstruction 방식과 마찬가지로 Anomaly-free normal 데이터로만 학습 된 Autoencoder를 사용하며 해당 Autoencoder의 encoder는 Input 이미지의 Feature를 추출하는 역할을 하게 된다. 
- 추출된 Features들은 Input 데이터의 Representation으로 사용되며 OC-SVM 이나 Isolation forest, SVDD와 같은 Machine learning 기반의 Anomaly detection 방법론을 통해 Anomaly Detection을 수행하게 된다. 
  





$\space$

# 2. 실험 : Augmentation에 따른 Anomaly Detection 성능 비교  
- Augmentation이란 한정된 데이터의 양을 늘리기 위해 원본에 각종 변환을 적용하여 데이터를 증강시키는 기법입니다. 일반적으로 학습 전 이미지에 변형을 가하는 전처리 형식으로 사용 됩니다. 
- 이 Augmentation이 Anomaly Detection에 적용되었을 때 성능이 어떻게 변화하는지 확인하고자 합니다. 
- 사용된 베이스라인은 AutoEncoder와 OC-SVM으로, OC-SVM은 학습된 AutoEncoder의 인코더로 추출된 임베딩 벡터를 인풋으로 사용합니다. 
- 실험을 통해 확인하고자 하는 바는 아래와 같습니다. 
```
    1. Preprocess : 학습 전 Augmentation 적용 및 종류에 따른 성능 변화 
    2. Postprocess : Augmentation 없이 오토인코더 학습시킨 뒤 Test에 Augmentation을 적용시켜 Anomaly Detection을 수행시킬 때 성능 변화  
    2. Mixed process : Preprocess 와 Postprocess를 모두 적용 시킬 경우의 성능 변화 
```
$\space$

## 2.0. 실험 세팅 
**Train** 
- 기본 Baseline model을 Convolution Auto Encoder를 사용하며 loss function으로는 reconstruction error : l2 norm을 사용합니다. 사용하는 데이터셋은 MVtecAD의 hazelnut 데이터셋과 Carpet 데이터셋을 사용합니다. 
  
- Auto Encoder를 학습시킬 때는 Anomaly-free Training data를 사용하며 Test data inference 시에는 Normal - Anoaml 만 분류합니다. (세부적인 Anomaly category 무시)
  
- 적용되는 Augmentation 종류는 아래와 같습니다. 
  - Identical (No any Augmentation)
  - RandomCrop 
  - RandomAutoContrast 
  - RandomRotation
  - RandomGaussianBlur 
  - RandomSolarize
  - RandomVerticlal-horizontal flip 
  
**Anomaly Detection**
1. Reconstruction 
   - Reconstruction 방식은 Test data 추론 시 원본 이미지와 복원된 이미지 간의 l2 norm(Reconstruction error)을 계산하여 이를 Anomaly Score로 사용합니다. 
2. Machine Learning 
   - Machine Learning 방식은 Reconstruction 방식과 마찬가지로 AutoEncoder를 학습시킨 뒤 Encoder를 이용해 각 Test 이미지의 Embedding Vector를 추출하여 이를 Input으로 사용하 Machine learning에 적용합니다. 
   - 이 경우 먼저 Normal Train 데이터로 OC-SVM을 학습시킨 뒤 Test 데이터를 Inference합니다. 
- Metric 
  - 성능 비교를 위해 **AUROC**를 사용합니다. 
  - Normal 과 Anomal간의 class 비율이 imbalance하며 anomaly score를 normal과 anomal로 구분할 Threshold를 따로 결정하지 않더라도 성능을 비교할 수 있기 때문에 선택하였습니다. 

$\space$

## 2.1. 베이스라인 
**전처리 및 데이터 로더**
- 디렉토리로 부터 이미지 디렉토리 및 라벨 로드
- Train - Valid split 
- 이미지 디렉토리와 라벨을 이용해 데이터셋 및 로더 생성 
- Augmentation은 Train 데이터에만 적용 됨 
```python
#Datadir_init : 데이터 폴더로 부터 이미지 디렉토리와 라벨 가져오는 클래스 
class Datadir_init:
    def __init__(self,Dataset_dir='./Dataset/hazelnut'):
        self.Dataset_dir = Dataset_dir 
        
    def test_load(self):
        test_label_unique = pd.Series(sorted(glob(f'{self.Dataset_dir}/test/*'))).apply(lambda x : x.split('/')[-1]).values
        test_label_unique = {key:value for value,key in enumerate(test_label_unique)}
        self.test_label_unique = test_label_unique 

        test_dir = f'{self.Dataset_dir}/test/'
        label = list(test_label_unique.keys())[0]

        test_img_dirs = [] 
        test_img_labels = [] 
        for label in list(test_label_unique.keys()):
            img_dir = sorted(glob(test_dir +f'{label}/*'))
            img_label = np.full(len(img_dir),test_label_unique[label])
            test_img_dirs.extend(img_dir)
            test_img_labels.extend(img_label)
        return np.array(test_img_dirs),np.array(test_img_labels) 

    def train_load(self):
        train_img_dirs = sorted(glob(f'{self.Dataset_dir}/train/good/*.png'))
        return np.array(train_img_dirs) 

#MVtecADDataset : 앞서 읽어온 이미지 디렉토리와 라벨 데이터를 Datset으로 만들어 주는 클래스 
class MVtecADDataset(Dataset):
    def __init__(self,cfg,img_dirs,labels=None,Augmentation=None):
        super(MVtecADDataset,self).__init__()
        self.cfg = cfg 
        self.dirs = img_dirs 
        self.augmentation = self.__init_aug__(Augmentation)
        self.labels = self.__init_labels__(labels)

    def __len__(self):
        return len(self.dirs)

    def __init_labels__(self,labels):
        if np.sum(labels) !=None:
            return labels 
        else:
            return np.zeros(len(self.dirs))
    
    def __init_aug__(self,Augmentation):
        if Augmentation == None:
            augmentation = transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Resize((self.cfg['img_size'],self.cfg['img_size']))
                                                #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                            ])
        else: 
            augmentation = Augmentation 
        return augmentation                                      

    def __getitem__(self,idx):
        img_dir = self.dirs[idx]
        img = Image.open(img_dir)
        img = self.augmentation(img)

        if np.sum(self.labels) !=None:
            return img,self.labels[idx] 
        else:
            return img

#이미지 디렉토리 및 라벨 로드 
Data_dir = Datadir_init()
train_dirs = Data_dir.train_load()
test_dirs,test_labels = Data_dir.test_load()

#augmentation 
augmentation = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Resize((256,256))
    ])    

#Train-Valid split 
indx = int(len(train_dirs)*0.8)
train_dset = MVtecADDataset(cfg,train_dirs[:indx],Augmentation=augmentation)
valid_dset = MVtecADDataset(cfg,train_dirs[indx:])
test_dset = MVtecADDataset(cfg,test_dirs,test_labels)

#DataLoader 
train_loader = DataLoader(train_dset,batch_size=cfg['batch_size'],shuffle=True)
valid_loader = DataLoader(valid_dset,batch_size=cfg['batch_size'],shuffle=False)
test_loader = DataLoader(test_dset,batch_size=cfg['batch_size'],shuffle=False)            
```

  
**Augmentation 구성**
- Augmentation은 아무것도 적용하지 않는 경우까지 포함해 총 7개 경우의 수가 있음 
- 각 Augmentation 구성은 1개의 Augmentation으로만 이루어져 있으며 Random Crop을 제외한 나머지는 모두 256으로 Resize되게 구성 됨  
- 위 코드에서는 예시로 Baseline Augmentation을 작성 하였지만 실제 실험에서는 아래 Augmentation을 하나씩 바꿔가며 진행 
```python
def create_transformation(cfg):
    
    aug1 = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.RandomCrop((256,256))
    ])
    
    aug2 = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.RandomAutocontrast(),
                            transforms.Resize((cfg['img_size'],cfg['img_size']))
    ])

    aug3 = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.RandomRotation(degrees=20),
                            transforms.Resize((cfg['img_size'],cfg['img_size']))
    ])
    aug4 = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.GaussianBlur(11),
                            transforms.Resize((cfg['img_size'],cfg['img_size']))
    ])

    aug5 = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.RandomSolarize(0.1),
                            transforms.Resize((cfg['img_size'],cfg['img_size']))
    ])
    aug6 = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomVerticalFlip(0.5),
                            transforms.Resize((cfg['img_size'],cfg['img_size']))\
    ])                                           
    aug_default = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Resize((256,256))
    ])                        

    Transformation_set = {} 
    Transformation_set = { key:value for key,value in enumerate([aug1,aug2,aug3,aug4,aug5,aug6,aug_default])}
    return  Transformation_set[cfg['aug_number']]
```
**모델구성** 
- Convolution Autoencoder는 대칭 구조로 인코더와 디코더로 구성되어 있음 
- 인코더는 Convolution layer를 기본으로 하며 `stride=2` 를 통해 Input 이 Downsampling이 되고 linear projection을 통해 embedding vector를 추출 함 
- 디코더는 Conovlution Transpose layer를 기본으로 하며 Input으로 embedding vector를 받으며 Convolution transpose에 적용할 수 있도록 reshape, unflatten을 거친뒤 ConvTranspose(`stride=2`)를 통해 Upsampling이 된다. 
  
```python
class MVtecEncoder(nn.Module):
    def __init__(self,encoded_space_dim):
        super(MVtecEncoder,self).__init__()

        self.encoder_cnn = nn.Sequential(
                                        nn.Conv2d(in_channels=3,out_channels=8,kernel_size=3,stride=2,padding=1),
                                        nn.BatchNorm2d(8),
                                        nn.ReLU(),
                                        nn.Conv2d(in_channels=8,out_channels=16,kernel_size=3,stride=2,padding=1),
                                        nn.BatchNorm2d(16),
                                        nn.ReLU(),
                                        nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=2,padding=1),
                                        nn.BatchNorm2d(32),
                                        nn.ReLU(),
                                        nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=2,padding=1),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(64),
                                        nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=2,padding=1),
                                        nn.ReLU()
)

        self.flatten = nn.Flatten(start_dim=1)
        self.encoder_lin = nn.Sequential(
                                        nn.Linear(8*8*128,512),
                                        nn.ReLU(True),
                                        nn.Linear(512,encoded_space_dim)
                                        )
        

    def forward(self,x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x 

class MVtecDecoder(nn.Module):
    def __init__(self,encoded_space_dim):
        super(MVtecDecoder,self).__init__()       

        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim,512),
            nn.ReLU(True),
            nn.Linear(512,7*7*128),
            nn.ReLU(True)
        )
        self.unflatten = nn.Unflatten(dim=1,unflattened_size=(128,7,7))      

        self.decoder_cnn = nn.Sequential(
                            nn.ConvTranspose2d(128,64,3,stride=2,output_padding=1),
                            nn.BatchNorm2d(64),
                            nn.ReLU(True),
                            nn.ConvTranspose2d(64,32,3,stride=2,padding=1,output_padding=1),
                            nn.BatchNorm2d(32),
                            nn.ReLU(True),
                            nn.ConvTranspose2d(32,16,3,stride=2,padding=1,output_padding=1),
                            nn.BatchNorm2d(16),
                            nn.ReLU(True),
                            nn.ConvTranspose2d(16,8,3,stride=2,padding=1,output_padding=1),
                            nn.BatchNorm2d(8),
                            nn.ReLU(True),
                            nn.ConvTranspose2d(8,3,3,stride=2,padding=1,output_padding=1)
        )
    def forward(self,x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_cnn(x)
        x = torch.sigmoid(x)
        return x 

class Convolution_Auto_Encoder(nn.Module):
    def __init__(self,Encoder,Decoder,encoded_space_dim ):
        super(Convolution_Auto_Encoder,self).__init__()
        self.encoder = Encoder(encoded_space_dim)
        self.decoder = Decoder(encoded_space_dim)

    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x         
```

**학습 전 객체 생성**
- 학습에 필요한 객체들을 생성해 줌 
```python 
#Config 에서 설정한 space_dim(bottleneck size) 대로 오토인코더 생성 
model = Convolution_Auto_Encoder(MVtecEncoder,MVtecDecoder,cfg['encoded_space_dim']).to(cfg['device'])
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=100,eta_min=0) #optimizer learning rate 조절 용 Scheduler 

```
**모델 학습**

```python
total_train_loss = [] 
total_valid_loss = [] 
best_valid_loss = np.inf 
for epoch in tqdm(range(cfg['Epochs'])):
    model.train() 
    optimizer.zero_grad()
    train_loss = [] 
#Train Process 
    for img,_ in train_loader:
        img = img.to(cfg['device']).type(torch.float32)
        y_pred = model(img).type(torch.float32)
        loss = criterion(img,y_pred)

        #Backpropagation 
        loss.backward()
        optimizer.step()
        train_loss.append(loss.detach().cpu().numpy())
    scheduler.step()        

#Valid process 
    model.eval()
    valid_loss = [] 
    with torch.no_grad():
        for img,_ in valid_loader:
            img = img.to(cfg['device'])
            y_pred = model(img)
            loss = criterion(img,y_pred)
            valid_loss.append(loss.detach().cpu().numpy())
    print(f'\t epoch : {epoch+1} valid loss : {np.mean(valid_loss):.3f}')
    #시각화 
    fig, (ax1,ax2) = plt.subplots(ncols=2,nrows=1,figsize=(5, 5))
    ax1.imshow(img[0].detach().cpu().permute(1,2,0).numpy())
    ax2.imshow(y_pred[0].detach().cpu().permute(1,2,0).numpy())
    plt.show()
```
**학습 후 테스트 Inference**
- 학습은 오토인코더 한개만 진행 되지만 Metric은 두가지가 진행 됨 
- **Reconstruction** 방식과 **Machine learning : OC-SVM** 
- 동일한 학습된 오토인코더를 사용하되 OC-SVM은 인코더만 사용 함
```python
#오토인코더를 모두 사용하는 Reconstruction과 Encoder만을 사용하는 Machinelearning 모두 진행 
encoder = model.encoder 
Pred_imgs = [] 
Pred_vecs = [] 
True_imgs = [] 
True_labels = []
for img,label in test_loader:
    img,label = img.to(cfg['device']).type(torch.float32),label.to(cfg['device']).type(torch.float32)

    with torch.no_grad():
        Pred_img = model(img)
        Pred_vec = encoder(img)

    Pred_imgs.extend(Pred_img.detach().cpu().numpy())
    Pred_vecs.extend(Pred_vec.detach().cpu().numpy())
    True_imgs.extend(img.detach().cpu().numpy())
    True_labels.extend(label.detach().numpy())

#머신러닝의 OC-SVM을 학습시키기 위했선 Training 데이터도 필요함 
Train_vecs = [] 
Train_labels = [] 

for img,label in train_loader:
    img,label = img.to(cfg['device']).type(torch.float32),label.to(cfg['device']).type(torch.float32)

    with torch.no_grad():
        Train_vec = encoder(img)
    Train_vecs.extend(Train_vec.detach().cpu().numpy())
    Train_labels.extend(label.detach().cpu().numpy())

Train_vecs = np.array(Train_vecs)
Train_labels = np.array(Train_labels)
```

**테스트 데이터 평가 : Reconstruction**
```python 
#Reconstruction 방법론을 이용한 경우 각 이미지의 Anomaly Score 계산 -> MSE를 사용 함 
from sklearn.metrics import roc_curve,auc 
test_score = np.mean(np.array(Pred_imgs-True_imgs).reshape(len(True_labels),-1)**2,axis=1)
fpr,tpr,threshold = roc_curve(True_labels,test_score,pos_label=0)
AUROC = round(auc(fpr,tpr),4)
```

**테스트 데이터 평가 : Machine learning**
```python
#Machine learning : OC-SVM 방법론을 이용한 경우 Train - normal 데이터로 학습한 뒤 Test 데이터 Inference 
#우선 normal 데이터의 스케일링 진행 
from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()
train_normalized_vecs = minmax.fit_transform(Train_vecs)
test_normalized_vecs = minmax.transform(Pred_vecs)

#Test 라벨 정제 
'''
- Test 라벨의 경우 Anomaly가 세부 카테고리로 나뉘어 있음 
- 이를 Anomaly  하나로 통합해야 함 
- 기존의 Normal의 라벨은 2 이므로 나머지는 모두 1로 바꾼 뒤 2를 -1로 변환 함 
- True_labels == Test 데이터의 진짜 라벨 
'''
True_labels = np.where(True_labels==2,True_labels,1)  
True_labels = np.where(True_labels==1,True_labels,-1) # Anomaly:-1, normal:1

#모델 학습 
from sklearn.svm import OneClassSVM
model = OneClassSVM()
model.fit(train_normalized_vecs)
Pred_labels = model.predict(test_normalized_vecs)
preds = model.score_samples(test_normalized_vecs)

#AUROC 계산 
fpr,tpr,threshold = roc_curve(preds,True_labels,pos_label=1)
AUROC = round(auc(fpr, tpr),4)
```

$\space$

# 3. 결과 
## 3.1 Preprocess : Augmentation을 모델 학습 전에 적용 시킨 경우
**AUROC 결과**  
<p align='center'><img src = 'https://user-images.githubusercontent.com/92499881/202158611-03d9286a-b914-46d1-89b0-65968eb6590e.png' width='90%',height='100%'>

**ROC-Curve**
<center>Dataset : Hazlenut</center>
<figure class='half'>
   <p align='center'><img src = 'https://user-images.githubusercontent.com/92499881/201528003-c1b27965-ca5a-49f4-8567-0194c2ab8839.png' width='35%',height='30%'>
   <img src = 'https://user-images.githubusercontent.com/92499881/201529458-e01bcb86-7460-4301-b1c4-68af7add49e1.png' width='35%',height='30%'>
</figure>
<center>Dataset : Carpet</center>
<figure class='half'>
   <p align='center'><img src = 'https://user-images.githubusercontent.com/92499881/202158919-fef19595-d6d5-464b-8200-60ed8691979d.png' width='35%',height='30%'>
   <img src = 'https://user-images.githubusercontent.com/92499881/202158928-49d51b02-e958-4629-90a0-747ebffa10fc.png' width='35%',height='30%'>
</figure>

```
결과 분석
- Augmentation을 적용하는 경우 Method 상관 없이 대체로 성능이 좋아지는 것을 볼 수 있다. 
- 대체로 Reconstruction 보다는 OC-SVM이 더 나은 성능을 보여주고 있는데, Image level에서 Anoamly Detection을 수행하는 것 보다는 Feature   
  space에서 Anoamly Detection을 수행하는 것이 더 좋다라고 생각할 수 있을 것 같다. 
- Carpet 데이터셋의 Flip과 Crop의 경우 Augmentation을 아무것도 적용하지 않은 Base와 꽤 큰 성능 차이를 보여주고 있다. 
- 이는 Carpet 데이터가 비슷한 패턴을 갖고 있기 때문에 형태 변화를 통해 모델이 다양한 Feature를 학습할 수 있었기 때문이라고 생각한다. 
```


## 3.2 Postprocess : 모델 학습 후 Test 데이터에 Augmentation을 적용하는 경우 
- 아무런 Augmentation을 적용하지 않은 Baseline AutoEncoder에 Test 데이터 Inference 시 Augmentation 적용
- Augmentation 세부 세팅값은 Reconstruction, OC-SVM 모두 동일 

**AUROC**

<p align='center'><img src = 'https://user-images.githubusercontent.com/92499881/202161851-8c0d00e4-e560-4dfa-a023-abd7fb553e98.png' width='90%',height='100%'>


**ROC-Curve**
<center>Dataset : Hazlenut</center>
<figure class='half'>
   <p align='center'><img src = 'https://user-images.githubusercontent.com/92499881/201528506-f8c65f5a-0e67-4b4f-bf91-95ed7b3187e2.png' width='35%',height='30%'>
   <img src = 'https://user-images.githubusercontent.com/92499881/201528520-f4f48f60-b6f5-4b40-a01e-eb91da1f4a78.png' width='35%',height='30%'>
</figure>
<center>Dataset : Carpet</center>
<figure class='half'>
   <p align='center'><img src = 'https://user-images.githubusercontent.com/92499881/202162084-8415fb04-2425-468d-ada3-5bdf21905efe.png' width='35%',height='30%'>
   <img src = 'https://user-images.githubusercontent.com/92499881/202162098-c44e720b-e4b7-4b20-8c5d-b4688dea55e6.png' width='35%',height='30%'>
</figure>


```
결과 분석
- Augmentation의 종류에 따라 편차는 존재하지만 대체적으로는 Augmentation을 사용하지 않은 경우 보다 성능이 증가한 것으로 보인다. 
- 다만 Hazelnut 데이터의 경우 Crop이 가장 좋은 성능을 보이고 있는데 이는 운이지 않을까 하는 생각이 든다. 
- Crop은 랜덤하게 이미지의 일부분을 지정한 사이즈에 맞게 자르는 방식으로, 자르는 과정에서 Anomaly area가 잘려 원본 이미지 보다 Anomaly 부분이 
  크게 포착되어 성능이 향상된 것으로 생각된다. 
- 반대로 생각하면 Crop하는 과정에서 Anomaly 파트는 사라질 수 있기 때문에 이는 좋은 방법이라고 생각되지 않는다. 
```

## 3.3 Mixed : Preprocess + Postprocess 
- Preprocess 와 Postprocess에서 가장 성능이 좋은 Augmentation 종류 각각 뽑아 같이 사용하는 경우 성능 측정 
- 각 데이터 및 방법론에 적용한 Augmentation은 아래와 같음 
<p align='center'><img src='https://user-images.githubusercontent.com/92499881/202165428-6126a76b-b792-4540-8679-53afedce8336.png'>

**AUROC**
<p align='center'><img src = 'https://user-images.githubusercontent.com/92499881/202163709-517dbae5-fec4-4860-9ac8-0b79e52f241b.png' width='90%',height='100%'>   


**ROC-Curve**

<center>Dataset : Hazlenut</center>
<figure class='half'>
   <p align='center'><img src = 'https://user-images.githubusercontent.com/92499881/201528953-7ea03864-25c7-4b1c-958c-2a7e70978fcd.png' width='40%',height='30%'>
   <img src = 'https://user-images.githubusercontent.com/92499881/202165816-7267ebe7-907d-4490-9e7a-88e5022b9ba1.png' width='40%',height='30%'>
</figure>
<center>Dataset : Carpet</center>
<figure class='half'>
   <p align='center'><img src = 'https://user-images.githubusercontent.com/92499881/202165951-c93cae99-e847-40cc-8266-88657c387af5.png' width='40%',height='30%'>
   <img src = 'https://user-images.githubusercontent.com/92499881/202166195-fd6ce4b7-bf1f-44f5-919f-5387ebdc1a48.png' width='40%',height='30%'>
</figure>



```
결과 분석
- 꽤나 큰 성능 향상 폭을 보이는데 Reconstruction, OC-SVM 모두 Preprocess만 적용한 경우, Post process만 적용한 경우 보다도 두개 모두 적용한 경우 더 나은 성능을 보인다. 
- 특히 OC-SVM의 경우 65%향상이라는 놀라운 결과를 보여준다. 
```
**전체 결과**
 <p align='center'><img src = 'https://user-images.githubusercontent.com/92499881/202166647-fa1b09d0-bbd6-48a5-b713-8c560b43d089.png'>

# 4. 결론 
- 본 실험을 통해 Augmentation이 Anomaly Detection에 어떤 영향을 끼치는 지 확인을 해 보았다. 적용한 Augmentation 종류에 따라 성능 차이는 컸으며 대체로 Preprocess 와 Postprocess 모두 사용하는 경우 성능이 더 좋아지는 결과를 보였다. 
- 다만 이 실험을 통해 무조건 Augmentation이 Anomaly Detection에 항상 좋다고 말할 수 없다. 이 실험에서는 해당 데이터 셋이 운이 좋게 Augmentation을 적용함에 따라 normal 과 anomal 간의 distinct boundary가 멀어짐에 따라 성능이 좋아진다고 생각할 수 있으며, 다른 데이터셋이나 Augmentation에 따른 영향은 추가적인 실험이 필요하다. 
- 또한 Augmentation을 적용하는 경우 단일 방법으로 사용하는 것이 다소 아쉬운데, 기회가 된다면 여러 Augmentation 조합을 구성하여 한다면 더 좋은 결과를 얻을 수 있을 것이라 생각된다. 
- 또한 Decision boundary나 데이터 instance에 대한 시각화를 통해 Preprocess 또는 Postprocess에 따라 어떻게 변화 혹은 이동하였는지 확인한다면 더 명확한 인사이트를 얻을 수 있을 것이라 생각된다. 

  
