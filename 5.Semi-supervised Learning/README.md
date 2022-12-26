# Semi Supervised Learning 

# Table of Contents 
- [Semi Supervised Learning](#semi-supervised-learning)
- [Table of Contents](#table-of-contents)
- [이론](#이론)
  - [개요](#개요)
  - [Semi Supervised Learning의 가정](#semi-supervised-learning의-가정)
  - [Semi Supervised Learning 방법론 종류](#semi-supervised-learning-방법론-종류)
- [튜토리얼](#튜토리얼)
  - [0. $\\Pi-model$ 이란?](#0-pi-model-이란)
  - [1. 데이터 로드](#1-데이터-로드)
    - [1.1. 데이터셋 로드](#11-데이터셋-로드)
    - [1.2. label - unlabel split](#12-label---unlabel-split)
    - [1.3. 학습을 위한 커스텀 데이터셋](#13-학습을-위한-커스텀-데이터셋)
  - [2. 모델](#2-모델)
    - [2.1 Pi model](#21-pi-model)
    - [2.2 사전학습 모델](#22-사전학습-모델)
  - [3. Loss function](#3-loss-function)
  - [4. 학습](#4-학습)
  - [5. 실험 결과](#5-실험-결과)

# 이론 

## 개요
- 학습 방법을 데이터 라벨링 관점에서 분류를 하면 크게 3가지로 나눌 수 있다. 
  - 지도학습(Supervised Learning)
  - 준지도학습(Semi Supvervised Learning)
  - 비지도학습(Unsupervised Learning)

$\space$
- 대표적인 방법인 지도학습은 우리가 쉽게 이해할 수 있듯이 데이터와 라벨이 모두 있는 경우를 말한다. 데이터와 라벨을 충분히 가지고 있는 경우 준수한 성능을 보여주기 때문이다. 하지만 항상 데이터와 라벨이 모두 있는 경우는 없으며 라벨링을 확보하기 어려운 분야들도 존재한다. 라벨링에 전문성이 필요하거나 많은 데이터를 확보해야 하기 때문에 그 비용이 비싸기 때문이다. 
  
- 따라서 적은 labelec data가 있으면서 추가로 활용할 수 있는 대용량의 unlabeled data가 있다면 semi-supervised learning을 고려할 수 있다. semi-supervised learning이란 소량의 labeled data에는 supervised learning을 적용하고 대용량 unlabeled data에는 unsupervised learning을 적용하여 추가적인 성능 향상을 목표로 하는 방법론이다.

- 이런 방법론에는 label을 맞추는 모델에서 벗어나 데이터 자체의 본질적인 특성이 모델링 된다면 소량의 labeled data를 통한 약간의 가이드로 일반화 성능을 끌어올릴 수 있다는 것을 가정으로 학습이 이루어진다. 

<p align='center'><img src = 'https://user-images.githubusercontent.com/92499881/209529611-e6c03b68-3f37-4dfb-a882-27dccd3fda17.png'>

- Semi Supervised Learning은 label data와 unlabeld data를 모두 활용하므로 Loss term 역시 Supervised Loss 와 Unsupervised Loss 두개의 term으로 구성되며 두개의 합을 최소화 하는 방향으로 학습이 이루어 진다. 
  
<center> Total Loss = Supervised Loss + Unsupervised Loss </center>



## Semi Supervised Learning의 가정 
- Semi Supervised Learning을 위한 기본적인 가정들이 존재한다. 
  - The smoothness assumption 
    - 만약 가까운 두 representation이 있을 경우 해당하는 출력들도 같아야 한다. 
    - 이는 같은 class에 속하고 같은 cluster인 두 입력이 입력공간 상에서 고밀도 지역에 위치하고 있다면, 해당하는 출력도 가까워야한다는 것을 의미한다. 반대도 역시 성립하는데, 만약 두 데이터포인트가 저밀도지역에서 멀리 위치한다면, 해당하는 출력도 역시 멀어야 한다. 이러한 가정은 classification엔 도움이 되는 가정이지만 regression에선 별로 도움이 안된다.
  - The cluster assumption
    - 만약 데이터 포인트들이 같은 클러스터에 있다면 같은 클래스일 것이다. 
    - 이 가정이 성립한다면 하나의 cluster는 하나의 class를 나타낼 것이고 이를 통해 decision boundary는 저밀도지역을 통과해야만 한다라고 말할 수 있다. 
  - The manifold assumption 
    - 고차원의 데이터를 저차원의 manifold로 보낼 수 있다. 
    - 고차원의 공간상에서 generative task를 위한 진짜 data distribution은 추정하기 어렵다. 또한 discriminative task에서도 고차원에서는 class간의 distance가 거의 유사하기 때문에 classification이 어렵다. 그러나 만약 data를 더 낮은 차원으로 보낼 수 있다면 우리는 unlabeled data를 사용해서 저차원 표현을 얻을 수 있고 labeled data를 사용해 더 간단한 task를 풀 수 있다

## Semi Supervised Learning 방법론 종류 
- Semi Supervised Learning은 Unlabeled data를 어떻게 다루냐에 따라 분류가 가능하며, 이번 튜토리얼에서 집중할 방법론은 **Consistency regularization** 이다. 해당 방법론은 Smoothness assumption에 기반하여 작은 perturbation을 가하더라도 예측의 결과에는 일관성이 있을 것이라는 가정 하에 출발한다. unlabeled data는 예측 결과를 알 수 없기 때문에 data augmentation을 통해 class가 바뀌지 않을 정도로 변화를 주었을 때 원 데이터의 예측 결과와 같아지도록 Unsupervised loss를 학습하게 된다. 

- 가장 유명한 모델은 $\Pi-model$, Temporal Ensembling, Mean Teacher 등이 있다. 


# 튜토리얼 
- 튜토리얼에서는 Consistency 기반의 Semi supervised learning, 그리고 그 중에서 Target Quality 와 관련된 방법인 Pi 모델을 구현하고자 한다. 

## 0. $\Pi-model$ 이란? 
- $\Pi-model$ 이란 Temporal Ensembling for semi-supervised learning 논문에서 제안 된 방법론으로 Temporal Ensembling 설명을 위해 사전에 개발된 모델이다. 굉장히 단순한 구조이지만 향후 나타나는 다양한 방법론들의 기본 구조가 되는 모델이며 간단하게 구현이 가능하다. 
- 단 1개의 Encoder를 사용하여 Supervised Loss를 구하고 Augmentation을 통해 Unsupervised Loss로 Regularization을 주게 된다. 

<p align='center'><img src = 'https://user-images.githubusercontent.com/92499881/209542860-23d8989c-3bd7-4840-9e0c-7a8b2d40c6b8.png'>


## 1. 데이터 로드 
- 학습 데이터로는 Cifar10과 Cifar100을 사용한다. 
- 해당 데이터셋은 32x32의 크기를 갖는 이미지 60000만장의 데이터셋으로 Cifar10은 class label 수가 10개, Cifar100은 class label 수가 100개를 의미한다. 
- 기본적인 Augmentation은 `ToTensor` 만을 사용하며 학습 과정 중에 Sthocastic data transformation이 가해지게 된다. 

### 1.1. 데이터셋 로드 
   - Pickle 형태로 저장되어 있는 Cifar10 데이터셋은 ndarray 타입으로 로드 함 
```python
def from_pickle_to_img(file,name):
    with open(file, 'rb') as f:
        data = pickle.load(f,encoding='bytes')
    if name == 'cifar10':
        #해당 데이터의 경우 flatten한 상태로 저장이 되어 있기 때문에 이를 image에 맞춰서 변형해줌 
        batch_imgs = data[b'data'].reshape(-1,3,32,32).transpose(0,2,3,1) 
        batch_labels = data[b'labels']
        return batch_imgs, np.array(batch_labels) 
    
    elif name == 'cifar100':
        batch_imgs = data[b'data'].reshape(-1,32,32,3)
        batch_labels = data[b'fine_labels']
        return batch_imgs, np.array(batch_labels) 

def load_cifar10():
    files = sorted(glob('./Dataset/cifar-10-batches-py/*')[1:-1])
    imgs = [] 
    labels = [] 
    for file in files:
        batch_imgs,batch_labels = from_pickle_to_img(file,'cifar10')
        imgs.extend(batch_imgs)
        labels.extend(batch_labels)
    labels = np.array(labels)
    imgs = np.array(imgs)
    return (imgs[:50000],labels[:50000]),(imgs[50000:],labels[50000:])

def load_cifar100():
    files = sorted(glob('./Dataset/cifar-100-python/*'))[-2:]
    train_imgs,train_labels = from_pickle_to_img(files[1],'cifar100')
    test_imgs,test_labels = from_pickle_to_img(files[0],'cifar100')
    return (train_imgs,train_labels),(test_imgs,test_labels)
```

### 1.2. label - unlabel split 
- 해당 데이터셋은 모두 라벨이 있는 Supervised learning 데이터셋 
- 따라서 원하는 비율 만큼 일부 데이터셋은 label이 없는 Unlabeled dataset으로 바꿈 
```python 
#데이터셋 로드 후 label - unlabel 데이터 만드는 메소드 
def label_unlabel_load(cfg):
    (train_imgs,train_labels),(test_imgs,test_labels) = dataset_load(cfg['dataset'])
    labels = np.unique(train_labels)
    label = labels[0]
    for label in labels:
        label_idx = (train_labels ==label).nonzero()[0]
        unlabel_idx = np.random.choice(label_idx,int(len(label_idx)*cfg['unlabel_ratio']),replace=False)
        train_labels[unlabel_idx] = -1 
        
        
    train = {'imgs':train_imgs,
            'labels':train_labels}
    test = {'imgs':test_imgs,
            'labels':test_labels}
    return train, test     
```

### 1.3. 학습을 위한 커스텀 데이터셋 
- 앞서 로드 한 imgs와 label을 학습 과정 중 배치 단위로 쓸 수 있도록 커스텀 데이터 셋 생성 
- 기본적인 Augmentation 은 `ToTensor` 만을 사용하며 학습 과정 중 Augmentation을 추가로 적용 함 

```python 
class CifarDataset(Dataset):
    def __init__(self,data,unlabel=False,transform=None):
        super(CifarDataset,self).__init__()
        self.transform = transform 
        self.imgs = data['imgs']
        self.labels = data['labels']
        self.unlabel = unlabel 
        self.transform = self.transfrom_init(transform)
        
    def __len__(self):
        return len(self.imgs)
    
    def transfrom_init(self,transform):
        if transform == None:
            return transforms.Compose([transforms.ToTensor()])
        else:
            return transform 

            
    def __getitem__(self,idx):
        if self.unlabel:
            img = self.transform(self.imgs[idx])
            return img
        else:
            img = self.transform(self.imgs[idx])
            label = self.labels[idx]
            return img,label 
```

## 2. 모델 
- Pi 모델의 경우 사전 학습 된 Encoder를 사용하는 것이 아닌 Vanilla CNN으루 구축 한 구조를 사용 함 
- 하지만 이번 튜토리얼에서는 Pretrained Network를 사용한 경우도 같이 비교 실험할 예정 

### 2.1 Pi model 
- Pi 모델은 CNN과 Maxpooling, Linear로만 구성되어 있으며 Input 데이터에 Gaussian noise를 추가하는 것이 특징이다. 

```python 
# Input data의 Gaussian noise를 추가해주는 레이어 
class GaussianNoise(nn.Module):
    def __init__(self, batch_size, input_shape=(1, 32, 32), std=0.05,device='cpu'):
        super(GaussianNoise, self).__init__()
        self.shape = (batch_size,) + input_shape
        self.noise = Variable(torch.zeros(self.shape)).to(device)
        self.std = std
        
        
    def forward(self, x):
        self.noise.data.normal_(0, std=self.std)
        return x + self.noise

#Input에 바로 가우시안 노이즈를 가하며 그 후 3개의 Convolution block을 지난 뒤 average pooling 그리고 linear를 통해 class 수에 맞게 projection 됨 
#가우시안 노이즈는 학습 과정 중에만 사용 됨 
class PiModel(nn.Module):
    def __init__(self,num_labels=10,batch_size=100,std=0.15,device='cpu'):
        super(PiModel,self).__init__()
        self.noise = GaussianNoise(batch_size,std=std,device=device)
        self.conv1 = self.conv_block(3,128).to(device)
        self.conv2 = self.conv_block(128,256).to(device)
        self.conv3 = self.conv3_block().to(device)
        self.linear = nn.Linear(128,num_labels).to(device)
        
        
    def conv_block(self,input_channel,num_filters):
        return nn.Sequential(
                                nn.Conv2d(input_channel,num_filters,3,1,1),
                                nn.LeakyReLU(0.1),
                                nn.Conv2d(num_filters,num_filters,3,1,1),
                                nn.LeakyReLU(0.1),
                                nn.Conv2d(num_filters,num_filters,3,1,1),
                                nn.LeakyReLU(0.1),
                                nn.MaxPool2d(2,2),
                                nn.Dropout(0.5)                    
                             )
    def conv3_block (self):
        return nn.Sequential(
                              nn.Conv2d(256,512,3,1,0),
                              nn.LeakyReLU(0.1),
                              nn.Conv2d(512,256,1,1),
                              nn.LeakyReLU(0.1),
                              nn.Conv2d(256,128,1,1),
                              nn.LeakyReLU(0.1)

        )
    def forward(self,x,train):
        if train:
            x = self.noise(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = F.avg_pool2d(x, x.size()[2:]).squeeze()
        x = self.linear(x)
        return x 
```

### 2.2 사전학습 모델 
- 사전학습을 Encoder로 사용하는 경우 역시 실험을 진행함 
- 사용 된 모델은 Resnet18 과 ssl_resnet50 

```python
class Model(nn.Module):
    def __init__(self,model_name='resnet18',dataset_name='cifar10'):
        super(Model,self).__init__()
        self.model_name = model_name 
        self.encoder = self.pretrained_encoder(model_name)
        self.linear = self.output_layer(dataset_name)
        
    
    def pretrained_encoder(self,model_name):
        res = timm.create_model(model_name,pretrained=True)
        encoder = nn.Sequential(*(list(res.children())[:-1]))
        return encoder 
    
    def output_layer(self,dataset_name):
        in_features = list(self.encoder[-2][-1].children())[-3].out_channels
        if dataset_name == 'cifar10':
            return nn.Linear(in_features = in_features,out_features= 10)
        else:
            return nn.Linear(in_features = in_features,out_features= 100)
        
    def forward(self,x,_):
        x = self.encoder(x)
        x = self.linear(x)
        return x 
```        
## 3. Loss function 
- Loss function은 크게 Supervised loss와 Unsupervised loss로 구성 된다. 
- Supervised Loss는 기존 Supervised learning과 동일하게 사용 되며 Unsupervised Loss는 서로 다른 Augmentation을 적용한 두 representation 간의 distance metric을 이용하여 계산하게 된다. 
- 두 개의 Loss term을 혼합하는 것 또한 중요한데 이 경우 Ramp up function을 사용한다. 
- Ramp up function이란 Epoch에 따라 Unsupervised Loss의 weight를 달리 하여 학습 정도를 조절하는 함수로 Gaussian distribution 형태를 갖고 있다. 

```python 
class PiCriterion:
    def __init__(self,cfg):
        self.label_criterion = nn.CrossEntropyLoss() #label 데이터 Loss function 
        self.unlabel_criterion = nn.MSELoss() #unlabel 데이터 loss function 
        self.n_labeled = 50000 * (1-cfg['unlabel_ratio'])
        self.superonly = cfg['super_only']

    #Epoch에 따라 Unsupervised Loss의 weight를 반환     
    def ramp_up(self,epoch, max_epochs=80, max_val=30, mult=-5):
        if epoch == 0:
            return 0.
        elif epoch >= max_epochs:
            return max_val
        return max_val * np.exp(mult * (1. - float(epoch) / max_epochs) ** 2)

    def weight_schedule(self,epoch, max_epochs=80, max_val=30,
                        mult=-5, n_samples=50000):
        
            max_val = max_val * (float(self.n_labeled) / n_samples)
            return self.ramp_up(epoch, max_epochs, max_val, mult)
    #모델의 evaluate 결과인 두 개의 y_pred와 label을 받음
    #batch_label은 label 데이터에 대해서만 loss를 계산 함 
    def __call__(self,y_pred_1,y_pred_2,batch_labels,epoch):
        label_idx = (batch_labels!=-1).nonzero().flatten()
    #supervised loss 
        tl_loss = self.label_criterion(y_pred_1[label_idx],
                                       batch_labels[label_idx])
    #unsupervised loss 
        weight = self.weight_schedule(epoch)
        weight = torch.autograd.Variable(torch.FloatTensor([weight]).cuda(), requires_grad=False)
        tu_loss = self.unlabel_criterion(y_pred_1,y_pred_2) * weight 
        
        if self.superonly:
            return tl_loss,torch.tensor([0]),torch.tensor([0]),torch.tensor([0])
        else:
            #total loss 
            loss = tl_loss + tu_loss  
            
            return loss, tl_loss, tu_loss, weight 
```            

## 4. 학습
- 학습은 일반적인 학습과 마찬가지로 진행되며 차이 점으로는 동일한 데이터 인스턴스에 대해 stochastic augmentation을 두번 적용하여 두개의 input을 만듬
- 이는 Unsupervised Loss를 위한 과정으로 두 개의 representation이 가까워 지는 방향으로 학습이 이뤄지게 됨 

```python 
#Stochastic Augmentation을 만들기 위한 메소드 
#colorjitter을 stochastic 하게 만들어 주기 위해 RandomApply 사용 
def make_transform():
    color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
    transformer = transforms.Compose([
                                          transforms.RandomApply([color_jitter],p=0.8),
                                          transforms.RandomResizedCrop(32),
                                          transforms.GaussianBlur(kernel_size=int(0.1*32))
                                         ])
    return transformer
#학습 과정 중 확인하기 위한 valid set 
# valid set은 Training dataset 중 일부를 사용 함 
def make_valid(dataset = 'cifar10'):
    (train_imgs,train_labels),(test_imgs,test_labels) = dataset_load(dataset)
    idx = np.random.choice(np.arange(len(train_imgs)),5000,replace=False)
    valid_set = {'imgs':train_imgs[idx],
        'labels':train_labels[idx]}
    return valid_set 

def train(model,criterion,optimizer,train_loader,cfg,transformer):
    global epoch 
    model.train() 
    tl_loss = [] 
    tu_loss = [] 
    total_loss = [] 
    for batch_img,batch_labels in tqdm(train_loader):
        #하나의 batch_img에 대해 두 번의 augmentation을 적용 함 
        #동일한 augmentation을 적용하지만 stochastic하기 때문에 실질적으로는 다르게 적용 됨 
        batch_img_1 = transformer(batch_img.type(torch.float32).to(cfg['device']))
        batch_img_2 = transformer(batch_img.type(torch.float32).to(cfg['device']))
        batch_labels = batch_labels.to(cfg['device'])
        
        y_pred_1 = model(batch_img_1,True)
        y_pred_2 = model(batch_img_2,True)
        loss,tl,tu,weight = criterion(y_pred_1,y_pred_2,batch_labels,epoch)
        
        total_loss.append(loss.detach().cpu().numpy())
        tl_loss.append(tl.detach().cpu().numpy())
        tu_loss.append(tu.detach().cpu().numpy())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return np.mean(total_loss),np.mean(tl_loss),np.mean(tu_loss),weight

def valid(model,valid_loader,cfg):
    labels = []
    y_preds = [] 
    model.eval() 
    for batch_imgs,batch_labels in valid_loader:
        batch_imgs = batch_imgs.type(torch.float32).to(cfg['device'])
        with torch.no_grad():
            y_pred = model(batch_imgs,False)
        y_pred = torch.argmax(F.softmax(y_pred),dim=1)
        y_pred = y_pred.detach().cpu().numpy()
        
        y_preds.extend(y_pred)
        labels.extend(batch_labels.detach().cpu().numpy())    
    f1 = f1_score(np.array(y_preds),np.array(labels),average='macro')
    auc = accuracy_score(np.array(y_preds),np.array(labels))
    return f1, auc
```    

## 5. 실험 결과 