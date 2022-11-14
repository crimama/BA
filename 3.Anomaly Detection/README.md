# Anomaly Detection 
- Part 1에서는 Anomaly Detection과 관련된 이론에 대해서 다룹니다. 
- Part 2에서는 Anomaly Detection과 Augmentation에 관련된 실험을 진행합니다.

# Table of Contents 
- [Anomaly Detection](#anomaly-detection)
- [Table of Contents](#table-of-contents)
- [1. Anomaly Detection 이론](#1-anomaly-detection-이론)
- [2. 실험 : Augmentation에 따른 Anomaly Detection 성능 비교](#2-실험--augmentation에-따른-anomaly-detection-성능-비교)
  - [2.0. 실험 세팅](#20-실험-세팅)
  - [2.1. 베이스라인](#21-베이스라인)
- [3. 결과](#3-결과)
  - [3.1 Preprocess : Augmentation에 따른 성능 비교](#31-preprocess--augmentation에-따른-성능-비교)
  - [3.2 Postprocess : Augmentation에 따른 성능 비교](#32-postprocess--augmentation에-따른-성능-비교)
  - [3.3 Mixed : Preprocess + Postprocess](#33-mixed--preprocess--postprocess)
- [4. 결론](#4-결론)

# 1. Anomaly Detection 이론 
- Anomaly Detection 이란 정상 데이터 속 이상, novel 데이터를 탐지하는 방법론 


# 2. 실험 : Augmentation에 따른 Anomaly Detection 성능 비교  
- Augmentation이란 한정된 데이터의 양을 늘리기 위해 원본에 각종 변환을 적용하여 데이터를 증강시키는 기법입니다. 일반적으로 학습 전 이미지에 변형을 가하는 전처리 형식으로 사용 됩니다. 
- 이 Augmentation이 Anomaly Detection에 적용되었을 때 성능이 어떻게 변화하는지 확인하고자 합니다. 
- 사용된 베이스라인은 AutoEncoder와 OC-SVM으로, OC-SVM은 학습된 AutoEncoder의 인코더로 추출된 임베딩 벡터를 인풋으로 사용합니다. 
- 실험을 통해 확인하고자 하는 바는 아래와 같습니다. 
```
    1. Augmentation 적용에 따른 Anomaly Detection 성능 변화 
    2. Augmentation 종류에 따른 Anomaly Detection 성능 차이 
    3. Postprocess로 Test 데이터 inference 시 Augmentation 적용에 따른 성능 변화 
```
$\space$

## 2.0. 실험 세팅 
**Train** 
- 기본 Baseline model을 Convolution Auto Encoder를 사용하며 loss function으로는 reconstruction error : l2 norm을 사용합니다. 사용하는 데이터셋은 MVtecAD의 hazelnut 데이터셋을 사용합니다. 
  
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
   - Reconstruction 방식은 Test data Inference 시 원본 이미지와 복원된 이미지 간의 l2 norm을 계산하여 이를 Anomaly Score로 사용합니다. 
2. Machine Learning 
   - Machine Learning 방식은 Reconstruction 방식과 마찬가지로 AutoEncoder를 학습시킨 뒤 Encoder를 이용해 각 Test 이미지의 Embedding Vector를 추출하여 이를 Input으로 사용하 Machine learning에 적용합니다. 
   - 이 경우 먼저 Train 데이터로 OC-SVM을 학습시킨 뒤 Test 데이터를 Inference합니다. 
- Metric 
  - 성능 비교를 위해 AUROC를 사용합니다. 
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
#preprocess : Datadir 클래스로 이미지 디렉토리를 읽어오고 Train-valid split한 뒤 Dataset, Dataloader를 생성 함 
 def preprocess(cfg,augmentation=None):
    #mk save dir 
    try:
        os.mkdir(f"./Save_models/{cfg['save_dir']}")
    except:
        pass
    torch.manual_seed(cfg['seed'])
    data_dir = cfg['Dataset_dir']
    Data_dir = Datadir_init()
    train_dirs = Data_dir.train_load()
    test_dirs,test_labels = Data_dir.test_load()
    indx = int(len(train_dirs)*0.8)

    train_dset = MVtecADDataset(cfg,train_dirs[:indx],Augmentation=augmentation)
    valid_dset = MVtecADDataset(cfg,train_dirs[indx:])
    test_dset = MVtecADDataset(cfg,test_dirs,test_labels)

    train_loader = DataLoader(train_dset,batch_size=cfg['batch_size'],shuffle=True)
    valid_loader = DataLoader(valid_dset,batch_size=cfg['batch_size'],shuffle=False)
    test_loader = DataLoader(test_dset,batch_size=cfg['batch_size'],shuffle=False)
    return train_loader,valid_loader,test_loader 
```

  
**Augmentation 구성**
- Augmentation은 아무것도 적용하지 않는 경우까지 포함해 총 7개 경우의 수가 있음 
- 각 Augmentation 구성은 1개의 Augmentation으로만 이루어져 있으며 Random Crop을 제외한 나머지는 모두 256으로 Resize되게 구성 됨  
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
- 
```python
class MVtecEncoder(nn.Module):
    def __init__(self,encoded_space_dim):
        super(MVtecEncoder,self).__init__()

        self.encoder_cnn = nn.Sequential(
                                        nn.Conv2d(in_channels=3,out_channels=8,kernel_size=3,stride=2,padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(in_channels=8,out_channels=16,kernel_size=3,stride=2,padding=1),
                                        nn.BatchNorm2d(16),
                                        nn.ReLU(),
                                        nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=2,padding=1),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(16),
                                        nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=2,padding=1),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(16),
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

**학습**
```python 
def train_epoch(model,dataloader,criterion,optimizer,scheduler,scaler):
       model.train()
       optimizer.zero_grad()
       train_loss = [] 
       for img,label in dataloader:
              img = img.to(cfg['device']).type(torch.float32)
              with torch.cuda.amp.autocast():
                    y_pred = model(img).type(torch.float32)
                    loss = criterion(img,y_pred)
              #y_pred = model(img).type(torch.float32)
              

              #Backpropagation
              scaler.scale(loss).backward()
              scaler.step(optimizer)
              scaler.update() 
              #loss.backward()
              #optimizer.step()

              #loss save 
              train_loss.append(loss.detach().cpu().numpy())
       scheduler.step() 
       print(f'\t epoch : {epoch+1} train loss : {np.mean(train_loss):.3f}')
       return np.mean(train_loss)

```
**학습 과정 중 Validation**
- ipnyb 노트북에서 실행 시 실시간으로 복원되는 이미지를 확인할 수 있음 
```python
def valid_epoch(model,dataloader,criterion):
       model.eval()
       valid_loss = [] 
       with torch.no_grad():
              for img,label in dataloader:
                     img = img.to(cfg['device'])
                     y_pred = model(img)
                     loss = criterion(y_pred,img)
                     valid_loss.append(loss.detach().cpu().numpy())
       print(f'\t epoch : {epoch+1} valid loss : {np.mean(valid_loss):.3f}')
       fig, (ax1,ax2) = plt.subplots(ncols=2,nrows=1,figsize=(5, 5))
       ax1.imshow(img[0].detach().cpu().permute(1,2,0).numpy())
       ax2.imshow(y_pred[0].detach().cpu().permute(1,2,0).numpy())
       plt.show()
       return np.mean(valid_loss) 
```

**학습 완료 후 저장**
- Configuration, Model, metric이 저장 됨 
```python
def Save_result(cfg):

    f = open(f"./Save_models/{cfg['save_dir']}/config.yaml",'w+')
    yaml.dump(cfg, f, allow_unicode=True)
    
    metric =  {} 
    metric['auto'] = {} 
    metric['machine']={}

    machine = Machine_Metric(cfg)
    auto = Reconstruction_Metric(cfg)

    [auroc,roc], [acc,pre,rec,f1] = machine.main()
    [AUROC,ROC], [ACC,PRE,RECALL,F1] = auto.main() 

    metric['auto']['auroc'] = AUROC
    metric['auto']['roc'] =  ROC 
    metric['auto']['metric'] =[ACC,PRE,RECALL,F1]
    metric['machine']['roc'] = roc 
    metric['machine']['auroc']=auroc 
    metric['machine']['metric']=[acc,pre,rec,f1]


    with open(f"./Save_models/{cfg['save_dir']}/Metric.json",'w') as f:
       json.dump(metric,f)

    return metric 
```
**Metric**
- 학습은 오토인코더 한개만 진행 되지만 Metric은 두가지가 진행 됨 
- Reconstruction 방식과 Machine learning : OC-SVM 
- 동일한 학습된 오토인코더를 사용하되 OC-SVM은 인코더만 사용 함 
```python
class Reconstruction_Metric:
    def __init__(self,cfg,augmentation=None):
        self.cfg = cfg 
        self.train_loader,self.test_loader = self._preprocess(cfg,augmentation)
        self.model = torch.load(f"./Save_models/{cfg['save_dir']}/best.pt").to(cfg['device'])

    def _preprocess(self,cfg,augmentation):
        torch.manual_seed(cfg['seed'])
        data_dir = cfg['Dataset_dir']
        Data_dir = Datadir_init(data_dir)
        train_dirs = Data_dir.train_load()
        test_dirs,test_labels = Data_dir.test_load()

        train_dset = MVtecADDataset(cfg,train_dirs,Augmentation=None)
        test_dset = MVtecADDataset(cfg,test_dirs,test_labels,Augmentation=augmentation)

        train_loader = DataLoader(train_dset,batch_size=cfg['batch_size'],shuffle=True)
        test_loader = DataLoader(test_dset,batch_size=cfg['batch_size'],shuffle=False)
        return train_loader,test_loader 

    def _test_data_inference(self):
        Pred_imgs = [] 
        True_imgs = [] 
        True_labels = [] 
        self.model.eval()
        for img,label in self.test_loader:
            img = img.to(self.cfg['device']).type(torch.float32)

            with torch.no_grad():
                Pred_img = self.model(img)
            
            Pred_imgs.extend(Pred_img.detach().cpu().numpy())
            True_imgs.extend(img.detach().cpu().numpy())
            True_labels.extend(label.detach().numpy())

        Pred_imgs = np.array(Pred_imgs)
        True_imgs = np.array(True_imgs)
        True_labels = np.array(True_labels)
        True_labels = np.where(True_labels==2,True_labels,1)
        True_labels = np.where(True_labels==1,True_labels,0) # Anomaly:1, normal:0 `
        test_score = np.mean(np.array(Pred_imgs-True_imgs).reshape(len(True_labels),-1)**2,axis=1)
        return test_score, True_labels

    def _train_data_inference(self):
        Train_Pred_imgs = [] 
        Train_True_imgs = [] 
        for img,_ in self.train_loader:
            img = img.to(self.cfg['device']).type(torch.float32)
            
            with torch.no_grad():
                Pred_img =  self.model(img)
            Train_Pred_imgs.extend(Pred_img.detach().cpu().numpy())
            Train_True_imgs.extend(img.detach().cpu().numpy())

        Train_Pred_imgs = np.array(Train_Pred_imgs)
        Train_True_imgs = np.array(Train_True_imgs)

        Train_score = np.mean(np.array(Train_Pred_imgs-Train_True_imgs).reshape(len(Train_True_imgs),-1)**2,axis=1)
        return Train_score 

    def Metric_auroc_roc(self,Test_score,True_labels,plot=0):
        fpr,tpr,threshold = roc_curve(True_labels,Test_score,pos_label=0)
        AUROC = round(auc(fpr, tpr),4)

        if plot != 0:
            plt.plot(fpr,tpr)
            plt.title("ROC curve")
            plt.show()
        
        return AUROC,[fpr.tolist(),tpr.tolist(),threshold.tolist()]

    def Metric_Score(self,Train_score,Test_score,True_labels):
        Threshold = np.percentile(Train_score,80)
        Pred_labels = np.array(Test_score>Threshold).astype(int)
        ACC = accuracy_score(True_labels,Pred_labels)
        PRE = precision_score(True_labels,Pred_labels)
        RECALL = recall_score(True_labels,Pred_labels)
        F1 = f1_score(True_labels,Pred_labels)
        return ACC,PRE,RECALL,F1
        

    def main(self):
        Test_score, True_labels = self._test_data_inference()
        Train_score  =  self._train_data_inference()
        AUROC, ROC = self.Metric_auroc_roc(Test_score,True_labels)
        ACC,PRE,RECALL,F1 = self.Metric_Score(Train_score,Test_score,True_labels)

        return [AUROC,ROC], [ACC,PRE,RECALL,F1]


class Machine_Metric:
    def __init__(self,cfg,augmentation=None):
        self.cfg = cfg 
        self.train_loader,self.test_loader = self._preprocess(cfg,augmentation)
        self.model = torch.load(f"./Save_models/{cfg['save_dir']}/best.pt").to(cfg['device'])
        self.encoder = self.model.encoder 

    def _preprocess(self,cfg,augmentation):
        torch.manual_seed(cfg['seed'])
        data_dir = cfg['Dataset_dir']
        Data_dir = Datadir_init(data_dir)
        train_dirs = Data_dir.train_load()
        test_dirs,test_labels = Data_dir.test_load()

        train_dset = MVtecADDataset(cfg,train_dirs,Augmentation=None)
        test_dset = MVtecADDataset(cfg,test_dirs,test_labels,Augmentation=augmentation)

        train_loader = DataLoader(train_dset,batch_size=cfg['batch_size'],shuffle=True)
        test_loader = DataLoader(test_dset,batch_size=cfg['batch_size'],shuffle=False)
        return train_loader,test_loader 

    def _train_data_inference(self):
        normal_vecs = [] 
        normal_labels = [] 
        
        #Inference 
        for img,label in self.train_loader:
            img = img.to(self.cfg['device']).type(torch.float32)
            
            with torch.no_grad():
                normal_vec =  self.encoder(img)
                
            normal_vecs.extend(normal_vec.detach().cpu().numpy())
            normal_labels.extend(label.detach().cpu().numpy())
            
        normal_vecs = np.array(normal_vecs)
        normal_labels = np.array(normal_labels)

        return normal_vecs,normal_labels 

    def _test_data_inference(self):
        test_vecs = [] 
        test_labels = [] 
        for img,label in self.test_loader:
            img = img.to(self.cfg['device']).type(torch.float32)

            with torch.no_grad():
                test_vec = self.encoder(img)
            test_vecs.extend(test_vec.detach().cpu().numpy())
            test_labels.extend(label.detach().cpu().numpy())
        test_vecs = np.array(test_vecs)
        test_labels = np.array(test_labels)
        #처음 라벨 인코딩 할 때 2가 good 이었음, 2를 제외한 나머지 1:anomaly로 변경 
        test_labels = np.where(test_labels==2,test_labels,1)  
        test_labels = np.where(test_labels==1,test_labels,-1) # Anomaly:-1, normal:1

        return test_vecs,test_labels 

    def _Metric_auroc_roc(self,Test_score,True_labels,plot=0):
        fpr,tpr,threshold = roc_curve(True_labels,Test_score,pos_label=1)
        AUROC = round(auc(fpr, tpr),4)

        if plot != 0:
            plt.plot(fpr,tpr)
            plt.title("ROC curve")
            plt.show()
        
        return AUROC,[fpr.tolist(),tpr.tolist(),threshold.tolist()]

    def _Metric_score(self,True_labels,Pred_labels):
        ACC = accuracy_score(True_labels,Pred_labels)
        PRE = precision_score(True_labels,Pred_labels)
        RECALL = recall_score(True_labels,Pred_labels)
        F1 = f1_score(True_labels,Pred_labels)
        return ACC,PRE,RECALL,F1

    def scaling(self,normal_vecs):
        self.minmax = MinMaxScaler()
        normalized_vecs = self.minmax.fit_transform(normal_vecs)
        return normalized_vecs

    def main(self):
        normal_vecs,normal_labels = self._train_data_inference()
        test_vecs , test_labels = self._test_data_inference()
        normalized_vecs = self.scaling(normal_vecs)
        normalized_test_vecs = self.minmax.transform(test_vecs)
        self.test_labels = test_labels 


        self.model = OneClassSVM()
        self.model.fit(normalized_vecs)
        Pred_labels = self.model.predict(normalized_test_vecs)
        
        preds = self.model.score_samples(normalized_test_vecs)
        self.preds = preds 

        ACC,PRE,RECALL,F1 = self._Metric_score(test_labels,Pred_labels)
        AUROC,ROC = self._Metric_auroc_roc(preds,test_labels)
        return [AUROC,ROC], [ACC,PRE,RECALL,F1]
```
**모듈 합친 학습 코드**
```python 
args = parse_arguments()
#init 
cfg = yaml.load(open('./init_config.yaml','r'), Loader=yaml.FullLoader)
cfg['save_dir'] = args.save_dir
cfg['aug_number'] = int(args.aug_number)

trans = create_transformation(cfg)
wandb.init(project='BA_MVtec2',name=cfg['save_dir'])
wandb.config = cfg
train_loader,valid_loader,test_loader   = preprocess(cfg,trans)
#trainig intit 
model = Convolution_Auto_Encoder(MVtecEncoder,MVtecDecoder,cfg['encoded_space_dim']).to(cfg['device'])
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=100,eta_min=0)
scaler = torch.cuda.amp.GradScaler()

#Record 
total_train_loss = [] 
total_valid_loss = [] 
best_valid_loss = np.inf 
print('Training start')
for epoch in tqdm(range(cfg['Epochs'])):
#training  
    train_loss = train_epoch(model,train_loader,criterion,optimizer,scheduler,scaler)
    valid_loss = valid_epoch(model,valid_loader,criterion)
#logging 
    total_train_loss.append(train_loss)
    total_valid_loss.append(valid_loss)
    wandb.log({"train_loss":train_loss})
    wandb.log({"valid_loss":valid_loss})

#check point 
    if valid_loss < best_valid_loss:
        torch.save(model,f"./Save_models/{cfg['save_dir']}/best.pt")
        best_valid_loss = valid_loss 
        print(f'\t Model save : {epoch} | best loss : {best_valid_loss :.3f}')

#prevent explosion 
    if valid_loss != valid_loss:
        model = torch.load(f"./Save_models/{cfg['save_dir']}/best.pt")
        print('Model rewinded')

#Save config 
print('Training Done')

metric = Save_result(cfg)
print('Metric Done')
```

$\space$

# 3. 결과 
## 3.1 Preprocess : Augmentation에 따른 성능 비교 
<p align='center'><img src = 'https://user-images.githubusercontent.com/92499881/201528660-1308423c-d6f1-4416-a6d1-624004f48a54.png' width='70%',height='100%'>


<figure class='half'>
   <p align='center'><img src = 'https://user-images.githubusercontent.com/92499881/201528003-c1b27965-ca5a-49f4-8567-0194c2ab8839.png' width='49%',height='30%'>
   <img src = 'https://user-images.githubusercontent.com/92499881/201529458-e01bcb86-7460-4301-b1c4-68af7add49e1.png' width='49%',height='30%'>
</figure>

```
결과 분석
- Reconstruction과 OC-SVM 모두 Augmentation을 적용했을 때 대체로 성능이 좋아지는 것을 확인할 수 있었음 
- 대체로 Reconstruction 보다 OC-SVM이 성능ㅇ ㅣ더 좋음 
```


## 3.2 Postprocess : Augmentation에 따른 성능 비교 
- 아무런 Augmentation을 적용하지 않은 Baseline AutoEncoder에 Test 데이터 Inference 시 Augmentation 적용
- Augmentation 세부 세팅값은 Reconstruction, OC-SVM 모두 동일 

<p align='center'><img src = 'https://user-images.githubusercontent.com/92499881/201528478-212fd82a-0575-4bcf-9445-a1771573bf48.png' width='70%',height='100%'>


<figure class='half'>
   <p align='center'><img src = 'https://user-images.githubusercontent.com/92499881/201528506-f8c65f5a-0e67-4b4f-bf91-95ed7b3187e2.png' width='49%',height='30%'>
   <img src = 'https://user-images.githubusercontent.com/92499881/201528520-f4f48f60-b6f5-4b40-a01e-eb91da1f4a78.png' width='49%',height='30%'>
</figure>

```
결과 분석
- Reconstruction의 경우 Preprocess로 적용한 경우 보다 Postprocess로 적용한 경우 성능이 더 좋게 나타난다 
- 반대로 OC-SVM의 경우 성능 향상 폭이 Pre-process만큼 크지 않다 
```

## 3.3 Mixed : Preprocess + Postprocess 
- Preprocess 와 Postprocess에서 가장 성능이 좋은 Augmentation 종류 각각 뽑아 같이 사용하는 경우 성능 측정 
- Reconstruction 
  - Preprocess : Rotate 
  - Postprocess : Crop 
- OC-SVM 
  - Preprocess : Contrast 
  - Postprocess : Crop 
<p align='center'><img src = 'https://user-images.githubusercontent.com/92499881/201528830-a6dbe60a-ff2d-4cfb-9bc5-91f724d44de5.png' width='70%',height='100%'>   

<figure class='half'>
   <p align='center'><img src = 'https://user-images.githubusercontent.com/92499881/201528953-7ea03864-25c7-4b1c-958c-2a7e70978fcd.png' width='49%',height='30%'>
   <img src = 'https://user-images.githubusercontent.com/92499881/201529219-76e3385b-9384-401b-bad9-afc310cd7a5f.png' width='49%',height='30%'>
</figure>

```
결과 분석
- 꽤나 큰 성능 향상 폭을 보이는데 Reconstruction, OC-SVM 모두 Preprocess만 적용한 경우, Post process만 적용한 경우 보다도 두개 모두 적용한 경우 더 나은 성능을 보인다. 
- 특히 OC-SVM의 경우 65%향상이라는 놀라운 결과를 보여준다. 
```

  
# 4. 결론 
- 본 실험을 통해 Augmentation이 Anomaly Detection에 어떤 영향을 끼치는 지 확인을 해 보았다. 적용한 Augmentation 종류에 따라 성능 차이는 컸으며 