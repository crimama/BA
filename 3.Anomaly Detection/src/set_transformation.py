from torchvision import transforms 
import torch.nn 

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



