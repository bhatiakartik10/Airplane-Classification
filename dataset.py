import os
import pdb
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

DATAPATH = 'C:\\Users\Kartik\Downloads\Retail Pulse ML Assignment Data'

def get_transform(resize, phase='train'):
    '''
    transform the image to tuple (h,w) from resize
    Randomflips are not used in validation
    '''
    if(phase == 'train'):
        return transforms.Compose([
            transforms.Resize((72,72)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((72,72)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229, 0.224, 0.225])
        ])

class AirplaneDataset(Dataset):
    '''
    Dataset class for the Airplane dataset
    Mapping is built for variants -> families
    '''
    def __init__(self, phase = 'train', resize = (72,72)):
        assert phase in ['train','val']
        self.phase = phase
        self.resize = resize
        
        image_variant_dict = {}
        with open(os.path.join(DATAPATH,'images_variant_train.txt'),'r') as f:
            for line in f.readlines():
                line = line.strip().split(' ',1)
                image_variant_dict[line[0]] = line[1]
        
        variant_family_dict = {}
        with open(os.path.join(DATAPATH,'images_family_train.txt'),'r') as f:
            for line in f.readlines():
                line = line.strip().split(' ',1)
                variant_family_dict[image_variant_dict[line[0]]] = line[1]
        
        family_dict = {}
        with open(os.path.join(DATAPATH,'families.txt'),'r') as f:
            for idx,line in enumerate(f.readlines()):
                family_dict[line.strip()] = idx
                
        self.num_classes = len(family_dict)
        self.images = []
        self.labels = []
        
        with open(os.path.join(DATAPATH,'variants.txt'),'r') as f:
            for idx, line in enumerate(f.readlines()):
                variant = line.strip()
                idx = format(idx + 1,'04d')
                for image in os.listdir(os.path.join(DATAPATH,self.phase,idx)):
                    self.images.append(idx + '/' + image)
                    self.labels.append(family_dict[variant_family_dict[variant]])
        
        self.transform = get_transform(self.resize, self.phase)
    
    def __getitem__(self, item):
        image = Image.open(os.path.join(DATAPATH,self.phase,self.images[item]))
        image = self.transform(image)
        
        return image,self.labels[item]
    
    def __len__(self):
        return len(self.images)

        