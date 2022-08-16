import os.path

from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets,transforms

data_transforms = {
    "train":transforms.Compose([
        transforms.Resize(256),
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]),
    "val":transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
}

data_dir = "F:\BaiduNetdiskDownload\hymenoptera_data"

image_datasets = {x:datasets.ImageFolder(os.path.join(data_dir,x),
                                         data_transforms[x])
                  for x in["train","val"]}
data_laoder = {x:DataLoader(image_datasets[x],batch_size=4,shuffle=True)
               for x in ["train","val"]}

datasize = {x:len(image_datasets[x]) for x in ["train","val"]}
class_names = image_datasets["train"].classes

print(class_names)
print(datasize)
for x,y in data_laoder["val"]:
    print(x.shape)
    print(y.shape)
    print(y)