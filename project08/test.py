import sklearn.metrics
import torch
from dataset import Mydataset
from torch.utils.data import DataLoader
from net_2 import Mynet
from torch import nn
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt


save_path = r'./param/net.pt'
test_dataset = Mydataset('F:\data',is_train=False)
test_loader = DataLoader(test_dataset,batch_size=36,shuffle=True)


net = Mynet().cuda()
net.load_state_dict(torch.load(save_path))
print('load success')
loss_fun = nn.NLLLoss()
# loss_fun = nn.CrossEntropyLoss()
sum_sorce = 0.
test_sum_loss = 0.
# f = nn.Softmax(dim=1)
for i,(data,labels) in enumerate(test_loader):
    net.eval()
    data,labels = data.cuda(),labels.cuda()
    labels = labels.long()
    test_out,mask = net(data)
    mask = mask.squeeze()
    test_loss = loss_fun(torch.log(test_out),labels)

    test_sum_loss += test_loss.item()
    pre = torch.argmax(test_out,dim=1)
    sorce = torch.mean(torch.eq(pre, labels).float())
    sum_sorce+= sorce

    labels1 = labels.detach().cpu()
    test_out1 = test_out.detach().cpu().numpy()[:,1]
    # test_out1 = f(test_out1)[:,1]
    a = torch.ones(1, 632).cuda()
    b = torch.zeros(1, 632).cuda()
    mask1 = torch.where(mask > 0.8, a, b)
    fpr, tpr, thresholds = metrics.roc_curve(labels1, test_out1)
    precision, recall, thresholds2 = metrics.precision_recall_curve(labels1, test_out1)
    # cm = metrics.confusion_matrix(labels1,test_out)
    roc_auc = metrics.auc(fpr,tpr)
    plt.subplot(121)
    plt.plot(fpr,tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel("True Positive Rate")
    plt.title("ROC")
    plt.subplot(122)
    plt.plot(recall, precision)
    plt.ylabel("Recall")
    plt.xlabel("Precision")
    plt.title("P-R")

    plt.subplots_adjust(wspace=0.35)

    plt.show()
    print('label:',labels)
    print(mask)
    print(mask1)

test_avg_loss = test_sum_loss / len(test_loader)
test_avg_score = sum_sorce / len(test_loader)
print("test_avg_loss:",test_avg_loss,"test_avg_score:", test_avg_score.item())


