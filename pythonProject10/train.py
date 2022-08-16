import torch
import torch.nn as nn
import os
from torchvision import transforms
from nets import MyNet
from centerloss import Centerloss
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils import data
from torchvision.datasets import MNIST
from arcsoftmax import Arcsoftmax

class Trainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_fn_cls = nn.NLLLoss()
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.1309,),(0.3084,))
        ])
    def train(self):
        BATCH_SIZE = 100
        save_path = "models/net_center.pth"
        if not os.path.exists("models"):
            os.mkdir("models")
        train_data = MNIST(root="F:\MNIST",train=True,download=False,transform=self.trans)
        train_loader = data.DataLoader(dataset=train_data,shuffle=True,batch_size=BATCH_SIZE)
        net = MyNet().to(self.device)
        arc = Arcsoftmax(2,10).to(self.device)
        # c_net = Centerloss().to(self.device)
        if os.path.exists(save_path):
            net.load_state_dict(torch.load(save_path))
        else:
            print("No Param")
        net_opt = torch.optim.SGD(net.parameters(),lr=0.001, momentum=0.9, weight_decay=0.0005)
        # net_opt = torch.optim.Adam(net.parameters())
        scheduler = lr_scheduler.StepLR(net_opt, 20, gamma=0.8)
        # c_net_opt = torch.optim.SGD(c_net.parameters(),lr=0.5)

        EPOCHS = 0
        while True:
            feat_loader = []
            label_loader = []
            for i,(x,y) in enumerate(train_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                feature,output = net(x)
                output = arc(feature)
                loss_cls = self.loss_fn_cls(output,y)
                y = y.float()
                # loss_center = c_net(feature,y)
                loss = loss_cls
                # loss = loss_cls+loss_center

                net_opt.zero_grad()
                # c_net_opt.zero_grad()
                loss.backward()
                net_opt.step()
                # c_net_opt.step()

                feat_loader.append(feature)
                label_loader.append(y)

                if i % 600 == 0:
                    # print("epoch:", EPOCHS, "i:", i, "total_loss:", (loss_cls.item() + loss_center.item()), "softmax_loss:",
                    #       loss_cls.item(), "center_loss:", loss_center.item())
                    print("epoch:", EPOCHS, "i:", i, "loss:", loss_cls.item())

            feat = torch.cat(feat_loader, 0)
            labels = torch.cat(label_loader, 0)

            net.visualize(feat.data.cpu().numpy(), labels.data.cpu().numpy(), EPOCHS)
            EPOCHS += 1
            torch.save(net.state_dict(), save_path)
            scheduler.step()
            # 150轮停止
            if EPOCHS == 150:
                break

if __name__ == '__main__':
    t = Trainer()
    t.train()


