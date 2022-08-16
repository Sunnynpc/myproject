import torch

from net import Res_Net18,Net_v2
from torch import nn
from torch.utils.data import DataLoader
from dataset import dataset
import os
import numpy as np
from PIL import Image,ImageDraw
from utils import iou

save_path = "model/net_v2.pt"
DEVICE = "cuda"
if __name__ == '__main__':

    data_set = dataset("data")

    net = Res_Net18().to(DEVICE)
    # 加载预训练权重
    if os.path.exists(save_path):
        net.load_state_dict(torch.load(save_path))
        print("已加载预训练权重！")

    loss_func1 = nn.MSELoss()
    loss_func2 = nn.BCELoss()

    opt = torch.optim.Adam(net.parameters())
    Train = False
    while True:
        if Train:
            train_loader = DataLoader(dataset=data_set,batch_size=50,shuffle=True)
            for i,(x,y1,y2) in enumerate(train_loader):
                # x = x.reshape(-1,300*300*3).to(DEVICE)

                net.train()
                x = x.permute(0,3,1,2).to(DEVICE)
                y1 = y1.to(DEVICE)
                y2 = y2.to(DEVICE)

                out = net(x)

                out1 = out[0]
                out2 = out[1].squeeze()
                loss1 = loss_func1(out1,y1)
                loss2 = loss_func2(out2,y2)
                loss = loss1+loss2



                opt.zero_grad()

                loss.backward()

                opt.step()

                if i%30 ==0:
                    torch.save(net.state_dict(), save_path)
                    print(loss.item())

        else:

            test_loader = DataLoader(dataset=data_set, batch_size=1, shuffle=True)

            for i, (x, y1,y2) in enumerate(test_loader):
                # x = x.reshape(-1, 300 * 300 * 3).to(DEVICE)

                net.eval()
                x = x.permute(0, 3, 1, 2).to(DEVICE)
                y1 = y1.to(DEVICE)
                y2 = y2.to(DEVICE)

                out = net(x)

                loss11 = loss_func1(out[0], y1)
                loss22 = loss_func2(out[1].squeeze(), y2)
                loss = loss11 + loss22
                print(loss.item())

                # x = x.reshape(-1,300,300,3).cpu()
                x = x.permute(0, 2,3,1).cpu()
                out1 = out[0].detach().cpu().numpy()*300
                y1 = y1.detach().cpu().numpy() * 300
                print(y1)

                print("iou=",iou(out1[0],y1[0]))

                img_data = np.array((x[0]+0.5)*255,dtype=np.uint8)
                img = Image.fromarray(img_data,"RGB")
                draw = ImageDraw.Draw(img)

                draw.rectangle(np.array(y1[0]), outline="red",width=2)
                draw.rectangle(np.array(out1[0]), outline="yellow",width=2)
                img.show()

