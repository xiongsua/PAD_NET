import os,argparse
import numpy as np
import torchvision
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as tfs 
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

abs=os.getcwd()+'/'
def tensorShow(tensors,titles=['haze']):
        fig=plt.figure()
        for tensor,tit,i in zip(tensors,titles,range(len(tensors))):
            img = make_grid(tensor)
            npimg = img.numpy()
            ax = fig.add_subplot(221+i)
            ax.imshow(np.transpose(npimg, (1, 2, 0)))
            ax.set_title(tit)
        plt.show()

parser=argparse.ArgumentParser()
parser.add_argument('--task',type=str,default='its',help='its or ots')
parser.add_argument('--test_imgs',type=str,default='test_imgs',help='Test imgs folder')
opt=parser.parse_args()
dataset=opt.task

img_dir=r'../test_imgs/'

output_dir=r'../Test'

print("pred_dir:",output_dir)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

device='cuda' if torch.cuda.is_available() else 'cpu'
net = torch.load(r'../XXXXX.pth',map_location=torch.device('cpu'))


net.eval()
for im in os.listdir(img_dir):
    print(f'\r {im}',end='',flush=True)
    haze = Image.open("IMAGE PATH")
    #dif = net(diff)
    haze1= tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])
    ])(haze)[None,::]
    haze_no=tfs.ToTensor()(haze)[None,::]  # ?
 #  dog = tfs.ToTensor()(dif)
    with torch.no_grad():
        haze1 = haze1#.cuda()
        resize=torchvision.transforms.Resize(1000)
        haze1=resize(haze1)
        pred = net(haze1)
    ts=torch.squeeze(pred.clamp(0,1).cpu())
    tensorShow([haze_no,pred.clamp(0,1).cpu()],['haze','dehaze'])
   # tensorShow([haze_no, pred.clamp(0, 1).cpu()], [ 'dehaze'])
    vutils.save_image(ts,output_dir+im.split('.')[0]+'.png')
