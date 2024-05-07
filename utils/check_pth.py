import torch

pthfile = r'checkpoints/handwriting_iter_070.pth' #faster_rcnn_ckpt.pth
net = torch.load(pthfile,map_location=torch.device('cpu'))
print(type(net))
print(len(net))