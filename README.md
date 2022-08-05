# Attention-based-Pyramid-Aggregation-Network-APANet


Project Not finished

## Current Problem
- [ ] issues for setting weights for the convolutional layer. Might have some difference compare to the article

## Introduction
APANet realization using pytorch based on [Attention-based Pyramid Aggregation Network for Visual Place Recognition](https://arxiv.org/abs/1808.00288)


## Usage Example

```
from APANet import APANet
import torch
import torch.nn as nn
from torchvision import models

#vgg16 backbone
encoder = models.vgg16(pretrained=True)
# capture only feature part and remove last relu and maxpool
layers = list(encoder.features.children())[:-2]

for l in layers[:-5]: 
    for p in l.parameters():
        p.requires_grad = False

encoder = nn.Sequential(*layers)

#Suppose we have 50 images
input_image = torch.rand((50,3,224,224))
x = encoder(input_image)

#perform single block and cascade block separately
Apa_single = APANet(scale_vector=(1,2,3) ,cascade=False)
Apa_cascade = APANet(scale_vector=(1,2,3) ,cascade=True)


x_single = torch.tensor([])
x_cascade = torch.tensor([])
#Process the image one by one
for i in range(x.shape[0]):
    x_single = torch.cat((x_single, Apa_single(x[i].unsqueeze(0))))
    x_cascade = torch.cat((x_cascade,Apa_cascade(x[i].unsqueeze(0))))
print(x_single.shape, x_cascade.shape, sep = ' ')
#Output torch.Size([50, 512, 1, 1]) torch.Size([50, 512, 1, 1])
```
