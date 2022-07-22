import torch
import torch.nn as nn
import math

#Spatial pyramid pooling that has approximately 50% overlap
def spp(feature_map, scale_vector, W, H):
    '''
    feature_map : the output feature maps from the convolutional layer

    scale_vector : should be a tuple or list that contains int with the scale factor of each block. 
    For example, (1, 2, 3), will produce three different pyramid levels that consist of 1, 4, and 9 blocks

    W : The width of the feature map

    H : The height of the feature  map
    '''
    spp = torch.tensor([])
    for n in scale_vector:
        p_width = math.ceil(2*W/(n+1))
        p_height = math.ceil(2*H/(n+1))
        s_width = math.ceil(W/(n+1))
        s_height = math.ceil(H/n+1)
        #print(p_width, p_height, s_width, s_height, sep=' ')
        max_pool = nn.MaxPool2d((p_width, p_height), stride=(s_width,s_height), ceil_mode=True)
        x = max_pool(feature_map)
        #print(x.shape)

        #Each block will have shape N*C*1*1, concat the block on the third dim
        spp = torch.cat((spp, x.view(feature_map.shape[0],feature_map.shape[1],-1,1)),dim=2)
    return spp