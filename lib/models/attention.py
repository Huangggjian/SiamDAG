# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by HUANG JIAN (jian.huang@hdu.edu.cn)
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
        m[-1].inited = True
    else:
        constant_init(m, val=0)
        m.inited = True


class ContextBlock_in_SKnet(nn.Module):
    def __init__(self, inplanes, pool='att'):
        super(ContextBlock_in_SKnet, self).__init__()
        assert pool in ['avg', 'att']
        self.inplanes = inplanes
        self.pool = pool

        if 'att' in pool:
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.reset_parameters()

    def reset_parameters(self):
        if self.pool == 'att':
            kaiming_init(self.conv_mask, mode='fan_in')
            self.conv_mask.inited = True

        

    def spatial_pool(self, x):      # (N,256,7,7)
        batch, channel, height, width = x.size()
        if self.pool == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(3)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
            # [N, C]
            # context = context.view(batch, channel)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)
            context = context.view(batch, channel)

        return context

    def forward(self, x):
        # [N, C]
        context = self.spatial_pool(x)
       
        return context   

class SKConv_2(nn.Module):
    def __init__(self, features, M=3, G=1,r=8, stride=1, L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv_2, self).__init__()
        d = max(int(features / r), L)    # 32
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])  
    
        self.convs.append(nn.Sequential(
            nn.Conv2d(features, features, kernel_size=(1, 3), stride=stride, padding=(0, 1), groups=G),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=False)
        ))
        self.convs.append(nn.Sequential(
            nn.Conv2d(features, features, kernel_size=(3, 3), stride=stride, padding=(1, 1), groups=G),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=False)
        ))
        self.convs.append(nn.Sequential(
            nn.Conv2d(features, features, kernel_size=(3, 1), stride=stride, padding=(1, 0), groups=G),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=False)
        ))
      
        self.context = ContextBlock_in_SKnet(features)
        self.fc = nn.Sequential(
            nn.Conv2d(features, d, kernel_size=1), 
            nn.LayerNorm([d, 1, 1]),
            nn.ReLU(inplace=False)
        )
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, features)    
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, crop=True):
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
      
        fea_U = torch.sum(feas, dim=1)
      
        fea_s = self.context(fea_U)
        fea_z = self.fc(fea_s).squeeze_()  
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)   
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)            
        attention_vectors = self.softmax(attention_vectors)

        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)  
  
        # TODO:  batchsize in testing is 1
        if len(attention_vectors.shape) == 4:
            attention_vectors = attention_vectors.permute(1, 0, 2, 3)   
            attention_vectors = attention_vectors.unsqueeze(0)         

        fea_v = (feas * attention_vectors).sum(dim=1)
        if crop:
            return fea_v[:,:,1:-1,1:-1]

        return fea_v


if __name__ == '__main__':
    
    x = torch.randn(32,256,31,31)
    model = SKConv_2(256)
    out_ = model(x)
    print(out_.shape)
