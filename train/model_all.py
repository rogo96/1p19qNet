from collections import OrderedDict
import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, args):
        
        super(Model, self).__init__()
        self.args = args
        
        self.score = nn.Linear(2048, 1) 
         
        self.fc_final = nn.Linear(args.max_r, 1)  
        self.mlp = nn.Sequential(OrderedDict([
            ('fc2', nn.Linear(2*args.max_r, 200)),
            ('dropout1', nn.Dropout(p=0.5)),
            ('sigmoid_1', nn.Sigmoid()),
            ('fc3', nn.Linear(200,100)),
            ('dropout2', nn.Dropout(p=0.5)),
            ('sigmoid_2', nn.Sigmoid()),
            ('fc4', self.fc_final),
        ]))     
        self.last_activation = nn.Sigmoid()
        
    def forward(self,x):    
        B, T, F = x.shape
        x = x.view(B*T, F)      
        x = self.score(x).squeeze(1)  
        x = x.view(-1, T)  
        tile_score = x
        x, _ = torch.sort(x, dim=1, descending=True)
        x = torch.cat((x[:,:self.args.max_r], x[:,x.shape[1]-self.args.max_r:]), 1)  
            
        x = self.mlp(x) 
        
        return self.last_activation(x), tile_score
