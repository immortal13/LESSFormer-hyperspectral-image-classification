import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from HSI2Token import HSI2Token
from utils import FeatureConverter
from einops import rearrange, repeat
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout) ### 这里有没有dropout
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 2, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = 256 #128 #dim_head * heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
       
    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads #_为多少维的embedding，n是有多少个embedding，h = 3
        qkv = self.to_qkv(x).chunk(3, dim = -1) # b, n, inner_dim * 3
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv) 
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale # b, h, n, n
        dots_softmax = dots[0][0].softmax(dim=-1).float()

        ### the core code block: SNN mask
        mask_value = -torch.finfo(dots.dtype).max
        if mask is not None: #[b,h,m] #先得到[b,m]，然后h这个维度是复制得到的
            
            e = dots
            zero_vec = -9e15 * torch.ones_like(e) 
            dots = torch.where(mask > 0, e, zero_vec) #+ I
            dots = dots[0][0].cpu().detach().numpy()
            np.fill_diagonal(dots,1)
            dots = torch.from_numpy(dots).cuda()
            s1,s2 = dots.shape[0], dots.shape[1]
            dots = dots.reshape(1,1,s1,s2)
            
        attn = dots.softmax(dim=-1).float() #+ I
        # print(attn.shape, v.shape, "attn.shape, v.shape")
        out = torch.einsum('qwij,bhjd->bhid', attn, v) # b, h, n, inner_dim
        out = rearrange(out, 'b h n d -> b n (h d)') # b, n, h*inner_dim 把8个头concate成一个头
        out =  self.to_out(out) # FC，dropout
        return out, attn, dots_softmax

class Encoder_layer(nn.Module):
    def __init__(self, dim, num_head=4, mlp_ratio=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = Attention(
            dim, num_head)
        self.norm_mlp = norm_layer(dim)
        self.mlp = FeedForward(dim, int(dim * mlp_ratio))

    def forward(self, embed, A=None): #torch.Size([840, 200, 1])
        embed = self.norm(embed)
        out, attn, dots_softmax = self.attn(embed, A)
        embed = embed + out
        embed = self.norm_mlp(embed)
        embed = embed + self.mlp(embed)
        return embed, attn, dots_softmax


class Encoder(nn.Module):
    def __init__(self, num_patches, dim=32, depth=2,
                 num_head=2, mlp_ratio=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, num_patches, dim))#+1
        blocks = []
        for i in range(depth):
            blocks.append(Encoder_layer(
                dim=dim, num_head=num_head, mlp_ratio=mlp_ratio, norm_layer=norm_layer))
        self.blocks = nn.ModuleList(blocks)
        self.norm = norm_layer(dim)

    def forward(self, embed, A=None):
        for blk in self.blocks:
            embed, attn, dots_softmax = blk(embed, A)
        embed = self.norm(embed)
        return embed, attn, dots_softmax


class LESSFormer(nn.Module):
    def __init__(self, h: int, w: int, channel: int, class_count: int, superpixel_scale, group, d1=2, d2=2, h1=1, h2=1):
        super(LESSFormer, self).__init__()
        self.class_count = class_count  # 类别数
        self.channel = 128 #channel #64 
        self.height = h
        self.width = w 

        ## HSI2Token module
        #### spe paritition（仅需设置1个超参数：groups）
        self.groups = groups = group
        # print("group:",group)
        #### spa paritition （仅需设置3个超参数：superpixel_scale、ETA_POS、n_iters，但基本只需调整superpixel_scale）
        self.n_spixels = n_spixels = int(self.height*self.width/superpixel_scale) 
        global A # adjacency matrix
        A=0
        n_iters = 5  # iteration of DSLIC
        ETA_POS = 1.8 # scale coefficient of positional pixel features I^xy, it is very important
        self.HSI2Token = HSI2Token(FeatureConverter(ETA_POS), n_iters, n_spixels, self.channel, channel, cnn=True, groups=self.groups)
        
        ## Local-Enhanced Transformer
        #### SSN Encoder
        depth1 = d1
        depth2 = d2
        self.head = head1 = h1
        head2 = h2
        mlp_ratio = 1
        self.SNN_Encoder = Encoder(self.n_spixels, dim=int(self.channel/groups), depth=depth1, num_head=head1, mlp_ratio=mlp_ratio, norm_layer=nn.LayerNorm)
        #### Encoder
        self.Encoder = Encoder(groups, dim=int(self.channel/groups), depth=depth2, num_head=head2, mlp_ratio=mlp_ratio, norm_layer=nn.LayerNorm)
        ## Classification Head
        self.fc = nn.Linear(self.channel, self.class_count)
    
    def get_A(self,segments_map):
        # print(np.max(segments_map),self.n_spixels)
        A = np.zeros([self.n_spixels,self.n_spixels], dtype=np.float32)
        (h, w) = segments_map.shape
        for i in range(h - 2):
            for j in range(w - 2):
                sub = segments_map[i:i + 2, j:j + 2]
                sub_max = np.max(sub).astype(np.int32)
                sub_min = np.min(sub).astype(np.int32)
                if sub_max != sub_min:
                    idx1 = sub_max
                    idx2 = sub_min
                    if A[idx1, idx2] != 0:
                        continue
                    A[idx1, idx2] = A[idx2, idx1] = 1   
        np.fill_diagonal(A,0)

        ## 由[M,M] → [B,H,M,M]
        A = np.expand_dims(A,0).repeat(int(self.head),axis=0)
        A = np.expand_dims(A,0)
        # print(A.shape,"A.shape")

        A_cuda=torch.from_numpy(A).cuda()
        return A_cuda,A

    def forward(self, x: torch.Tensor):   
        (h, w, c) = x.shape

        ## HSI2Token module
        """
        Q:[1, 9, 145, 145] f:([1, 202, 145, 145]) spf:([1, 128, 841]) pf:([1, 128, 145, 145])
        Q is the pixel-superpixel assignment matrix, f = pos + HSI, spf is superpixel feature, pf is pixel feature
        A is the SNN mask
        """
        x_cnn = torch.unsqueeze(x.permute([2, 0, 1]), 0)
        Q, ops, f, spf, pf = self.HSI2Token(x_cnn) 
        Q_d = Q.detach()
        segments_map_cuda = ops['map_idx'](torch.argmax(Q_d, 1, True).int()) 
        segments_map = segments_map_cuda[0,0].cpu().numpy() #[1, 1, 145, 145]
        self.n_spixels = np.max(segments_map) + 1
        global A, vis_A_mask
        if isinstance(A,int):
            A,vis_A_mask=self.get_A(segments_map)

        ## Local-Enhanced Transformer
        #### SSN Encoder    
        spa_token = rearrange(spf[0], '(g c1) m -> g m c1', g=self.groups)
        spa_token, spa_attn, dots_softmax = self.SNN_Encoder(spa_token, A) #[G,L,C]
        
        #### Encoder
        spe_spa_token = rearrange(spa_token, 'g m c -> m g c') #spe_token.squeeze(-1)..squeeze(-1) #(M, 320)
        spe_spa_token, spe_spa_attn, _ = self.Encoder(spe_spa_token) #[m,8,16]
        spe_spa_token = rearrange(spe_spa_token, 'm g c -> m (g c)')

        ## Classification Head
        spe_spa_token = ops['map_sp2p'](torch.unsqueeze(spe_spa_token.t(),0).contiguous(), Q_d) #[1,C,H,W]
        spe_spa_token = torch.squeeze(spe_spa_token, 0).permute([1, 2, 0]).reshape([h * w, -1]) #[H*W,C]
        x = self.fc(spe_spa_token)
        x = F.softmax(x, -1)
        
        return x, Q, ops, segments_map_cuda

