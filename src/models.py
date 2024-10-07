"""
Originally inspired by impl at https://github.com/facebookresearch/DiT/blob/main/models.py

Modified by Haoyu Lu, for video diffusion transformer
"""
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# DiT: https://github.com/facebookresearch/DiT/blob/main/models.py
# 
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from einops import rearrange, reduce, repeat

from Embed import GraphEmbedding,GraphEmbedding2, TokenEmbedding_S, TokenEmbedding_ST, get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid, TimeEmbedding

class Memory(nn.Module):
    """ Memory prompt
    """
    def __init__(self, num_memory, memory_dim, args=None):
        super().__init__()

        self.args = args

        self.num_memory = num_memory
        self.memory_dim = memory_dim

        self.memMatrix = nn.Parameter(torch.zeros(num_memory, memory_dim))  # M,C
        self.keyMatrix = nn.Parameter(torch.zeros(num_memory, memory_dim))  # M,C

        self.memMatrix.requires_grad = True
        self.keyMatrix.requires_grad = True

        self.x_proj = nn.Linear(memory_dim, memory_dim)

        self.initialize_weights()

        print("model initialized memory")


    def initialize_weights(self):
        torch.nn.init.trunc_normal_(self.memMatrix, std=0.02)
        torch.nn.init.trunc_normal_(self.keyMatrix, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self,x):
        """
        :param x: query features with size [N,C], where N is the number of query items,
                  C is same as dimension of memory slot

        :return: query output retrieved from memory, with the same size as x.
        """
        # dot product


        assert x.shape[-1]==self.memMatrix.shape[-1]==self.keyMatrix.shape[-1], "dimension mismatch"

        x_query = torch.tanh(self.x_proj(x))

        att_weight = F.linear(input=x_query, weight=self.keyMatrix)  # [N,C] by [M,C]^T --> [N,M]

        att_weight = F.softmax(att_weight, dim=-1)  # NxM

        out = F.linear(att_weight, self.memMatrix.permute(1, 0))  # [N,M] by [M,C]  --> [N,C]

        return dict(out=out, att_weight=att_weight)



def modulate(x, shift, scale, T):
    N, M = x.shape[-2], x.shape[-1]
    B = scale.shape[0]
    if T==1:
        B = x.shape[0]
    x = rearrange(x, '(b t) n m-> b (t n) m',b=B,t=T,n=N,m=M)
    x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    x = rearrange(x, 'b (t n) m-> (b t) n m',b=B,t=T,n=N,m=M)
    return x


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb



def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

#################################################################################
#                                 Core UrbanDiT Model                                #
#################################################################################

class UrbanDiTBlock(nn.Module):
    """
    A UrbanDiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0,  num_frames=24, is_spatial=1, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.num_frames = num_frames

        self.is_spatial = is_spatial
        
        ## Temporal Attention Parameters
        self.temporal_norm1 = nn.LayerNorm(hidden_size)
        self.temporal_attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True)
        self.temporal_fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, c, num_frames=None, pt = None, ps = None, pf=None, pms=None, pmt=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        T = self.num_frames if num_frames is None else num_frames
        K, N, M = x.shape
        B = K // T

        if num_frames is None:
            x = rearrange(x, '(b t) n m -> (b n) t m',b=B,t=T,n=N,m=M)

            if pt is not None:
                x = torch.cat((pt.reshape(-1,1,pt.shape[-1]), x),dim=1) # seq_len + 1
            if pf is not None:
                x = torch.cat((pf.reshape(-1,1,pf.shape[-1]), x),dim=1) # seq_len + 2
            if pmt is not None:
                x = torch.cat((pmt, x),dim=1) # seq_len + 3

            res_temporal = self.temporal_attn(self.temporal_norm1(x))
            if pt is not None:
                res_temporal = res_temporal[:,1:]
                x = x[:,1:]
            if pf is not None:
                res_temporal = res_temporal[:,1:]
                x = x[:,1:]
            if pmt is not None:
                res_temporal = res_temporal[:,1:]
                x = x[:,1:]
            
            res_temporal = rearrange(res_temporal, '(b n) t m -> (b t) n m',b=B,t=T,n=N,m=M)
            res_temporal = self.temporal_fc(res_temporal)
            
            x = rearrange(x, '(b n) t m -> (b t) n m',b=B,t=T,n=N,m=M)
            x = x + res_temporal

        if ps is not None:
            x = torch.cat((ps.reshape(-1,1,ps.shape[-1]), x),dim=1)

        if pms is not None:
            x = torch.cat((pms, x),dim=1)

        if self.is_spatial == 1:
            attn = self.attn(modulate(self.norm1(x), shift_msa, scale_msa, self.num_frames if num_frames is None else num_frames))

            if ps is not None:
                attn = attn[:,1:]
                x = x[:,1:]

            if pms is not None:
                attn = attn[:,1:]
                x = x[:,1:]

            attn = rearrange(attn, '(b t) n m-> b (t n) m',b=B,t=T,n=N,m=M)
            attn = gate_msa.unsqueeze(1) * attn
            attn = rearrange(attn, 'b (t n) m-> (b t) n m',b=B,t=T,n=N,m=M)
            x = x + attn

            mlp = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp, self.num_frames if num_frames is None else num_frames))
            mlp = rearrange(mlp, '(b t) n m-> b (t n) m',b=B,t=T,n=N,m=M)
            mlp = gate_mlp.unsqueeze(1) * mlp
            mlp = rearrange(mlp, 'b (t n) m-> (b t) n m',b=B,t=T,n=N,m=M)
            x = x + mlp

        return x

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class FinalLayer(nn.Module):
    """
    The final layer of UrbanDiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels, num_frames, stride, dim=1):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels * stride * dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.num_frames = num_frames

    def forward(self, x, c, num_frames = None):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale, self.num_frames if num_frames is None else num_frames)
        x = self.linear(x)
        return x


class UrbanDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=16,
        in_channels=1,
        hidden_size=16,
        depth=4,
        num_heads=8,
        mlp_ratio=2,
        class_dropout_prob=0.1,
        learn_sigma=False,
        args = None
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        #self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        if args.time_patch==1:
            self.x_embedder = TokenEmbedding_ST(in_channels, hidden_size, patch_size, args.t_patch_len, args.stride)
            self.x_embedder_super = TokenEmbedding_ST(in_channels, hidden_size, patch_size, 1, 1)
            self.mask_embedder = TokenEmbedding_ST(in_channels, hidden_size, patch_size, args.t_patch_len, args.stride)
            self.graph_embedder = GraphEmbedding(1, hidden_size, args=args)

            self.x_embedder_graph = TokenEmbedding_ST(in_channels, hidden_size, 1, args.t_patch_len, args.stride)
            self.mask_embedder_graph = TokenEmbedding_ST(in_channels, hidden_size, 1, args.t_patch_len, args.stride)

        else:
            self.x_embedder = TokenEmbedding_S(in_channels, hidden_size, patch_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.time_embedder = TimeEmbedding(hidden_size, args.t_patch_len, args.stride)
        self.time_embedder_super = TimeEmbedding(hidden_size, 1, 1)
        self.args = args
        
        self.num_frames = args.pred_len + args.his_len
        if self.args.time_patch==1:
            self.num_frames = self.num_frames // self.args.stride
        self.time_embed = nn.Parameter(torch.zeros(1, args.pred_len + args.his_len, hidden_size), requires_grad=False)
        self.time_drop = nn.Dropout(p=0)

        self.blocks = nn.ModuleList([
            UrbanDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,  num_frames=self.num_frames, is_spatial = args.is_spatial) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, self.num_frames, self.args.stride)
        self.final_graph_layer = FinalLayer(hidden_size, 1, self.out_channels, self.num_frames, self.args.stride, dim=1)
        self.final_GraphGDSH  = FinalLayer(hidden_size, 1, self.out_channels, self.num_frames, self.args.stride, dim=210)
        self.final_GraphGDBJ  = FinalLayer(hidden_size, 1, self.out_channels, self.num_frames, self.args.stride, dim=105)
        self.final_GraphGDNJ  = FinalLayer(hidden_size, 1, self.out_channels, self.num_frames, self.args.stride, dim=105)
        
        self.W_P = nn.Linear(args.t_patch_len, 1)

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.args.t_patch_len-self.args.stride)) 

        self.enc_memory_temporal = Memory(num_memory=args.num_memory, memory_dim=hidden_size, args=self.args)
        self.enc_memory_spatial = Memory(num_memory=args.num_memory, memory_dim=hidden_size, args=self.args)
        self.enc_memory_freq = Memory(num_memory=args.num_memory, memory_dim=hidden_size, args=self.args)

        self.encoder_t = Attention(hidden_size, num_heads=4, qkv_bias=True)

        if args.fft==0:
            self.encoder_f = TokenEmbedding_S(int(args.pred_len+args.his_len+2), hidden_size, patch_size)
            self.linear_f = nn.Linear(int(args.pred_len+args.his_len+2), hidden_size)

            self.encoder_f_graph = TokenEmbedding_S(int(args.pred_len+args.his_len+2), hidden_size, 1)

        elif args.fft==1:
            self.encoder_f = TokenEmbedding_S(int(args.pred_len+args.his_len), hidden_size, patch_size)
            self.linear_f = nn.Linear(int(args.pred_len+args.his_len), hidden_size)

            self.encoder_f_graph = TokenEmbedding_S(int(args.pred_len+args.his_len), hidden_size, 1)

        self.encoder_s = Attention(hidden_size, num_heads=4, qkv_bias=True)
        self.mask_t = Attention(hidden_size, num_heads=1, qkv_bias=True)
        self.mask_s = Attention(hidden_size, num_heads=1, qkv_bias=True)
        
        self.initialize_weights()

    def get_prompt(self, raw_x, B, T, x_f):
        _, M, D = raw_x.shape # B*T, M, D 
        x_t = raw_x.reshape(B, T, M, D).permute(0,2,1,3).reshape(-1,T,D)
        x_t = self.encoder_t(x_t)[:,-1] # B * M
        x_s = self.encoder_s(raw_x)[:,-1]


        prompt_f = self.enc_memory_freq(x_f.reshape(-1,x_f.shape[-1]))['out'].reshape(B, M, D)
        prompt_t = self.enc_memory_temporal(x_t)['out'].reshape(B, M, D)
        prompt_s = self.enc_memory_spatial(x_s)['out'].reshape(B, T, D)

        return dict(pt=prompt_t, ps=prompt_s, pf = prompt_f)

    def get_pos_emb(self, size1, size2):
        pos_embed = get_2d_sincos_pos_embed(
            self.hidden_size,
            grid_size1 = size1,
            grid_size2 = size2
        )
        pos_emb = nn.Parameter(torch.zeros(pos_embed.shape).unsqueeze(dim=0), requires_grad=False)
        pos_emb.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        pos_emb.require_grad=False

        return pos_emb

    def get_time_emb(self):
        grid_num_frames = np.arange(self.num_frames, dtype=np.float32)
        pos_embed = get_1d_sincos_pos_embed_from_grid(self.hidden_size, grid_num_frames)
        pos_emb = nn.Parameter(torch.zeros(pos_embed.shape).unsqueeze(dim=0), requires_grad=False)
        pos_emb.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        pos_emb.require_grad=False
        return pos_emb


    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)


        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in UrbanDiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        nn.init.constant_(self.final_graph_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_graph_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_graph_layer.linear.weight, 0)
        nn.init.constant_(self.final_graph_layer.linear.bias, 0)

        nn.init.constant_(self.final_GraphGDSH.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_GraphGDSH.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_GraphGDSH.linear.weight, 0)
        nn.init.constant_(self.final_GraphGDSH.linear.bias, 0)

        nn.init.constant_(self.final_GraphGDBJ.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_GraphGDBJ.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_GraphGDBJ.linear.weight, 0)
        nn.init.constant_(self.final_GraphGDBJ.linear.bias, 0)

        nn.init.constant_(self.final_GraphGDNJ.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_GraphGDNJ.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_GraphGDNJ.linear.weight, 0)
        nn.init.constant_(self.final_GraphGDNJ.linear.bias, 0)

    def unpatchify(self, x, h, w, mask_idx=None, data_name='',node_split=None, p=None):
        """
        x: (N, T, patch_size**2 * C * stride)
        """
        c = self.out_channels
        if p is None:
            p = self.patch_size


        if 'Graph' not in data_name:
            assert h * w == x.shape[1]
            if mask_idx is None:
                x = x.reshape(shape=(x.shape[0], h, w, p, p, self.args.stride, c))
            else:
                x = x.reshape(shape=(x.shape[0], h, w, p, p, 1, c))
            x = torch.einsum('nhwpqsc->nschpwq', x)
            if mask_idx is None:
                imgs = x.reshape(shape=(x.shape[0], self.args.stride,  c, h * p, w * p))
            else:
                imgs = x.reshape(shape=(x.shape[0], 1,  c, h * p, w * p))
            return imgs

        else:
            seq_lengths = [len(i) for i in node_split]
            x = x.reshape(shape=(x.shape[0], x.shape[1], 1, -1, 1,  self.args.stride, c))
            x = torch.einsum('nhwpqsc->nschpwq', x)
            x = x.squeeze(dim=(-1,-2,-3))
            if len(x.shape)==4:
                x = x.unsqueeze(dim=-1)
            else:
                x = torch.cat([x[:,:,:,g,:seq_lengths[g]] for g in range(x.shape[3])],dim=3).unsqueeze(dim=-1)
            return x


    def forward(self, x, t, timestamps, hour_num, mask_idx=0, coarse=None, mask=None, data_name = '', node_split=None,topo=None):
        """
        Forward pass of UrbanDiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        """

        B, T, C, H, W = x.shape # 32 24 1 32 32 


        assert timestamps.shape == (B, T, 2)

        if self.args.fft==0:
            x_f = torch.fft.rfft(x.permute(0,2,3,4,1), n = T, norm = "ortho", dim = -1).squeeze(dim=1)
            x_f  = torch.cat((x_f.real, x_f.imag), dim = -1).permute(0,3,1,2)
        elif self.args.fft==1:
            x_f = torch.fft.rfft(x.permute(0,2,3,4,1), n = T, norm = "ortho", dim = -1).squeeze(dim=1)
            amplitude = torch.abs(x_f)
            if self.args.fft_thred==0:
                threshold = amplitude.mean()
                mask_temp = amplitude > threshold
            elif self.args.fft_thred==1:
                threshold = torch.quantile(amplitude, 0.8).to(x)
                mask_temp = amplitude > threshold
            elif self.args.fft_thred==2:
                max_freq_idx = 3
                mask_temp = torch.zeros_like(x_f, dtype=torch.bool).to(x)
                mask_temp[:max_freq_idx] = True
            X_f_filtered = x_f * mask_temp
            x_f = torch.fft.irfft(X_f_filtered, n = T, norm = "ortho", dim = -1).permute(0,3,1,2)

        if 'Graph' not in data_name:
            if 'TrafficG' not in data_name:
                x_f = self.encoder_f(x_f)
                pos_embed = self.get_pos_emb(x.shape[-2]//self.patch_size,x.shape[-1]//self.patch_size)
            else:
                x_f = self.encoder_f_graph(x_f)
                pos_embed = self.get_pos_emb(x.shape[-2]//1,x.shape[-1]//1)

        else:
            x_f = self.linear_f(x_f.permute(0,2,3,1)) 
            max_len = max([len(i) for i in node_split])
            n_group = len(node_split)
            x_f = torch.cat([torch.mean(torch.gather(x_f, 1 ,group.view(1, group.shape[0], 1, 1).expand(x_f.shape[0], group.shape[0], x_f.shape[2], x_f.shape[3]).to(x).long()),dim=1) for group in node_split],1)
            pos_embed = self.get_pos_emb(len(node_split),1)

        time_pos_embed = self.get_time_emb().unsqueeze(dim=2).repeat(B,1,pos_embed.shape[-2],1).reshape(-1,pos_embed.shape[-2],pos_embed.shape[-1])

        D = pos_embed.shape[-1]

        if self.args.time_patch==1:
            T = T//self.args.stride
            x = x.contiguous().permute(0,2,1,3,4)
            if 'Graph' not in data_name:
                if 'TrafficG' not in data_name:
                    x = self.x_embedder(x).reshape(B, T, -1, D)
                    mask = self.mask_embedder(mask.unsqueeze(dim=1).float().to(x)).reshape(B*T,-1,D)
                else:
                    x = self.x_embedder_graph(x).reshape(B, T, -1, D)
                    mask = self.mask_embedder_graph(mask.unsqueeze(dim=1).float().to(x)).reshape(B*T,-1,D)
            else:
                x = self.graph_embedder(x, topo, node_split).reshape(B, T, -1, D)
                mask = self.mask_embedder_graph(mask.unsqueeze(dim=1).float().to(x), topo, node_split).reshape(B*T,-1,D)
                
            time_embed = self.time_embedder(timestamps, hour_num).unsqueeze(dim=-2).repeat(1,1,x.shape[2],1).view(B*T, -1, D)

            x = x.reshape(B * T, -1, D)

            prompt = dict(pt=None, ps=None, pf = None, pmt=None, pms=None)

            if self.args.is_prompt==1:
                if 'pt' in self.args.prompt_content or 'ps' in self.args.prompt_content or 'pf' in self.args.prompt_content:
                    prompt = self.get_prompt(x, B, T, x_f)
                    if 'pt' not in self.args.prompt_content:
                        prompt['pt'] = None
                    if 'ps' not in self.args.prompt_content:
                        prompt['ps'] = None
                    if 'pf' not in self.args.prompt_content:
                        prompt['pf'] = None
                if 'pm' in self.args.prompt_content:
                    prompt['pms'] = self.mask_s(mask)[:,-1:]
                    prompt['pmt'] = self.mask_t(rearrange(mask, '(b t) n m -> (b n) t m',b=B,t=T,n=mask.shape[1],m=mask.shape[2]))[:,-1:]
                else:
                    prompt['pms'] = None
                    prompt['pmt'] = None

            x = x + time_embed + pos_embed.to(x) + time_pos_embed.to(x) # (N, T, D), where T = H * W / patch_size ** 2

        else:
            x = x.contiguous().view(-1,C,H,W)
            time_embed = self.time_embedder(timestamps, hour_num).unsqueeze(dim=-2).repeat(1,1,pos_embed.shape[1],1).view(B*T, -1, D)
            x = self.x_embedder(x) 
            x = x + pos_embed.to(x) + time_embed + time_pos_embed.to(x)


        # Temporal embed
        x = rearrange(x, '(b t) n m -> (b n) t m',b=B,t=T)
        ## Resizing time embeddings in case they don't match
        #x = x + self.time_embed
        x = self.time_drop(x)
        x = rearrange(x, '(b n) t m -> (b t) n m',b=B,t=T)
        
        t = self.t_embedder(t)                   # (N, D)
        #y = self.y_embedder(y, self.training)    # (N, D)

        #c = t + y                             # (N, D)
        c = t
        
        for block in self.blocks:
            x = block(x, c, num_frames=None)      , pt=prompt['pt'], ps=prompt['ps'], pf = prompt['pf'], pms=prompt['pms'],pmt=prompt['pmt'])                      # (N, T, D)
        
        if 'Graph' not in data_name:
            if 'TrafficG' not in data_name:
                x = self.final_layer(x, c, num_frames=None)                # (N, T, patch_size ** 2 * out_channels)
            else:
                x = self.final_graph_layer(x, c, num_frames=None)      
        else:
            if 'BJ' in data_name:
                x = self.final_GraphGDBJ(x, c, num_frames=None)      
            elif 'NJ' in data_name:
                x = self.final_GraphGDNJ(x, c, num_frames=None)      
            elif 'SH' in data_name:
                x = self.final_GraphGDSH(x, c, num_frames=None)      

        if 'TrafficG' not in data_name:
            x = self.unpatchify(x, H//self.patch_size, W//self.patch_size, data_name = data_name, node_split = node_split) # (N, out_channels, H, W)
        else:
            x = self.unpatchify(x, H, W, data_name = data_name, node_split = node_split, p=1)

        x = x.view(B, T, x.shape[-4], x.shape[-3], x.shape[-2], x.shape[-1])
        x = x.reshape(B, -1, x.shape[-3], x.shape[-2], x.shape[-1])

        if 'Graph' in data_name:
            index = torch.cat(node_split,dim=0).to(x).long()
            inverse_index = torch.zeros_like(index).to(index)
            inverse_index.scatter_(dim=0, index=index, src=torch.arange(index.size(0)).to(index))
            inverse_index = inverse_index.view(1, 1, 1, -1, 1).expand(x.shape[0], x.shape[1], 1, -1, 1)
            x = torch.gather(x, dim=3, index=inverse_index)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of UrbanDiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)





#################################################################################
#                                   UrbanDiT Configs                                  #
#################################################################################

def UrbanDiT_S_1(**kwargs):
    return UrbanDiT(depth=4, hidden_size=256, patch_size=2, num_heads=4,  **kwargs)

def UrbanDiT_S_2(**kwargs):
    return UrbanDiT(depth=6, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def UrbanDiT_S_3(**kwargs):
    return UrbanDiT(depth=12, hidden_size=384, patch_size=2, num_heads=12, **kwargs)


UrbanDiT_models = {
    'UrbanDiT-S/1':  UrbanDiT_S_1,
    'UrbanDiT-S/2':  UrbanDiT_S_2, 
    'UrbanDiT-S/3':  UrbanDiT_S_3,  
}
