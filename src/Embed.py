import numpy as np
import torch.nn as nn
import torch
from torch_geometric.nn.conv import GCNConv
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_att=None):
        x = self.conv1(x, edge_index, edge_att)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_att)
        return x

class GraphEmbedding2(nn.Module):
    def __init__(self, c_in, d_model, args=None):
        super(GraphEmbedding2, self).__init__()
        self.c_in = c_in
        self.d_model = d_model
        self.gcn = GCN(d_model, d_model, d_model)
        self.args = args
        self.temporal_conv = nn.Conv1d(in_channels=1, out_channels=d_model, kernel_size=args.t_patch_len, stride=args.t_patch_len)


    def forward(self, x, edges=None, node_split=None):
        '''
        x: N, T, 1, H, 1
        '''
        N, _, T, H, _ = x.shape
        
        x = x.squeeze((1,4)).permute(0,2,1).reshape(N * H, 1, self.args.his_len + self.args.pred_len)
        temporal_value_emb = self.temporal_conv(x).permute(0,2,1).reshape(N, H, -1, self.d_model).permute(0,2,1,3).reshape(-1, H, self.d_model)
        TokenEmb = self.gcn(temporal_value_emb, edges.permute(1,0)).reshape(N, -1, H, self.d_model) # N * seqlen//t_patch_len * H * D
        TokenEmb = torch.cat([torch.mean(torch.gather(TokenEmb, 2 ,group.view(1, 1, group.shape[0], 1).expand(TokenEmb.shape[0], TokenEmb.shape[1], group.shape[0], TokenEmb.shape[3]).to(x).long()),dim=2) for group in node_split],2).reshape(N, -1, TokenEmb.shape[-1])

        return TokenEmb


class GraphEmbedding(nn.Module):
    def __init__(self, c_in, d_model, in_dim = 105, args=None):
        super(GraphEmbedding, self).__init__()
        self.c_in = c_in
        self.d_model = d_model
        self.args = args
        self.temporal_conv = nn.Conv1d(in_channels=1, out_channels=d_model, kernel_size=args.t_patch_len, stride=args.t_patch_len)


    def forward(self, x, edges=None, node_split=None):
        '''
        x: N, T, 1, H, 1
        '''
        N, _, T, H, _ = x.shape
        
        x = x.squeeze((1,4)).permute(0,2,1).reshape(N * H, 1, self.args.his_len + self.args.pred_len)
        temporal_value_emb = self.temporal_conv(x).permute(0,2,1).reshape(N, H, -1, self.d_model).permute(0,2,1,3).reshape(-1, H, self.d_model)

        # seq_len = [len(group) for group in node_split]
        # max_len = max(seq_len)

        #temporal_value_emb = torch.cat((temporal_value_emb, torch.zeros(temporal_value_emb.shape[0],1,self.d_model).to(x)),dim=1) # 加一个维度

        TokenEmb = torch.cat([torch.mean(torch.index_select(temporal_value_emb,1,group),dim=1).unsqueeze(dim=1) for group in node_split],dim=1)                                
        TokenEmb = TokenEmb.reshape(N, -1, TokenEmb.shape[-2], TokenEmb.shape[-1]).reshape(N,-1,self.d_model)

        return TokenEmb



class TokenEmbedding_S(nn.Module):
    def __init__(self, c_in, d_model,  patch_size):
        super(TokenEmbedding_S, self).__init__()
        kernel_size = [patch_size, patch_size]
        self.tokenConv = nn.Conv2d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=kernel_size, stride=kernel_size)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # B*T, C, H, W = x.shape
        x = self.tokenConv(x)
        x = x.flatten(2)
        x = torch.einsum("ncs->nsc", x)  # [N, H*W, C]
        return x

class TokenEmbedding_ST(nn.Module):
    def __init__(self, c_in, d_model, patch_size, t_patch_len,  stride):
        super(TokenEmbedding_ST, self).__init__()
        kernel_size = [t_patch_len, patch_size, patch_size]
        stride_size = [stride, patch_size,  patch_size]
        self.tokenConv = nn.Conv3d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=kernel_size, stride=stride_size)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # B, C, T, H, W = x.shape
        x = self.tokenConv(x)
        x = x.flatten(3)
        x = torch.einsum("ncts->ntsc", x)  # [N, T, H*W, C]
        x = x.reshape(x.shape[0], -1, x.shape[-1])  # [N, T*H*W, C]
        return x


class TimeEmbedding(nn.Module):
    def __init__(self, d_model, t_patch_len, stride):
        super(TimeEmbedding, self).__init__()

        self.weekday_embed = nn.Embedding(7, d_model)
        self.hour_embed_24 = nn.Embedding(24, d_model)
        self.hour_embed_48 = nn.Embedding(48, d_model)
        self.hour_embed_96 = nn.Embedding(96, d_model)
        self.hour_embed_288 = nn.Embedding(288, d_model)
        self.padding_patch_layer = nn.ReplicationPad1d((0, t_patch_len-stride)) 
        self.timeconv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=t_patch_len, stride=stride)

        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x_mark, hour_num):
        # B, T, 2 = x_mark.shape
        if '24' in hour_num:
            TimeEmb = self.hour_embed_24(x_mark[:,:,1]) + self.weekday_embed(x_mark[:,:,0])
        elif '48' in hour_num:
            TimeEmb = self.hour_embed_48(x_mark[:,:,1]) + self.weekday_embed(x_mark[:,:,0])
        elif '288' in hour_num:
            TimeEmb = self.hour_embed_288(x_mark[:,:,1]) + self.weekday_embed(x_mark[:,:,0])
        elif '96' in hour_num:
            TimeEmb = self.hour_embed_96(x_mark[:,:,1]) + self.weekday_embed(x_mark[:,:,0])
        TimeEmb = self.padding_patch_layer(TimeEmb.transpose(1,2))
        TimeEmb = self.timeconv(TimeEmb).transpose(1,2)
        return TimeEmb


def get_2d_sincos_pos_embed(embed_dim, grid_size1, grid_size2):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim]
    """
    grid_h = np.arange(grid_size1, dtype=np.float32)
    grid_w = np.arange(grid_size2, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size1, grid_size2])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed



def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb