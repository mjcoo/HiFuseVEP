import torch
import torch.nn as nn
import math

class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.pos_encoding = LearnablePositionalEncoding(d_model)
        self.dropout = nn.Dropout(rate)
        
        self.enc_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, dff, rate)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, mask=None):
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask)
            
        return x

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(LearnablePositionalEncoding, self).__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pe, std=0.02)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.GELU(),
            nn.Dropout(rate),
            nn.Linear(dff, d_model)
        )
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)
        
        # 参数初始化
        nn.init.xavier_normal_(self.ffn[0].weight)
        self.ffn[0].bias.data.zero_()
        nn.init.xavier_normal_(self.ffn[-1].weight)
        self.ffn[-1].bias.data.zero_()
    
    def forward(self, x, mask=None):
        # Pre-LN结构
        x_norm = self.layernorm1(x)
        attn_output = self.mha(x_norm, x_norm, x_norm, mask)
        attn_output = self.dropout1(attn_output)
        out1 = x + attn_output
        
        out1_norm = self.layernorm2(out1)
        ffn_output = self.ffn(out1_norm)
        ffn_output = self.dropout2(ffn_output)
        return out1 + ffn_output

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_position=512):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads
        self.max_position = max_position

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        
        # 相对位置编码参数
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(2 * max_position - 1, num_heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        
        # 生成相对位置索引
        coords = torch.arange(max_position)
        relative_coords = coords[:, None] - coords[None, :]
        relative_coords += max_position - 1
        self.register_buffer("relative_position_index", relative_coords)
        
        self.dense = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)
        
        # 参数初始化
        nn.init.xavier_normal_(self.wq.weight)
        nn.init.xavier_normal_(self.wk.weight)
        nn.init.xavier_normal_(self.wv.weight)
        self.wq.bias.data.zero_()
        self.wk.bias.data.zero_()
        self.wv.bias.data.zero_()
        
    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, q, k, v, mask=None):
        batch_size, seq_len = q.size(0), q.size(1)
        
        # 线性投影并分头
        q = self.split_heads(self.wq(q), batch_size)
        k = self.split_heads(self.wk(k), batch_size)
        v = self.split_heads(self.wv(v), batch_size)
        
        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.depth)
        
        # 添加相对位置偏置
        if seq_len > self.max_position:
            raise ValueError(f"Sequence length {seq_len} exceeds max_position {self.max_position}")
        rel_pos_bias = self.relative_position_bias_table[
            self.relative_position_index[:seq_len, :seq_len].flatten()
        ].view(seq_len, seq_len, -1).permute(2, 0, 1)
        attn_scores += rel_pos_bias.unsqueeze(0)
        
        # 应用mask
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # 计算注意力权重
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 上下文向量
        context = torch.matmul(attn_weights, v)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, seq_len, -1)
        return self.dense(context)