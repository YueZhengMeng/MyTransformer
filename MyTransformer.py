import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Implement the PE function."""

    def __init__(self, max_len=5000, d_model=512, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化Shape为(max_len, d_model)的PE(positional encoding)
        pe = torch.zeros(max_len, d_model)
        # 初始化一个tensor [[0, 1, 2, 3, ...]],(max_len, 1)维
        # position是每个词向量在句子中的位置下标
        position = torch.arange(0, max_len).unsqueeze(1)
        # i是词向量中每个维度的位置下标
        # 计算公式中的 1/(10000^(2i/d_model)),用公式e^(lnx) = x 换底
        # 1/(10000^(2i/d_model)) = 1/(e^(2i/d_model * ln(10000)))
        #                        = e^(-2i/d_model * ln(10000))
        #                        = e^(2i * -ln(10000) / d_model)
        # 最终得到一个 d_model/2 维的向量
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        # 计算PE(pos, 2i), (max_len, d_model/2)维
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算PE(pos, 2i+1), (max_len, d_model/2)维
        pe[:, 1::2] = torch.cos(position * div_term)
        # 为了后续与word_embedding相加,在最外面在unsqueeze出一个batch_size维
        pe = pe.unsqueeze(0)
        # 如果一个参数不参与梯度下降,但又希望保存model的时候将其保存下来
        # 这个时候就可以用register_buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        # 将x和positional encoding相加。
        # 取前seq_len个位置的positional encoding
        # x的shape是(batch_size, seq_len, d_model)
        # pe的shape是(1, max_len, d_model)
        x = x + self.pe[:, : x.size(1), :].requires_grad_(False)
        return self.dropout(x)


class MyMultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, nhead=8):
        super(MyMultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.nhead = nhead
        # 每个head的维度
        # d_model维的词向量可以被视作是nhead个head_dim维的词向量的拼接
        self.head_dim = d_model // nhead

        # 定义Q、K、V的线性变换
        self.q_linear = nn.Linear(d_model, d_model)
        # K的bias在注意力计算中会被约去,所以这里不需要bias
        # 实际上,Transformer中所有的线性层的bias都可以去掉,有助于训练过程中的梯度稳定
        self.k_linear = nn.Linear(d_model, d_model, bias=False)
        self.v_linear = nn.Linear(d_model, d_model)
        # 最后的线性变换,将多头注意力的结果整合,映射回d_model
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, key_padding_mask=None, causal_mask=None):
        # 获取batch size
        batch_size = q.size(0)
        # 将Q、K、V进行线性变换
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)
        # 将Q、K、V进行分头
        q = q.view(batch_size, -1, self.nhead, self.head_dim)  # (batch_size, seq_len, nhead, head_dim)
        q = q.transpose(1, 2)  # (batch_size, nhead, seq_len, head_dim)
        k = k.view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        # 计算注意力
        # k.transpose(-2, -1) (batch_size, nhead, head_dim, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # print(scores[0][0])
        """
        key padding mask用于在 Q@K.T/sqrt(d_k) 得到的attention score矩阵中,将<pad> token对应的 列 全部置为-inf
        -inf在softmax之后就会变为0,之后基于attention score矩阵对 V 的加权求和时,就不会加入<pad> token的信息(乘以0)
        <pad> token对应的 行 不会被置为-inf,这是为了让<pad> token也能够接收其他token的信息,更新自己的词向量,得到训练
        项目根部录下有两张相关的示意图,可以帮助理解
        src只需要mask掉<pad> token
        而tgt还需要基于causal_mask,mask掉未来的token,确保训练与推理时的任务一致性
        pytorch官方实现中，会先把key_padding_mask和causal_mask都扩展到(batch_size, nhead, seq_len, seq_len)的形状后相加
        得到的merged_mask再加到scores上，这里为了方便阅读和理解，分开逐步处理
        """
        # 应用padding mask
        if key_padding_mask is not None:
            # (batch_size, seq_len) -> (batch_size, 1, 1, seq_len) --> (batch_size, nhead, seq_len, seq_len)
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(1).repeat(1, scores.shape[1], scores.shape[2], 1)
            # print(key_padding_mask[0][0])
            scores = scores + key_padding_mask
        # print(scores[0][0])
        # 应用causal mask
        if causal_mask is not None:
            # (seq_len, seq_len) --> (batch_size, nhead, seq_len, seq_len)
            causal_mask = causal_mask.expand(scores.shape[0], scores.shape[1], causal_mask.shape[0],
                                             causal_mask.shape[1])
            # print(causal_mask[0][0])
            scores = scores + causal_mask
        # print(scores[0][0])
        # 计算注意力权重
        scores = nn.functional.softmax(scores, dim=-1)
        # 注意力加权
        out = torch.matmul(scores, v)
        # 合并多头
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        # 最后的线性变换
        out = self.out(out)
        return out


class MyTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1):
        super(MyTransformerEncoderLayer, self).__init__()
        # 定义self attention层
        self.self_attn = MyMultiHeadAttention(d_model, nhead)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        # 定义feed forward层
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, src_key_padding_mask=None):
        # self attention
        src2 = self.self_attn(src, src, src, key_padding_mask=src_key_padding_mask, causal_mask=None)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # feedforward
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src


class MyTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1):
        super(MyTransformerDecoderLayer, self).__init__()
        # 定义self attention层
        self.self_attn = MyMultiHeadAttention(d_model, nhead)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        # 定义encoder-decoder attention层
        self.cross_attn = MyMultiHeadAttention(d_model, nhead)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        # 定义feed forward层
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, tgt_mask=None, memory_key_padding_mask=None, tgt_key_padding_mask=None):
        # self attention
        tgt2 = self.self_attn(tgt, tgt, tgt, key_padding_mask=tgt_key_padding_mask, causal_mask=tgt_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # encoder-decoder attention
        tgt2 = self.cross_attn(tgt, memory, memory, key_padding_mask=memory_key_padding_mask, causal_mask=None)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        # feedforward
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class MyTransformerEncoder(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=6, dropout=0.1):
        super(MyTransformerEncoder, self).__init__()
        encoder_layer = MyTransformerEncoderLayer(d_model, nhead, dropout=dropout)
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])

    def forward(self, src, src_key_padding_mask=None):
        for layer in self.layers:
            src = layer(src, src_key_padding_mask)
        return src


class MyTransformerDecoder(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=6, dropout=0.1):
        super(MyTransformerDecoder, self).__init__()
        decoder_layer = MyTransformerDecoderLayer(d_model, nhead, dropout=dropout)
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])

    def forward(self, tgt, memory, tgt_mask=None, memory_key_padding_mask=None, tgt_key_padding_mask=None):
        # decoder每一层输入的memory都是相同的,即encoder的输出
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask, memory_key_padding_mask, tgt_key_padding_mask)
        return tgt


class MyTransformer(nn.Module):
    def __init__(self, vocab_size, max_len, d_model=512, nhead=8, num_layers=6, dropout=0.1):
        super(MyTransformer, self).__init__()

        # 定义词向量编码
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        # 定义位置编码
        self.positional_encoding = PositionalEncoding(max_len=max_len, d_model=d_model, dropout=dropout)
        # 定义TransformerEncoder
        self.encoder = MyTransformerEncoder(d_model, nhead, num_layers, dropout)
        # 定义TransformerDecoder
        self.decoder = MyTransformerDecoder(d_model, nhead, num_layers, dropout)
        # 定义最后的线性层
        self.predictor = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, tgt_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None):
        # 对src和tgt进行编码
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        # 给src和tgt的token增加位置信息
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)
        # 将准备好的数据送给transformer encoder和decoder
        memory = self.encoder(src, src_key_padding_mask)
        out = self.decoder(tgt, memory, tgt_mask, src_key_padding_mask, tgt_key_padding_mask)
        # 这里直接返回transformer的输出。因为训练和推理时的流程不一样,所以在模型外再进行线性层的预测。
        return out
