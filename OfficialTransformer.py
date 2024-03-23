import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Implement the PE function."""

    def __init__(self, d_model=512, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化Shape为(max_len, d_model)的PE (positional encoding)
        pe = torch.zeros(max_len, d_model)
        # 初始化一个tensor [[0, 1, 2, 3, ...]]
        position = torch.arange(0, max_len).unsqueeze(1)
        # 这里就是sin和cos括号中的内容，通过e和ln进行了变换
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        # 计算PE(pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算PE(pos, 2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 为了方便计算，在最外面在unsqueeze出一个batch
        pe = pe.unsqueeze(0)
        # 如果一个参数不参与梯度下降，但又希望保存model的时候将其保存下来
        # 这个时候就可以用register_buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        # 将x和positional encoding相加。
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class OfficialTransformer(nn.Module):

    def __init__(self, vocab_size, d_model=128, nhead=8, num_layers=6, dropout=0.1):
        super(OfficialTransformer, self).__init__()

        # 定义词向量，词典数为10。
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        # 定义位置编码器
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)
        # 定义Transformer。超参是我拍脑袋想的
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers,
                                          dim_feedforward=d_model * 4,
                                          dropout=dropout,
                                          batch_first=True)
        # 定义最后的线性层，这里并没有用Softmax，因为后面的CrossEntropyLoss中自带了
        self.predictor = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, tgt_mask, src_key_padding_mask, tgt_key_padding_mask):
        # 对src和tgt进行编码
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        # 给src和tgt的token增加位置信息
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)
        # 将准备好的数据送给transformer
        out = self.transformer(src, tgt,
                               src_mask=None,
                               tgt_mask=tgt_mask,
                               memory_mask=None,
                               src_key_padding_mask=src_key_padding_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask,
                               memory_key_padding_mask=src_key_padding_mask)
        # 这里直接返回transformer的输出。因为训练和推理时的流程不一样，所以在模型外再进行线性层的预测。
        return out
