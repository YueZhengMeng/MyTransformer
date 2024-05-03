import torch

# 随机种子
seed = 42

# tokenizer参数
# 最大输入序列长度,包括<bos>和<eos>
# 长度不足max_length的序列用<pad>填充到max_length
max_length = 16
# 词表大小
vocab_size = 10
# <pad>的token id
pad = 0
# <bos>的token id
bos = 8
# <eos>的token id
eos = 9

# Transformer参数
# 词向量维度,论文默认是512
d_model = 256
# 多头注意力的头数,论文默认是8
nhead = 4
# 编码器和解码器的层数,论文默认是6
num_layers = 3
# dropout概率,论文默认是0.1
dropout = 0.1

# 训练参数
# batch_size
batch_size = 4
# 学习率,不要太大,不然模型会不收敛
learning_rate = 4e-5
# 训练的最大epoch,本项目中,一个epoch只有一个batch_size,相当于一个step
# 我也不知道为什么,pytorch版本的transformer收敛速度更慢
# 要达到较好的效果,OfficialTransformer需要训练6000步,而MyTransformer只需要3000步
# 进一步增加训练步数,两个模型的loss都还有下降的空间
# 但是对于复读机任务,当前训练步数已经足够了
# epoch = 6000  # OfficialTransformer使用
epoch = 3000  # MyTransformer使用
# 每log_step次打印一次loss
log_step = 100

# CUDA设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
