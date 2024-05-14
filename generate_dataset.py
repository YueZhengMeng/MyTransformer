import random
import torch
from args import pad, bos, eos


def get_key_padding_mask(tokens):
    """
    用于生成key_padding_mask
    """
    key_padding_mask = torch.where(tokens != pad, 0, float('-inf'))
    return key_padding_mask


def get_causal_mask(tokens):
    """
    用于生成tgt causal mask
    """
    causal_mask = torch.triu(torch.full((tokens.size()[-1], tokens.size()[-1]), float('-inf'), dtype=torch.float32),diagonal=1)
    return causal_mask


def generate_random_batch(batch_size, max_length):
    """
    生成训练数据集
    复读机任务：输出一串随机数字序列,预测输出这串数字序列
    """
    src = []
    # 生成batch_size个句子
    for i in range(batch_size):
        # 随机生成句子长度,长度为1到max_length-2,为<bos>和<eos>留出位置
        random_len = random.randint(1, max_length - 2)
        # 随机生成句子词汇,并在开头和结尾增加<bos>和<eos>
        random_nums = [bos] + [random.randint(1, 7) for _ in range(random_len)] + [eos]
        # 如果句子长度不足max_length,用<pad>进行填充
        random_nums = random_nums + [pad] * (max_length - random_len - 2)
        src.append(random_nums)

    # 将src转换为LongTensor
    src = torch.LongTensor(src)

    # tgt不要<eos>, 即<eos>不作为预测器输入,不预测之后的token
    # 将src中的<eos>替换为<pad>后,去掉最后一个token,作为tgt
    tgt = torch.where(src != eos, src, pad)[:, :-1]

    # tgt_y不要<bos>,即<bos>不作为预测的标签,只预测之后的token
    # 将src中的<bos>,即第一个token去掉后,作为tgt_y
    tgt_y = src[:, 1:]

    return src, tgt, tgt_y
