import random
import torch
from args import pad, bos, eos


def get_key_padding_mask(tokens, device):
    """
    用于key_padding_mask
    """
    key_padding_mask = torch.zeros(tokens.size())
    key_padding_mask[tokens == pad] = True
    return key_padding_mask.bool().to(device)


def generate_random_batch(batch_size, max_length, device):
    """
    生成训练数据集
    任务：输出一串随机数字序列，预测输出这串数字序列
    """
    src = []
    for i in range(batch_size):
        # 随机生成句子长度
        random_len = random.randint(1, max_length - 2)
        # 随机生成句子词汇，并在开头和结尾增加<bos>和<eos>
        random_nums = [bos] + [random.randint(1, 7) for _ in range(random_len)] + [eos]
        # 如果句子长度不足max_length，进行填充
        random_nums = random_nums + [pad] * (max_length - random_len - 2)
        src.append(random_nums)
    src = torch.LongTensor(src)
    # tgt不要<eos>, 即<eos>不作为输入,不预测之后的token
    # 将src中的<eos>替换为<pad>
    tgt = torch.where(src != eos, src, pad)[:, :-1]
    # tgt_y不要<bos>，即<bos>不作为预测的标签，只预测之后的token
    tgt_y = src[:, 1:]

    # 计算tgt_y，即要预测的有效token的数量
    # 这里的n_tokens指的是我们要预测的tgt_y中有多少有效的token，后面计算loss要用
    n_tokens = torch.sum(tgt_y != pad, dim=1)

    src = src.to(device)
    tgt = tgt.to(device)
    tgt_y = tgt_y.to(device)

    return src, tgt, tgt_y, n_tokens
