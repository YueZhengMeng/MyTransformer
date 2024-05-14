import random
import numpy as np
import torch
from torch import nn
from OfficialTransformer import OfficialTransformer
from MyTransformer import MyTransformer
from generate_dataset import generate_random_batch, get_key_padding_mask, get_causal_mask
from args import seed, max_length, pad, bos, eos, d_model, vocab_size, nhead, num_layers, dropout, batch_size, \
    learning_rate, epoch, log_step, device


def seed_everything(seed):
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(model, batch_size, max_length, learning_rate, log_step):
    # 设置model为训练模式
    model = model.train()
    # 定义损失函数,这样可以设置ignore_index=pad,实现不计算pad token的loss
    # 但这种方法不够灵活,不能mask掉其他token,后面会使用更加灵活的mask方法
    criteria = nn.CrossEntropyLoss()
    # 定义优化器,学习率不要太高,否则模型会不收敛
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    # 定义学习率调节器
    # 前10%的total_steps预热(warm up),学习率从 max_lr/10 到 max_lr
    # 后面90%的total_steps学习率从 max_lr 线性(linear)衰减到 max_lr/10
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, total_steps=epoch * 1,
                                                    anneal_strategy='linear', pct_start=0.1, div_factor=10
                                                    , final_div_factor=10)
    # 记录loss
    total_loss = 0
    # 开始训练,这里每个epoch只训练一个batch,只有一步
    for step in range(epoch):
        # 生成数据
        src, tgt, tgt_y = generate_random_batch(batch_size, max_length)
        # 生成causal mask下三角矩阵
        tgt_mask = get_causal_mask(tgt)
        # 生成key_padding_mask
        src_key_padding_mask = get_key_padding_mask(src)
        tgt_key_padding_mask = get_key_padding_mask(tgt)
        # 清空梯度
        optimizer.zero_grad()
        # 数据送入cuda
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_y = tgt_y.to(device)
        tgt_mask = tgt_mask.to(device)
        src_key_padding_mask = src_key_padding_mask.to(device)
        tgt_key_padding_mask = tgt_key_padding_mask.to(device)
        # 进行transformer的计算
        out = model(src, tgt, tgt_mask, src_key_padding_mask, tgt_key_padding_mask)
        # 将结果送给最后的线性层进行预测
        # 用每一个词向量作为预测的输入,预测下一个词
        # 由于nn.CrossEntropyLoss()中自带softmax,所以这里不需要再进行softmax
        out = model.predictor(out)
        # 只计算有效token的loss,注意这里mask是要保留的token
        loss_mask = torch.where(tgt_key_padding_mask.view(-1) == 0, True, False)
        # 转换为nn.CrossEntropyLoss()的输入格式
        predict = out.view(-1, vocab_size)[loss_mask]
        target = tgt_y.view(-1)[loss_mask]
        # 计算梯度
        loss = criteria(predict, target)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # 更新学习率调节器
        scheduler.step()
        # 记录loss
        total_loss += loss
        # 每log_step次打印一下loss
        if step != 0 and step % log_step == 0:
            print("Step {}, aver_total_loss: {}".format(step, total_loss / log_step))
            total_loss = 0


def evaluate(model, max_length):
    # 设置model为评估模式
    model = model.eval()
    # 随便定义一个src
    src = torch.LongTensor([[bos, 4, 3, 4, 6, 5, 2, 5, 7, eos, pad, pad]])
    # tgt从<bos>开始,看看能不能重新输出src中的值
    tgt = torch.LongTensor([[bos]])
    # 生成mask
    tgt_mask = get_causal_mask(tgt)
    src_key_padding_mask = get_key_padding_mask(src)
    # 数据送入cuda
    src = src.to(device)
    tgt = tgt.to(device)
    tgt_mask = tgt_mask.to(device)
    src_key_padding_mask = src_key_padding_mask.to(device)
    # 一个一个词预测,直到预测为<eos>,或者达到句子最大长度
    for i in range(max_length):
        # 进行transformer计算
        out = model(src, tgt, tgt_mask, src_key_padding_mask, None)
        # 预测结果,因为只需要看最后一个词,所以取out[:, -1]
        out = out[:, -1]
        predict = model.predictor(out)
        # 找出最大值的index
        y = torch.argmax(predict, dim=1)
        # 和之前的预测结果拼接到一起
        tgt = torch.concat([tgt, y.unsqueeze(0)], dim=1)
        # 更新mask
        tgt_mask = get_causal_mask(tgt)
        tgt_mask = tgt_mask.to(device)
        # 如果为<eos>,说明预测结束,跳出循环
        if y == eos:
            break
    # 打印预测结果
    print("src: ", src)
    print("tgt: ", tgt)


if __name__ == '__main__':
    seed_everything(seed)
    # 我也不知道为什么,pytorch版本的transformer收敛速度更慢
    # 要达到较好的效果,OfficialTransformer需要训练6000步,而MyTransformer只需要3000步
    # 切换模型的时候,记得注释掉另一个模型的创建,并且在args.py中修改epoch数量
    # model = OfficialTransformer(vocab_size, max_length, d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout).to(device)
    model = MyTransformer(vocab_size, max_length, d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout).to(device)
    train(model, batch_size, max_length, learning_rate, log_step)
    evaluate(model, max_length)
