import torch
from torch import nn
from CopyTaskModel import CopyTaskModel
from generate_dataset import generate_random_batch, get_key_padding_mask
from args import max_length, pad, bos, eos, d_model, vocab_size, nhead, num_layers, dropout, batch_size, learning_rate, \
    epoch, log_step, device


def train(model, batch_size, max_length, learning_rate, log_step):
    # 设置model为训练模式
    model = model.train()
    # 定义损失函数
    criteria = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    # 定义学习率调度器
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, total_steps=epoch * 10,
                                                    anneal_strategy='linear', pct_start=0.1, div_factor=10
                                                    , final_div_factor=100)
    # 记录loss
    total_loss = 0
    # 开始训练
    for step in range(epoch):
        # 生成数据
        src, tgt, tgt_y, n_tokens = generate_random_batch(batch_size, max_length, device)
        # 生成causal mask下三角矩阵
        tgt_mask = torch.triu(
            torch.full((tgt.size()[-1], tgt.size()[-1]), float('-inf'), dtype=torch.float32, device=device), diagonal=1)
        """
        生成padding mask
        在Q@K.T/sqrt(d_k)得到的attention score矩阵中，将pad token对应的列全部置为-inf
        这样softmax之后就会变为0，之后对V的加权求和时就不会加入pad token的信息
        pad token对应的行不会被置为-inf，这是为了让pad token也能够接收其他token的信息，更新自己的词向量，得到训练
        src只需要mask掉pad token
        tgt还需要mask掉未来的token，tgt_key_padding_mask与tgt_mask会一起使用，合并为一个mask
        """
        src_key_padding_mask = get_key_padding_mask(src, device)
        tgt_key_padding_mask = get_key_padding_mask(tgt, device)
        # 清空梯度
        optimizer.zero_grad()
        # 进行transformer的计算
        out = model(src, tgt, tgt_mask, src_key_padding_mask, tgt_key_padding_mask)
        # 将结果送给最后的线性层进行预测
        # 用每一个词向量作为预测的输入，预测下一个词
        out = model.predictor(out)
        """
        计算损失。
        由于训练时我们的是对所有的输出都进行预测，所以需要对out进行reshape一下。
        取出每个batch的有效token（非pad），然后计算loss。
        """
        # 先取出第一个batch的有效token的预测logits和标签
        predict = out[0][:n_tokens[0], :]
        target = tgt_y[0][:n_tokens[0]]
        # 再拼接该batch其他的有效token的预测logits和标签
        for i in range(1, batch_size):
            predict = torch.cat([predict, out[i][:n_tokens[i], :]], dim=0)
            target = torch.cat([target, tgt_y[i][:n_tokens[i]]], dim=0)
        # 计算梯度
        loss = criteria(predict, target)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # 更新学习率
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
    src = torch.LongTensor([[bos, 4, 3, 4, 6, 5, 2, 5, 7, eos, pad, pad]]).to(device)
    # tgt从<bos>开始，看看能不能重新输出src中的值
    tgt = torch.LongTensor([[bos]]).to(device)
    # 生成mask
    tgt_mask = torch.triu(
        torch.full((tgt.size()[-1], tgt.size()[-1]), float('-inf'), dtype=torch.float32, device=device), diagonal=1)
    src_key_padding_mask = get_key_padding_mask(src, device)
    tgt_key_padding_mask = get_key_padding_mask(tgt, device)
    # 一个一个词预测，直到预测为<eos>，或者达到句子最大长度
    for i in range(max_length):
        # 进行transformer计算
        out = model(src, tgt, tgt_mask, src_key_padding_mask, tgt_key_padding_mask)
        # 预测结果，因为只需要看最后一个词，所以取`out[:, -1]`
        out = out[:, -1]
        predict = model.predictor(out)
        # 找出最大值的index
        y = torch.argmax(predict, dim=1)
        # 和之前的预测结果拼接到一起
        tgt = torch.concat([tgt, y.unsqueeze(0)], dim=1)
        # 更新mask
        tgt_mask = torch.triu(
            torch.full((tgt.size()[-1], tgt.size()[-1]), float('-inf'), dtype=torch.float32, device=device), diagonal=1)
        tgt_key_padding_mask = get_key_padding_mask(tgt, device)
        # 如果为<eos>，说明预测结束，跳出循环
        if y == eos:
            break
    # 打印预测结果
    print("src: ", src)
    print("tgt: ", tgt)


if __name__ == '__main__':
    model = CopyTaskModel(vocab_size, d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout).to(device)
    train(model, batch_size, max_length, learning_rate, log_step)
    evaluate(model, max_length)
