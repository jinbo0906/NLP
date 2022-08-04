# %%
# code by Tae Hwan Jung @graykode
import torch
import torch.nn as nn
import torch.optim as optim


def make_batch():
    """
    input_batch:一组batch中前steps个单词的索引
    target_batch:一组batch中每句话待预测单词的索引
    这里是为了预测句子的最后一个词是啥
    """
    input_batch = []
    target_batch = []

    for sen in sentences:
        word = sen.split()  # space tokenizer
        input = [word_dict[n] for n in word[:-1]]  # create (1~n-1) as input
        target = word_dict[word[-1]]  # create (n) as target, We usually call this 'casual language model'

        input_batch.append(input)
        target_batch.append(target)

    return input_batch, target_batch


# Model
class NNLM(nn.Module):
    def __init__(self):
        super(NNLM, self).__init__()
        self.C = nn.Embedding(n_class, m)  # 词表
        # nn.Embedding：即给一个编号，嵌入层就能返回这个编号对应的嵌入向量，嵌入向量反映了各个编号代表的符号之间的语义关系。即输入为一个编号列表，输出为对应的符号嵌入向量列表。
        # index为（0,n_class-1）
        self.H = nn.Linear(n_step * m, n_hidden, bias=False)  # 输入层到隐藏层的权重
        self.d = nn.Parameter(torch.ones(n_hidden))  # 隐藏层的偏置
        # nn.Parameter: 作为nn.Module中的可训练参数使用
        self.U = nn.Linear(n_hidden, n_class, bias=False)  # 隐藏层到输出层的权重
        self.W = nn.Linear(n_step * m, n_class, bias=False)  # 输入层到输出层的权重
        self.b = nn.Parameter(torch.ones(n_class))  # 输出层的偏置

    def forward(self, X):
        """
        input: [batchsize, step]
        x: [batchsize, step*m]
        hidden_layer: [batchsize, hidden]
        output: [batchsize, nlen]
        """
        X = self.C(X)  # X : [batch_size, n_step, m]   获得一个batch的词向量的词表
        X = X.view(-1, n_step * m)  # [batch_size, n_step * m]
        tanh = torch.tanh(self.d + self.H(X))  # [batch_size, n_hidden]   获取隐藏层输出
        output = self.b + self.W(X) + self.U(tanh)  # [batch_size, n_class]   获得输出层输出
        return output


if __name__ == '__main__':
    n_step = 2  # len(text[0].split())-1  步长，即用前几个单词来预测下一个单词（这里是预测最后一个词是什么）
    n_hidden = 2  # 隐藏层的参数量（即节点数）
    m = 2  # 嵌入词向量的维度

    sentences = ["i like dog", "i love coffee", "i hate milk"]
    # 将句子中的词语提出来，并且去重
    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    # 对单词建立索引
    word_dict = {w: i for i, w in enumerate(word_list)}  # 单词-索引
    number_dict = {i: w for i, w in enumerate(word_list)}  # 索引-单词
    n_class = len(word_dict)  # 获得词典长度

    model = NNLM()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    input_batch, target_batch = make_batch()
    input_batch = torch.LongTensor(input_batch)
    target_batch = torch.LongTensor(target_batch)

    # Training
    for epoch in range(5000):
        optimizer.zero_grad()
        output = model(input_batch)

        # output : [batch_size, n_class], target_batch : [batch_size]
        loss = criterion(output, target_batch)
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    # Predict
    predict = model(input_batch).data.max(1, keepdim=True)[1]

    # Test
    print([sen.split()[:2] for sen in sentences], '->', [number_dict[n.item()] for n in predict.squeeze()])
