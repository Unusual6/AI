import numpy as np
import torch.nn as nn
import torch.optim

# 准备数据集
x_values = [i for i in range(11)]
y_values = [2 * i + 1 for i in range(11)]

x_train = np.array(x_values, dtype=np.float32)
x_train = x_train.reshape(-1, 1)
y_train = np.array(y_values, dtype=np.float32)
y_train = y_train.reshape(-1, 1)


class LinearRegressionModel(nn.Module):
    def __init__(self, input_d, outout_d):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_d, outout_d)

    def forward(self, x):
        out = self.linear(x)
        return out


input_d = 1
outout_d = 1
model = LinearRegressionModel(input_d, outout_d)

#使用GPU来训练
device =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 指定好参数和损失函数
epochs = 1000
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# 训练模型
for epoch in range(epochs):
    epoch += 1
    # 转numpy成tensor
    inputs = torch.from_numpy(x_train).to(device)
    labels = torch.from_numpy(y_train).to(device)

    # 梯度要清零每次迭代
    optimizer.zero_grad()

    # 向前传播
    outputs = model(inputs)

    # 计算损失
    loss = criterion(outputs, labels)

    # 反向传播
    loss.backward()

    # 更新权重参数
    optimizer.step()
    if epoch % 100 == 0:
        print('epoch {},loss {}'.format(epoch, loss.item()))

# 测试模型
predicted = model(torch.from_numpy(x_train).requires_grad_()).data.numpy()
print(predicted)

#模型的保存与读取
torch.save(model.state_dict(),'model.pkl')
model.load_state_dict(torch.load('model.pkl'))

