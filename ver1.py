import numpy as np
import matplotlib.pyplot as plt

# 定义激活函数及其导数
def sigmoid(x):
    """
    实现 sigmoid 激活函数: f(x) = 1 / (1 + e^(-x))
    用于将输入压缩到 0-1 之间
    """
    return 1 / (1 + np.exp(-x))

def dsigmoid(x):
    """
    sigmoid 函数的导数: f'(x) = f(x) * (1 - f(x))
    用于反向传播时计算梯度
    """
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    """
    双曲正切激活函数
    将输入压缩到 -1 到 1 之间
    """
    return np.tanh(x)

def dtanh(x):
    """
    tanh 函数的导数: f'(x) = 1 - tanh^2(x)
    用于反向传播时计算梯度
    """
    return 1 - np.tanh(x)**2

class LSTMParams:
    """
    LSTM 模型的参数类，存储和初始化所有权重和偏置
    """
    def __init__(self, input_size, hidden_size):
        # 初始化门控单元的权重矩阵，使用随机小数值
        # hidden_size: 隐藏层大小
        # input_size: 输入维度
        # 权重矩阵形状为 (hidden_size, hidden_size + input_size)
        self.Wf = np.random.randn(hidden_size, hidden_size + input_size) * 0.01  # 遗忘门权重
        self.Wi = np.random.randn(hidden_size, hidden_size + input_size) * 0.01  # 输入门权重
        self.Wo = np.random.randn(hidden_size, hidden_size + input_size) * 0.01  # 输出门权重
        self.Wc = np.random.randn(hidden_size, hidden_size + input_size) * 0.01  # 候选状态权重
        
        # 初始化偏置向量为零向量
        self.bf = np.zeros((hidden_size, 1))  # 遗忘门偏置
        self.bi = np.zeros((hidden_size, 1))  # 输入门偏置
        self.bo = np.zeros((hidden_size, 1))  # 输出门偏置
        self.bc = np.zeros((hidden_size, 1))  # 候选状态偏置
        
        # 初始化输出层参数
        self.Wy = np.random.randn(1, hidden_size) * 0.01  # 输出层权重
        self.by = np.zeros((1, 1))  # 输出层偏置

class LSTM:
    """
    LSTM 模型的主类，实现前向传播和反向传播
    """
    def __init__(self, input_size, hidden_size):
        """
        初始化 LSTM 模型
        input_size: 输入维度
        hidden_size: 隐藏层大小
        """
        self.params = LSTMParams(input_size, hidden_size)
        self.hidden_size = hidden_size
        
    def forward(self, x, h_prev, c_prev):
        """
        LSTM 前向传播
        x: 当前输入
        h_prev: 上一时刻隐藏状态
        c_prev: 上一时刻单元状态
        """
        # 将当前输入和上一时刻隐藏状态拼接
        concat = np.vstack((h_prev, x))
        
        # 计算三个门和候选状态
        f = sigmoid(np.dot(self.params.Wf, concat) + self.params.bf)  # 遗忘门
        i = sigmoid(np.dot(self.params.Wi, concat) + self.params.bi)  # 输入门
        o = sigmoid(np.dot(self.params.Wo, concat) + self.params.bo)  # 输出门
        c_tilde = tanh(np.dot(self.params.Wc, concat) + self.params.bc)  # 候选状态
        
        # 更新单元状态和隐藏状态
        c = f * c_prev + i * c_tilde  # 新的单元状态
        h = o * tanh(c)  # 新的隐藏状态
        
        # 计算输出预测
        y = np.dot(self.params.Wy, h) + self.params.by
        
        # 返回结果和中间状态（用于反向传播）
        return h, c, y, (concat, f, i, o, c_tilde, c_prev)
    
    def backward(self, dy, cache, dh_next, dc_next):
        """
        LSTM 反向传播
        dy: 输出梯度
        cache: 前向传播的中间状态
        dh_next: 下一时刻的隐藏状态梯度
        dc_next: 下一时刻的单元状态梯度
        """
        concat, f, i, o, c_tilde, c_prev = cache
        
        # 计算隐藏状态和单元状态的梯度
        dh = np.dot(self.params.Wy.T, dy) + dh_next
        dc = dc_next + dh * o * dtanh(tanh(c_tilde * i + f * c_prev))
        
        # 计算候选状态的梯度
        dc_tilde = dc * i * dtanh(c_tilde)
        dWc = np.dot(dc_tilde, concat.T)
        dbc = dc_tilde
        
        # 计算输入门的梯度
        di = dc * c_tilde * dsigmoid(i)
        dWi = np.dot(di, concat.T)
        dbi = di
        
        # 计算遗忘门的梯度
        df = dc * c_prev * dsigmoid(f)
        dWf = np.dot(df, concat.T)
        dbf = df
        
        # 计算输出门的梯度
        do = dh * tanh(c_tilde * i + f * c_prev) * dsigmoid(o)
        dWo = np.dot(do, concat.T)
        dbo = do
        
        # 计算输入梯度
        dconcat = (np.dot(self.params.Wf.T, df) +
                 np.dot(self.params.Wi.T, di) +
                 np.dot(self.params.Wo.T, do) +
                 np.dot(self.params.Wc.T, dc_tilde))
        
        # 分离输入和隐藏状态的梯度
        dx = dconcat[self.hidden_size:, :]  # 输入梯度
        dh_prev = dconcat[:self.hidden_size, :]  # 前一时刻隐藏状态梯度
        dc_prev = f * dc  # 前一时刻单元状态梯度
        
        return dx, dh_prev, dc_prev, (dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc)

# 生成训练数据
time = np.arange(0, 100, 0.1)  # 生成时间序列
data = np.sin(time) + 0.1 * np.random.randn(len(time))  # 生成带噪声的正弦波数据

# 创建序列数据
n_steps = 10  # 序列长度
# 创建输入序列 X 和目标值 y
X = np.array([data[i:i+n_steps] for i in range(len(data)-n_steps)])
y = np.array([data[i+n_steps] for i in range(len(data)-n_steps)])

# 设置训练参数
hidden_size = 16  # 隐藏层大小
epochs = 50  # 训练轮数
lr = 0.01  # 学习率

# 初始化 LSTM 模型
model = LSTM(input_size=1, hidden_size=hidden_size)
losses = []  # 存储训练损失

# 训练模型
for epoch in range(epochs):
    total_loss = 0
    for seq_idx in range(len(X)):
        # 初始化每个序列的状态
        h = np.zeros((hidden_size, 1))  # 初始隐藏状态
        c = np.zeros((hidden_size, 1))  # 初始单元状态
        caches = []  # 存储中间状态
        
        # 对序列进行前向传播
        for t in range(n_steps):
            x = np.array([[X[seq_idx][t]]])  # 获取当前时间步的输入
            h, c, y_pred, cache = model.forward(x, h, c)
            caches.append(cache)
        
        # 计算均方误差损失
        loss = (y_pred - y[seq_idx])**2
        total_loss += loss.item()
        
        # 计算输出层梯度
        dWy = (y_pred - y[seq_idx]) * h.T
        dby = (y_pred - y[seq_idx])
        dh_next = np.zeros_like(h)
        dc_next = np.zeros_like(c)
        
        # 初始化梯度累加器
        grads = {
            'Wf': 0, 'Wi': 0, 'Wo': 0, 'Wc': 0,
            'bf': 0, 'bi': 0, 'bo': 0, 'bc': 0
        }
        
        # 反向传播through time (BPTT)
        for t in reversed(range(n_steps)):
            # 只在序列最后一步计算输出梯度
            dy = (y_pred - y[seq_idx]) if t == n_steps-1 else 0
            dx, dh_next, dc_next, grad = model.backward(
                dy, caches[t], dh_next, dc_next)
            
            # 累加各个时间步的梯度
            grads['Wf'] += grad[0]
            grads['Wi'] += grad[1]
            grads['Wo'] += grad[2]
            grads['Wc'] += grad[3]
            grads['bf'] += grad[4].sum(axis=1, keepdims=True)
            grads['bi'] += grad[5].sum(axis=1, keepdims=True)
            grads['bo'] += grad[6].sum(axis=1, keepdims=True)
            grads['bc'] += grad[7].sum(axis=1, keepdims=True)
        
        # 使用梯度更新参数
        model.params.Wf -= lr * grads['Wf']
        model.params.Wi -= lr * grads['Wi']
        model.params.Wo -= lr * grads['Wo']
        model.params.Wc -= lr * grads['Wc']
        model.params.bf -= lr * grads['bf']
        model.params.bi -= lr * grads['bi']
        model.params.bo -= lr * grads['bo']
        model.params.bc -= lr * grads['bc']
        model.params.Wy -= lr * dWy
        model.params.by -= lr * dby
    
    # 计算并记录平均损失
    avg_loss = total_loss / len(X)
    losses.append(avg_loss)
    print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')

# 使用训练好的模型进行预测
predictions = []
for seq_idx in range(len(X)):
    # 初始化状态
    h = np.zeros((hidden_size, 1))
    c = np.zeros((hidden_size, 1))
    # 对每个序列进行预测
    for t in range(n_steps):
        x = np.array([[X[seq_idx][t]]])
        h, c, y_pred, _ = model.forward(x, h, c)
    predictions.append(y_pred[0][0])

# 绘制预测结果与真实值的对比图
plt.figure(figsize=(12, 6))
plt.plot(time[n_steps:], y, label='True')
plt.plot(time[n_steps:], predictions, label='Predicted', alpha=0.7)
plt.title('Time Series Prediction with LSTM')
plt.legend()
plt.show()

# 绘制训练过程中的损失曲线
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.show()