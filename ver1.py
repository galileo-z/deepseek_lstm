import numpy as np
import matplotlib.pyplot as plt

# 激活函数及导数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def dtanh(x):
    return 1 - np.tanh(x)**2

# LSTM参数初始化
class LSTMParams:
    def __init__(self, input_size, hidden_size):
        # 门控参数
        self.Wf = np.random.randn(hidden_size, hidden_size + input_size) * 0.01
        self.Wi = np.random.randn(hidden_size, hidden_size + input_size) * 0.01
        self.Wo = np.random.randn(hidden_size, hidden_size + input_size) * 0.01
        self.Wc = np.random.randn(hidden_size, hidden_size + input_size) * 0.01
        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        
        # 输出参数
        self.Wy = np.random.randn(1, hidden_size) * 0.01
        self.by = np.zeros((1, 1))

# LSTM模型
class LSTM:
    def __init__(self, input_size, hidden_size):
        self.params = LSTMParams(input_size, hidden_size)
        self.hidden_size = hidden_size
        
    def forward(self, x, h_prev, c_prev):
        concat = np.vstack((h_prev, x))
        
        # 门计算
        f = sigmoid(np.dot(self.params.Wf, concat) + self.params.bf)
        i = sigmoid(np.dot(self.params.Wi, concat) + self.params.bi)
        o = sigmoid(np.dot(self.params.Wo, concat) + self.params.bo)
        c_tilde = tanh(np.dot(self.params.Wc, concat) + self.params.bc)
        
        # 状态更新
        c = f * c_prev + i * c_tilde
        h = o * tanh(c)
        
        # 输出预测
        y = np.dot(self.params.Wy, h) + self.params.by
        
        return h, c, y, (concat, f, i, o, c_tilde, c_prev)
    
    def backward(self, dy, cache, dh_next, dc_next):
        concat, f, i, o, c_tilde, c_prev = cache
        
        # 输出层梯度
        dh = np.dot(self.params.Wy.T, dy) + dh_next
        dc = dc_next + dh * o * dtanh(tanh(c_tilde * i + f * c_prev))
        
        # 候选状态梯度
        dc_tilde = dc * i * dtanh(c_tilde)
        dWc = np.dot(dc_tilde, concat.T)
        dbc = dc_tilde
        
        # 输入门梯度
        di = dc * c_tilde * dsigmoid(i)
        dWi = np.dot(di, concat.T)
        dbi = di
        
        # 遗忘门梯度
        df = dc * c_prev * dsigmoid(f)
        dWf = np.dot(df, concat.T)
        dbf = df
        
        # 输出门梯度
        do = dh * tanh(c_tilde * i + f * c_prev) * dsigmoid(o)
        dWo = np.dot(do, concat.T)
        dbo = do
        
        # 输入梯度
        dconcat = (np.dot(self.params.Wf.T, df) +
                 np.dot(self.params.Wi.T, di) +
                 np.dot(self.params.Wo.T, do) +
                 np.dot(self.params.Wc.T, dc_tilde))
        
        dx = dconcat[self.hidden_size:, :]
        dh_prev = dconcat[:self.hidden_size, :]
        dc_prev = f * dc
        
        return dx, dh_prev, dc_prev, (dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc)

# 生成数据
time = np.arange(0, 100, 0.1)
data = np.sin(time) + 0.1 * np.random.randn(len(time))

# 创建序列数据
n_steps = 10
X = np.array([data[i:i+n_steps] for i in range(len(data)-n_steps)])
y = np.array([data[i+n_steps] for i in range(len(data)-n_steps)])

# 训练参数
hidden_size = 16
epochs = 50
lr = 0.01

# 初始化模型
model = LSTM(input_size=1, hidden_size=hidden_size)
losses = []

# 训练循环
for epoch in range(epochs):
    total_loss = 0
    for seq_idx in range(len(X)):
        # 初始化状态
        h = np.zeros((hidden_size, 1))
        c = np.zeros((hidden_size, 1))
        caches = []
        
        # 前向传播
        for t in range(n_steps):
            x = np.array([[X[seq_idx][t]]])
            h, c, y_pred, cache = model.forward(x, h, c)
            caches.append(cache)
        
        # 计算损失
        loss = (y_pred - y[seq_idx])**2
        total_loss += loss.item()
        
        # 反向传播
        dWy = (y_pred - y[seq_idx]) * h.T
        dby = (y_pred - y[seq_idx])
        dh_next = np.zeros_like(h)
        dc_next = np.zeros_like(c)
        grads = {
            'Wf': 0, 'Wi': 0, 'Wo': 0, 'Wc': 0,
            'bf': 0, 'bi': 0, 'bo': 0, 'bc': 0
        }
        
        # 时间步反向传播
        for t in reversed(range(n_steps)):
            dy = (y_pred - y[seq_idx]) if t == n_steps-1 else 0
            dx, dh_next, dc_next, grad = model.backward(
                dy, caches[t], dh_next, dc_next)
            
            # 累加梯度
            grads['Wf'] += grad[0]
            grads['Wi'] += grad[1]
            grads['Wo'] += grad[2]
            grads['Wc'] += grad[3]
            grads['bf'] += grad[4].sum(axis=1, keepdims=True)
            grads['bi'] += grad[5].sum(axis=1, keepdims=True)
            grads['bo'] += grad[6].sum(axis=1, keepdims=True)
            grads['bc'] += grad[7].sum(axis=1, keepdims=True)
        
        # 参数更新
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
    
    # 记录损失
    avg_loss = total_loss / len(X)
    losses.append(avg_loss)
    print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')

# 预测结果
predictions = []
for seq_idx in range(len(X)):
    h = np.zeros((hidden_size, 1))
    c = np.zeros((hidden_size, 1))
    for t in range(n_steps):
        x = np.array([[X[seq_idx][t]]])
        h, c, y_pred, _ = model.forward(x, h, c)
    predictions.append(y_pred[0][0])

# 可视化结果
plt.figure(figsize=(12, 6))
plt.plot(time[n_steps:], y, label='True')
plt.plot(time[n_steps:], predictions, label='Predicted', alpha=0.7)
plt.title('Time Series Prediction with LSTM')
plt.legend()
plt.show()

# 训练损失曲线
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.show()