import numpy as np
import matplotlib.pyplot as plt
import pickle  # 导入pickle模块

# 生成时间序列数据
np.random.seed(42)
t = np.arange(0, 100, 0.1)
data = np.sin(t) + 0.1 * np.random.randn(len(t))

# 创建输入输出样本
def create_dataset(data, input_length=20, output_length=5):
    X, Y = [], []
    for i in range(len(data) - input_length - output_length +1):
        X.append(data[i:i+input_length])
        Y.append(data[i+input_length:i+input_length+output_length])
    return np.array(X), np.array(Y)

X, Y = create_dataset(data)
X = X.reshape(-1, 20, 1)  # (样本数, 时间步长, 输入特征)
Y = Y.reshape(-1, 5)       # (样本数, 输出维度)

# 划分训练集和测试集
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

# LSTM参数初始化
input_size = 1
hidden_size = 32
output_size = 5

# 尝试加载已保存的参数
try:
    with open("lstm_params.pkl", "rb") as f:
        W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_fc, b_fc = pickle.load(f)
    print("参数加载成功")
except FileNotFoundError:
    print("未找到已保存的参数，使用随机初始化参数")
    # 初始化门控权重
    W_xi = np.random.randn(hidden_size, input_size) * 0.01
    W_hi = np.random.randn(hidden_size, hidden_size) * 0.01
    b_i = np.zeros((hidden_size, 1))

    W_xf = np.random.randn(hidden_size, input_size) * 0.01
    W_hf = np.random.randn(hidden_size, hidden_size) * 0.01
    b_f = np.zeros((hidden_size, 1))

    W_xo = np.random.randn(hidden_size, input_size) * 0.01
    W_ho = np.random.randn(hidden_size, hidden_size) * 0.01
    b_o = np.zeros((hidden_size, 1))

    W_xc = np.random.randn(hidden_size, input_size) * 0.01
    W_hc = np.random.randn(hidden_size, hidden_size) * 0.01
    b_c = np.zeros((hidden_size, 1))

    # 输出层权重
    W_fc = np.random.randn(output_size, hidden_size) * 0.01
    b_fc = np.zeros((output_size, 1))

# 激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh_derivative(x):
    return 1 - x**2

# 前向传播
def lstm_forward(x_seq):
    h = np.zeros((hidden_size, 1))
    c = np.zeros((hidden_size, 1))
    cache = []
    
    for t in range(x_seq.shape[0]):
        x = x_seq[t].reshape(-1, 1)
        
        # 输入门
        i = sigmoid(W_xi @ x + W_hi @ h + b_i)
        # 遗忘门
        f = sigmoid(W_xf @ x + W_hf @ h + b_f)
        # 输出门
        o = sigmoid(W_xo @ x + W_ho @ h + b_o)
        # 新候选值
        c_tilde = np.tanh(W_xc @ x + W_hc @ h + b_c)
        # 更新细胞状态
        c = f * c + i * c_tilde
        # 更新隐藏状态
        h = o * np.tanh(c)
        
        cache.append((x, h.copy(), c.copy(), i, f, o, c_tilde))
    
    # 输出层
    output = W_fc @ h + b_fc
    return output, cache, h, c

# 训练参数
learning_rate = 0.1
epochs = 10

# 训练循环
losses = []
for epoch in range(epochs):
    total_loss = 0
    for i in range(len(X_train)):
        # 前向传播
        output, cache, h, c = lstm_forward(X_train[i])
        loss = np.mean((output.ravel() - Y_train[i])**2)
        total_loss += loss
        
        # 反向传播（简化版）
        d_output = 2 * (output.ravel() - Y_train[i]).reshape(-1, 1) / output_size
        d_W_fc = d_output @ cache[-1][1].T
        d_b_fc = d_output
        
        # 仅更新输出层权重
        W_fc -= learning_rate * d_W_fc
        b_fc -= learning_rate * d_b_fc
        
    avg_loss = total_loss / len(X_train)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# 保存训练好的参数
with open("lstm_params.pkl", "wb") as f:
    pickle.dump((W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_fc, b_fc), f)
    print("参数已保存")

# 预测测试集
test_sample = X_test[0]
true_values = Y_test[0]

predicted, _, _, _ = lstm_forward(test_sample)

# 绘制结果
plt.figure(figsize=(12, 6))
plt.plot(range(20), test_sample, label='Input Sequence', marker='o')
plt.plot(range(20, 25), true_values, label='True Output', marker='x')
plt.plot(range(20, 25), predicted.ravel(), label='Predicted Output', marker='^')
plt.title('Time Series Prediction with LSTM')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

# 绘制训练损失
plt.figure(figsize=(10, 5))
plt.plot(losses, label='Training Loss')
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()