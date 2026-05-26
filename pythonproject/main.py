import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import torch

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# ------------------------------
# 1. 读取与预处理
# ------------------------------
path = r""
data = pd.read_csv(path, sep=';', index_col=False)
data['当地时间 杭州市(机场)'] = pd.to_datetime(data['当地时间 杭州市(机场)'], format='%d.%m.%Y %H:%M')
data['month'] = data['当地时间 杭州市(机场)'].dt.month
data['day'] = data['当地时间 杭州市(机场)'].dt.day
data['hour'] = data['当地时间 杭州市(机场)'].dt.hour

X = data[['month', 'day', 'hour']].values
y = data['T'].values.astype(np.float32)

# ------------------------------
# 2. 划分训练集和测试集
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------
# 3. 标准化（仅基于训练集）
# ------------------------------
scaler = preprocessing.StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 转换为 torch 张量
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# ------------------------------
# 4. 构建神经网络
# ------------------------------
input_size = X_train.shape[1]  # 3
hidden_size = 32
output_size = 1
batch_size = 16

my_nn = torch.nn.Sequential(
    torch.nn.Linear(input_size, hidden_size),
    torch.nn.ReLU(),  # 使用 ReLU 替代 Sigmoid
    torch.nn.Linear(hidden_size, output_size)
)

cost = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(my_nn.parameters(), lr=0.001)

# ------------------------------
# 5. 训练网络
# ------------------------------
epochs = 300
train_losses = []
test_losses = []

for epoch in range(epochs):
    my_nn.train()
    batch_loss = []
    for start in range(0, len(X_train_t), batch_size):
        end = start + batch_size
        xx = X_train_t[start:end]
        yy = y_train_t[start:end]
        pred = my_nn(xx)
        loss = cost(pred, yy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.item())
    avg_train_loss = np.mean(batch_loss)
    train_losses.append(avg_train_loss)

    # 每 100 轮评估测试集
    if epoch % 100 == 0:
        my_nn.eval()
        with torch.no_grad():
            test_pred = my_nn(X_test_t)
            test_loss = cost(test_pred, y_test_t).item()
            test_losses.append(test_loss)
        print(f"Epoch {epoch:4d} | Train Loss: {avg_train_loss:.6f} | Test Loss: {test_loss:.6f}")

# ------------------------------
# 6. 最终评估
# ------------------------------
my_nn.eval()
with torch.no_grad():
    y_pred = my_nn(X_test_t).numpy().flatten()
r2 = r2_score(y_test, y_pred)
print(f"\n测试集 R² 分数: {r2: .4f}")

# ------------------------------
# 7. 预测未来时刻（需要标准化）
# ------------------------------
month = 12
day = 31
hour = 14
input_raw = np.array([[month, day, hour]], dtype=np.float32)
input_scaled = scaler.transform(input_raw)  # 关键：必须用训练时的 scaler
x_tensor = torch.tensor(input_scaled, dtype=torch.float32)
with torch.no_grad():
    pred_temp = my_nn(x_tensor).item()
print(f"\n预测 2024年12月31日下午14点 大气温度: {pred_temp:.2f} °C")