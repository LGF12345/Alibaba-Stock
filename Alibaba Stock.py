import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
#中文处理
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

# 1. 数据加载和预处理
# parse_dates=['Date'] 将日期列解析为 datetime 类型，便于后续处理
data = pd.read_csv('./Ali_Baba_Stock_Data.csv', parse_dates=['Date'])
# 按日期排序，确保数据按时间顺序排列
data = data.sort_values('Date')
#由于我的数据存在非连续性（即股价数据之间的时间间隔不固定，可能是1天，也可能是几天），这一点会对时间序列预测任务带来一定的影响。
#将seq_length直接设置为60（或其他固定值）可能会引入一些问题
#解决办法：
# 使用时间嵌入（Time Embedding）
# 方法：在模型中引入时间信息（例如日期差或时间戳），作为额外的输入特征，让模型显式地学习时间间隔的影响。
    # 实现步骤：
    # 计算每条记录与前一条记录的日期差，作为一个特征。
    # 将日期差归一化后，与收盘价一起作为输入特征（input_size=2）。
    # 调整模型的输入维度。

# 提取收盘价，并计算日期差
# 'Close'列为预测目标，提取为numpy数组
close_prices = data['Close'].values
# 计算每条记录与前一条记录的日期差（单位：天），用作时间嵌入特征
# fillna(0) 处理第一条记录的日期差（无前一条记录）
data['DateDiff'] = data['Date'].diff().dt.days.fillna(0)
date_diff = data['DateDiff'].values

# 归一化收盘价和日期差
# 使用 MinMaxScaler 将数据归一化到 [0, 1] 区间，适合神经网络输入
# 分别对收盘价和日期差进行归一化，避免量纲差异
scaler_close = MinMaxScaler(feature_range=(0, 1))
scaler_diff = MinMaxScaler(feature_range=(0, 1))
close_prices_scaled = scaler_close.fit_transform(close_prices.reshape(-1, 1))
date_diff_scaled = scaler_diff.fit_transform(date_diff.reshape(-1, 1))


# 2. 构建时间序列数据
# 定义函数，将数据构建为时间序列，包含收盘价和日期差两个特征
# 输入：归一化后的收盘价、日期差、序列长度
# 输出：X为序列（每个样本为seq_length长的输入），y为目标值（下一天的收盘价）
def create_sequences_with_time(close_data, diff_data, seq_length):
    X, y = [], []
    # 从第seq_length条记录开始，构建序列
    for i in range(seq_length, len(close_data)):
        # 提取前seq_length条记录的收盘价和日期差，stack成 (seq_length, 2) 的形状
        X.append(np.stack([close_data[i - seq_length:i, 0], diff_data[i - seq_length:i, 0]], axis=1))
        # 目标值为第i条记录的收盘价
        y.append(close_data[i, 0])
    return np.array(X), np.array(y)


# 设置序列长度为15（记录条数），后续通过时间嵌入解决非连续性问题
seq_length = 15
X, y = create_sequences_with_time(close_prices_scaled, date_diff_scaled, seq_length)

# 3. 划分数据集
# 将数据集划分为训练集（60%）、验证集（20%）、测试集（20%）
train_size = int(len(X) * 0.6)
val_size = int(len(X) * 0.2)
test_size = len(X) - train_size - val_size

# 分割输入和目标值
X_train, X_val, X_test = X[:train_size], X[train_size:train_size + val_size], X[train_size + val_size:]
y_train, y_val, y_test = y[:train_size], y[train_size:train_size + val_size], y[train_size + val_size:]

# 将数据转换为PyTorch张量
# 输入形状为 (样本数, seq_length, 2)，2表示特征数（收盘价+日期差）
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).float()
X_val_tensor = torch.from_numpy(X_val).float()
y_val_tensor = torch.from_numpy(y_val).float()
X_test_tensor = torch.from_numpy(X_test).float()
y_test_tensor = torch.from_numpy(y_test).float()

# 创建DataLoader，用于批量训练和验证
# shuffle=False 确保时间序列数据按顺序加载
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


# 4. 定义LSTM模型
# 定义LSTM模型类，输入维度为2（收盘价+日期差），输出维度为1（预测下一天的收盘价）
class LSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=50, num_layers=3, output_size=1):
        super(LSTMModel, self).__init__()
        # LSTM层：输入维度为2，隐藏层维度为hidden_size，层数为num_layers
        # dropout=0.2 防止过拟合（仅在num_layers>1时生效）
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        # 全连接层：将LSTM最后一个时间步的隐藏状态映射到输出维度
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x的形状为 (batch_size, seq_length, input_size)
        # out的形状为 (batch_size, seq_length, hidden_size)
        out, _ = self.lstm(x)
        # 取最后一个时间步的输出，映射到预测值
        out = self.fc(out[:, -1, :])
        return out


# 5. 设置hidden_sizes并训练模型
#将hidden_sizes设置为列表，对每个值训练模型，评估验证集上的性能（RMSE和MAE），
#并通过训练误差和验证误差判断是否过拟合或欠拟合。

#设置hidden_sizes，分别训练模型并评估性能
hidden_sizes = [30,35,40,45]
num_epochs = 100
results = {}  # 存储每个hidden_size的评估结果
models = {}  # 存储每个hidden_size的模型

# 遍历每个hidden_size
for hidden_size in hidden_sizes:
    print(f"正在训练hidden_size = {hidden_size}")

    # 初始化模型、损失函数和优化器
    #由于数据不连续，使用时间嵌入（Time Embedding），引入日期差作为额外特征，调整input_size=2。
    model = LSTMModel(input_size=2, hidden_size=hidden_size, num_layers=3, output_size=1)
    loss_function = nn.MSELoss()  # 使用均方误差损失
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam优化器，学习率0.001

    # 记录训练和验证损失，用于绘制损失曲线
    train_losses = []
    val_losses = []

    # 训练循环
    #epochs表示整个训练集完整地喂给 LSTM 多少次，也就是多少轮，如果损失（loss）在例如20-50 轮后趋于平稳，说明训练基本收敛
    for epoch in range(num_epochs):
        # 训练模式
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()  # 清空梯度
            y_pred = model(X_batch)  # 前向传播
            loss = loss_function(y_pred, y_batch.unsqueeze(1))  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            train_loss += loss.item() * X_batch.size(0)  # 累计批量损失
        train_loss /= len(train_loader.dataset)  # 计算平均训练损失
        train_losses.append(train_loss)

        # 验证模式
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred = model(X_batch)
                loss = loss_function(y_pred, y_batch.unsqueeze(1))
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)  # 计算平均验证损失
        val_losses.append(val_loss)

        # 每10个周期打印一次损失
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

    #绘制损失曲线（loss curve），观察训练和验证损失是否收敛，判断训练是否合适。
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'损失曲线——hidden_size = {hidden_size}')
    plt.legend()
    plt.show()

    # 改进后的过拟合/欠拟合判断
    # 基于最终损失值和验证损失趋势判断
    final_train_loss = train_losses[-1]
    final_val_loss = val_losses[-1]
    # 计算验证损失最后10个周期的趋势
    val_loss_trend = val_losses[-1] - val_losses[-10] if len(val_losses) > 10 else 0
    status = "Unknown"

#过拟合的判断应该结合损失曲线的趋势（例如验证损失是否上升）、测试集的性能（RMSE/MAE是否异常高），
#以及模型是否在训练集上表现过好而在验证/测试集上表现很差。
#改进过拟合/欠拟合的判断逻辑，结合以下几点：
# 损失曲线的趋势：如果验证损失在训练后期上升，而训练损失持续下降，可能是过拟合。
# 测试集性能：如果测试集的RMSE/MAE显著高于验证集，可能是过拟合。
# 训练/验证损失的绝对值：如果训练和验证损失都很高（例如>0.1），可能是欠拟合。

    # 判断过拟合：训练损失远小于验证损失，且验证损失有上升趋势
    if final_train_loss < final_val_loss and (
            final_val_loss - final_train_loss) / final_train_loss > 0.2 and val_loss_trend > 0:
        status = "Overfitting"
        print(f"可能过拟合了！当hidden_size = {hidden_size}")
    # 判断欠拟合：训练和验证损失都较高，且验证损失几乎不下降
    elif final_train_loss > 0.05 and final_val_loss > 0.05 and abs(val_loss_trend) < 0.001:
        status = "Underfitting"
        print(f"可能欠拟合了！当hidden_size = {hidden_size}")
    # 其他情况：训练可能合适
    else:
        status = "Appropriate"
        print(f"OK！训练刚刚好！当hidden_size = {hidden_size}")

    print(f"Final Train Loss: {final_train_loss:.6f}, Final Val Loss: {final_val_loss:.6f}")

    # 在测试集上评估模型性能
    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test_tensor)
        y_pred_test = y_pred_test.numpy()
        y_pred_test = scaler_close.inverse_transform(y_pred_test)  # 反归一化
        y_test_unscaled = scaler_close.inverse_transform(y_test.reshape(-1, 1))

    # 计算测试集的RMSE和MAE
    rmse = np.sqrt(mean_squared_error(y_test_unscaled, y_pred_test))
    mae = mean_absolute_error(y_test_unscaled, y_pred_test)
    print(f"Test RMSE: {rmse:.2f}, Test MAE: {mae:.2f}")

    # 保存结果和模型
    results[hidden_size] = {
        'rmse': rmse,
        'mae': mae,
        'train_loss': final_train_loss,
        'val_loss': final_val_loss,
        'status': status
    }
    models[hidden_size] = model

# 6. 选择最佳hidden_size
# 优先选择状态为“训练合适”的hidden_size，并从中选择RMSE最低的一个
appropriate_hidden_sizes = [hs for hs in hidden_sizes if results[hs]['status'] == "Appropriate"]
if appropriate_hidden_sizes:
    best_hidden_size = min(appropriate_hidden_sizes, key=lambda k: results[k]['rmse'])
    print(f"\nBest hidden_size (Training Appropriate) based on RMSE: {best_hidden_size}")
    print(f"Best Test RMSE: {results[best_hidden_size]['rmse']:.2f}, Test MAE: {results[best_hidden_size]['mae']:.2f}")
else:
    # 如果没有“训练合适”的hidden_size，则基于RMSE选择，并给出提示
    print("\nNo hidden_size with 'Training Appropriate' status found. Choosing based on RMSE alone.")
    best_hidden_size = min(results, key=lambda k: results[k]['rmse'])
    print(f"Best hidden_size based on RMSE: {best_hidden_size}")
    print(f"Best Test RMSE: {results[best_hidden_size]['rmse']:.2f}, Test MAE: {results[best_hidden_size]['mae']:.2f}")

# 7. 使用最佳hidden_size的模型进行预测并绘制结果
# 从保存的模型中取出最佳hidden_size对应的模型
model = models[best_hidden_size]
model.eval()
with torch.no_grad():
    y_pred_test = model(X_test_tensor)
    y_pred_test = y_pred_test.numpy()
    y_pred_test = scaler_close.inverse_transform(y_pred_test)  # 反归一化预测值
    y_test_unscaled = scaler_close.inverse_transform(y_test.reshape(-1, 1))  # 反归一化真实值

# 绘制预测结果与实际值的对比图
test_dates = data['Date'][train_size + val_size + seq_length:]
plt.figure(figsize=(12, 6))
plt.plot(test_dates, y_test_unscaled, label='实际股价')
plt.plot(test_dates, y_pred_test, label='预测股价')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title(f'阿里巴巴股价预测图 (hidden_size = {best_hidden_size})')
plt.legend()
plt.xticks(rotation=45)
plt.show()