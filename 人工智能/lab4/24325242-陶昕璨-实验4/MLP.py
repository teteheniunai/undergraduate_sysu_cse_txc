import argparse
import io
import sys
import time
import tracemalloc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ==========================================
# 1. Data preprocessing
# ==========================================
class StandardScalerManual:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0) # 均值
        self.std = np.std(X, axis=0) # 标准差
        self.std[self.std == 0] = 1e-8 # 防止除0
        return self

    def transform(self, X): # 标准化
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_scaled): # 反标准化
        return X_scaled * self.std + self.mean


# ==========================================
# 2. Loss functions
# ==========================================
class LossFunctions:
    @staticmethod
    def mse(y_true, y_pred): # 均方差损失
        loss = np.mean((y_pred - y_true) ** 2)
        grad = 2.0 * (y_pred - y_true) / y_true.shape[0]
        return loss, grad

    @staticmethod
    def huber(y_true, y_pred, delta=1.0): # Huber损失
        error = y_pred - y_true
        abs_error = np.abs(error)
        mask = abs_error <= delta

        quadratic = 0.5 * (error ** 2)
        linear = delta * (abs_error - 0.5 * delta)
        loss = np.mean(np.where(mask, quadratic, linear)) # np.where(条件，满足时取值，不满足时取值)
        grad = np.where(mask, error, delta * np.sign(error)) / y_true.shape[0]
        return loss, grad

    @staticmethod
    def get_loss(loss_name, delta=1.0):
        loss_name = loss_name.lower()
        if loss_name == "mse":
            return LossFunctions.mse
        if loss_name == "huber":
            return lambda y_true, y_pred: LossFunctions.huber(y_true, y_pred, delta=delta)
        raise ValueError(f"Unsupported loss function: {loss_name}")


# ==========================================
# 3. MLP + SGD with momentum
# ==========================================
class RawMLP:
    # 网络初始化
    def __init__(self, layer_sizes):
        self.num_layers = len(layer_sizes)
        self.params = {} # 保存权值和偏置
        self.grads = {} # 保存导数
        self.velocities = {} # 保存momentum速度

        np.random.seed(42)
        for i in range(1, self.num_layers):
            weight_key = f"W{i}"
            bias_key = f"b{i}"
            # He初始化
            self.params[weight_key] = np.random.randn(layer_sizes[i - 1], layer_sizes[i]) * np.sqrt(
                2.0 / layer_sizes[i - 1]
            )
            self.params[bias_key] = np.zeros((1, layer_sizes[i]))
            self.velocities[weight_key] = np.zeros_like(self.params[weight_key])
            self.velocities[bias_key] = np.zeros_like(self.params[bias_key])

    # 激活函数
    def relu(self, Z):
        return np.maximum(0, Z)
    # 激活函数导数
    def relu_derivative(self, Z):
        return (Z > 0).astype(float)

    # 前向传播
    def forward(self, X):
        self.cache = {"A0": X} # 便于反向传播
        A = X

        # 隐藏层计算
        for i in range(1, self.num_layers - 1):
            Z = np.dot(A, self.params[f"W{i}"]) + self.params[f"b{i}"]
            A = self.relu(Z)
            self.cache[f"Z{i}"] = Z
            self.cache[f"A{i}"] = A
        # 输出层计算
        last = self.num_layers - 1
        Z = np.dot(A, self.params[f"W{last}"]) + self.params[f"b{last}"]

        self.cache[f"Z{last}"] = Z
        self.cache[f"A{last}"] = Z
        return Z
    
    # 反向传播
    def backward(self, d_out):
        last = self.num_layers - 1
        dA = d_out

        for i in range(last, 0, -1):
            if i == last:
                dZ = dA
            else:
                dZ = dA * self.relu_derivative(self.cache[f"Z{i}"])

            A_prev = self.cache[f"A{i - 1}"]
            self.grads[f"W{i}"] = np.dot(A_prev.T, dZ)
            self.grads[f"b{i}"] = np.sum(dZ, axis=0, keepdims=True)
            dA = np.dot(dZ, self.params[f"W{i}"].T)

    # L2正则化，防止过拟合
    def compute_l2_penalty(self, l2_lambda=0.0):
        if l2_lambda <= 0:
            return 0.0

        penalty = 0.0
        for i in range(1, self.num_layers): # 2/λ​∑W2
            penalty += np.sum(self.params[f"W{i}"] ** 2)
        return 0.5 * l2_lambda * penalty

    def update_params_gd(self, current_lr, momentum=0.9, l2_lambda=0.0):
        
        for i in range(1, self.num_layers):
            weight_key = f"W{i}"
            bias_key = f"b{i}"

            weight_grad = self.grads[weight_key]
            if l2_lambda > 0:
                weight_grad = weight_grad + l2_lambda * self.params[weight_key]
            
            # 先计算momentum速度
            self.velocities[weight_key] = momentum * self.velocities[weight_key] + current_lr * weight_grad
            # # W=W−ηg
            self.params[weight_key] -= self.velocities[weight_key]

            bias_grad = self.grads[bias_key]
            self.velocities[bias_key] = momentum * self.velocities[bias_key] + current_lr * bias_grad
            self.params[bias_key] -= self.velocities[bias_key]

# ==========================================
# 4. Metrics：回归模型评估工具
# ==========================================
class Metrics:
    @staticmethod
    def regression_report(y_true, y_pred):
        mse = np.mean((y_true - y_pred) ** 2) # 均方误差
        rmse = np.sqrt(mse) # 均方根误差
        mae = np.mean(np.abs(y_true - y_pred)) # 平均绝对误差 MAE
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) 
        ss_res = np.sum((y_true - y_pred) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8)) # 决定系数R²：衡量模型拟合程度，范围0~1，越接近1拟合越好
        return mse, rmse, mae, r2

# ==========================================
# 5. Early stopping：防止过拟合，节省训练时间，保存最佳模型
# ==========================================
class EarlyStopping:
    # 允许20各epoch没有提升
    # 提升幅度必须超过0.0001
    def __init__(self, patience=20, min_delta=1e-4): 
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -np.inf
        self.best_epoch = 0
        self.wait = 0 # 记录连续多少轮没有提升
        self.best_params = None
        self.should_stop = False

    def update(self, score, epoch, model):
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.best_epoch = epoch
            self.wait = 0
            self.best_params = {key: value.copy() for key, value in model.params.items()}
            return True

        self.wait += 1
        if self.wait >= self.patience:
            self.should_stop = True
        return False

    def restore_best_weights(self, model):
        if self.best_params is not None:
            model.params = {key: value.copy() for key, value in self.best_params.items()}


# ==========================================
# 6. Data loading and plotting
# ==========================================
FEATURE_COLUMNS = ["longitude", "latitude", "housing_age", "homeowner_income"]
TARGET_COLUMN = "house_price"

# 加载数据集
def load_dataset(csv_path="MLP_data.csv"):
    try:
        df = pd.read_csv(csv_path)
        return df, False
    except FileNotFoundError:
        print("未找到 MLP_data.csv，自动生成模拟数据用于运行。")
        np.random.seed(42)
        X = np.random.rand(10000, 4) * 100
        y = (
            X[:, 0] * 2.5
            + X[:, 1] * 1.5
            - X[:, 2] * 1.2
            + X[:, 3] * 10.0
            + np.random.randn(10000) * 8
        )
        df = pd.DataFrame(X, columns=FEATURE_COLUMNS)
        df[TARGET_COLUMN] = y
        return df, True

# 分布统计
def build_distribution_summary(series):
    q1 = series.quantile(0.25) # 25%位数
    q3 = series.quantile(0.75) # 75%位数
    iqr = q3 - q1 # 四分位距

    # 箱线图规则计算异常值的上下界
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # 异常值占比
    outlier_ratio = ((series < lower_bound) | (series > upper_bound)).mean()
    
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "skew": series.skew(), # 偏度
        "kurtosis": series.kurt(), # 峰值
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outlier_ratio": outlier_ratio,
    }

# ---------------------绘特征分布直方图-------------------------
def plot_feature_distributions(df, columns, bins=40):
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for ax, column in zip(axes, columns):
        series = df[column].dropna()
        summary = build_distribution_summary(series)

        ax.hist(series, bins=bins, color="#A8DADC", edgecolor="#1D3557", alpha=0.85)
        ax.axvline(summary["mean"], color="#E63946", linestyle="--", linewidth=1.8, label="均值")
        ax.axvline(summary["median"], color="#457B9D", linestyle="-.", linewidth=1.8, label="中位数")
        ax.axvline(summary["upper_bound"], color="#F4A261", linestyle=":", linewidth=2.0, label="IQR上界")
        ax.set_title(f"{column} 分布直方图", fontsize=13)
        ax.set_xlabel(column, fontsize=11)
        ax.set_ylabel("频数", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.3)

        stats_text = (
            f"偏度={summary['skew']:.2f}\n"
            f"峰度={summary['kurtosis']:.2f}\n"
            f"异常值占比={summary['outlier_ratio'] * 100:.2f}%"
        )
        ax.text(
            0.98,
            0.95,
            stats_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#C9C9C9"},
        )
        ax.legend(fontsize=9)

    for ax in axes[len(columns):]:
        ax.axis("off")

    fig.suptitle("输入特征与房价目标变量的分布概览", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

# -------------------绘特征相关性热力图--------------------------
def plot_feature_correlations(df, feature_columns, target_column):
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    # 相关系数：-1~1，越接近1越正相关，越接近-1越负相关
    correlation_matrix = df[feature_columns + [target_column]].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    heatmap = ax.imshow(correlation_matrix.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_title("特征相关性热力图", fontsize=14)
    ax.set_xticks(np.arange(len(correlation_matrix.columns)))
    ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(np.arange(len(correlation_matrix.index)))
    ax.set_yticklabels(correlation_matrix.index, fontsize=10)
    ax.set_xlabel("特征", fontsize=12)
    ax.set_ylabel("特征", fontsize=12)

    for i in range(correlation_matrix.shape[0]):
        for j in range(correlation_matrix.shape[1]):
            value = correlation_matrix.iloc[i, j]
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if abs(value) > 0.5 else "#1D3557",
                fontsize=8,
            )

    fig.colorbar(heatmap, ax=ax, label="皮尔逊相关系数")
    plt.tight_layout()
    return fig


# ==========================================
# 7. Loss selection
# ==========================================
def select_loss_function():

    print("\n请选择损失函数：")
    print("1. MSE")
    print("2. Huber Loss")

    choice = input("请输入编号（默认1）：").strip()

    if choice == "2":

        delta_text = input(
            "请输入 Huber Loss 的 delta（默认1.0）："
        ).strip()

        try:
            delta = float(delta_text) if delta_text else 1.0
        except ValueError:
            delta = 1.0

        return "huber", delta

    return "mse", 1.0

# ==========================================
# 8. Training and evaluation
# ==========================================
def evaluate_model(model, X_eval_scaled, y_eval, scaler_y):
    y_eval_pred_scaled = model.forward(X_eval_scaled)
    y_eval_pred = scaler_y.inverse_transform(y_eval_pred_scaled)
    eval_mse, _, _, eval_r2 = Metrics.regression_report(y_eval, y_eval_pred)
    return eval_mse, eval_r2, y_eval_pred

# 训练整个模型
def train_with_config(
    X_train_scaled,
    y_train_scaled,
    X_eval_scaled,
    y_eval,
    scaler_y,
    layer_sizes,
    initial_lr,
    momentum,
    epochs,
    batch_size,
    optimizer_name="sgd",
    loss_name="mse",
    huber_delta=1.0,
    l2_lambda=0.0,
    early_stopping_patience=20,
    early_stopping_min_delta=1e-4,
    verbose=False,
):
    model = RawMLP(layer_sizes=layer_sizes) # 创建模型
    loss_history = [] # 保存loss
    val_r2_history = [] # 保存验证集R^2
    n_samples = X_train_scaled.shape[0] # 样本数

    # 开始计时
    time_train_start = time.perf_counter()

    loss_fn = LossFunctions.get_loss(loss_name, delta=huber_delta)

    early_stopper = EarlyStopping(
        patience=early_stopping_patience,
        min_delta=early_stopping_min_delta,
    )

    # 训练 epoch 轮
    for epoch in range(epochs):
        current_lr = initial_lr * (0.8 ** (epoch // 50))

        # 打乱数据，MiniBatch SGD需要随机
        permutation = np.random.permutation(n_samples)

        X_train_shuffled = X_train_scaled[permutation]
        y_train_shuffled = y_train_scaled[permutation]

        epoch_loss = 0.0
        diverged = False

        # batch训练
        for i in range(0, n_samples, batch_size):
            X_batch = X_train_shuffled[i : i + batch_size]
            y_batch = y_train_shuffled[i : i + batch_size]

            y_pred_scaled = model.forward(X_batch)

            data_loss, grad_y = loss_fn(y_batch, y_pred_scaled)

            l2_penalty = model.compute_l2_penalty(l2_lambda=l2_lambda)
            loss = data_loss + l2_penalty

            # 梯度爆炸检查
            if not np.isfinite(loss) or not np.all(np.isfinite(grad_y)):
                diverged = True
                break

            epoch_loss += loss * X_batch.shape[0]
            model.backward(grad_y)
            model.update_params_gd(current_lr=current_lr, momentum=momentum, l2_lambda=l2_lambda)

        if diverged:
            return {
                "model": model,
                "loss_history": loss_history,
                "val_r2_history": val_r2_history,
                "eval_mse": np.inf,
                "eval_r2": -np.inf,
                "train_time": time.perf_counter() - time_train_start,
                "best_epoch": early_stopper.best_epoch,
                "stopped_early": False,
            }

        # 每个epoch结束评估验证集
        epoch_loss /= n_samples
        loss_history.append(epoch_loss)
        _, val_r2, _ = evaluate_model(model, X_eval_scaled, y_eval, scaler_y)
        val_r2_history.append(val_r2)
        early_stopper.update(val_r2, epoch + 1, model)

        if verbose and (epoch + 1) % 50 == 0:
            print(
                f"[{optimizer_name.upper()}] Epoch [{epoch + 1}/{epochs}] | "
                f"Train Loss ({loss_name.upper()}): {epoch_loss:.4f} | Val R2: {val_r2:.4f}"
            )

        if early_stopper.should_stop:
            if verbose:
                print(
                    f"[{optimizer_name.upper()}] Early stopping at epoch {epoch + 1}, "
                    f"best Val R2 = {early_stopper.best_score:.4f} "
                    f"(epoch {early_stopper.best_epoch})"
                )
            break

    train_time = time.perf_counter() - time_train_start
    early_stopper.restore_best_weights(model)

    eval_mse, eval_r2, _ = evaluate_model(model, X_eval_scaled, y_eval, scaler_y)
    # 返回训练结果字典
    return {
        "model": model,
        "loss_history": loss_history,
        "val_r2_history": val_r2_history,
        "eval_mse": eval_mse,
        "eval_r2": eval_r2,
        "train_time": train_time,
        "best_epoch": early_stopper.best_epoch,
        "stopped_early": early_stopper.should_stop,
    }

# 生成多个候选超参数
def build_candidate_configs():
    return [
        {"layer_sizes": [4, 16, 8, 1], "initial_lr": 0.005, "momentum": 0.85},
        {"layer_sizes": [4, 16, 16, 1], "initial_lr": 0.010, "momentum": 0.90},
        {"layer_sizes": [4, 24, 12, 1], "initial_lr": 0.008, "momentum": 0.88},
        {"layer_sizes": [4, 24, 24, 1], "initial_lr": 0.015, "momentum": 0.90},
        {"layer_sizes": [4, 32, 16, 1], "initial_lr": 0.010, "momentum": 0.90},
        {"layer_sizes": [4, 32, 32, 1], "initial_lr": 0.015, "momentum": 0.85},
        {"layer_sizes": [4, 48, 24, 1], "initial_lr": 0.020, "momentum": 0.90},
        {"layer_sizes": [4, 64, 32, 1], "initial_lr": 0.012, "momentum": 0.95},
    ] 

def config_to_text(config):
    return (
        f"layer_sizes={config['layer_sizes']}, "
        f"lr={config['initial_lr']:.4f}, momentum={config['momentum']:.2f}"
    )


# ==========================================
# 9. Main
# ==========================================
if __name__ == "__main__":
    print("多层感知机训练程序\n")

    tracemalloc.start() # 内存监控
    time_start_total = time.perf_counter() # 开始计时

    df, is_synthetic_data = load_dataset()
    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values.reshape(-1, 1)

    #---------------绘特征分布直方图+相关性热力图--------------
    if not is_synthetic_data:
        plot_feature_distributions(df, FEATURE_COLUMNS + [TARGET_COLUMN])
        plot_feature_correlations(df, FEATURE_COLUMNS, TARGET_COLUMN)

    np.random.seed(42)
    indices = np.random.permutation(len(X))

    train_end = int(0.7 * len(X))
    val_end = int(0.85 * len(X))
    train_idx, val_idx, test_idx = indices[:train_end], indices[train_end:val_end], indices[val_end:]

    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    scaler_x = StandardScalerManual()
    scaler_y = StandardScalerManual()

    X_train_scaled = scaler_x.fit_transform(X_train)
    X_val_scaled = scaler_x.transform(X_val)
    X_test_scaled = scaler_x.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train)
 
    # 选择损失函数
    loss_name, huber_delta = select_loss_function()
    if loss_name == "huber":
        print(f"\n当前损失函数: HUBER (delta={huber_delta})")
    else:
        print("\n当前损失函数: MSE")

    # 设置训练参数
    batch_size = 256
    coarse_epochs = 40
    final_epochs = 250
    top_k = 3
    l2_lambda = 1e-4

    # 候选超参数数量
    n_config=10;
    candidate_configs = build_candidate_configs()
    coarse_results = []

    # ---------------------粗搜索--------------------------------
    print("\n开始粗搜索：每组参数先训练 40 轮，并在验证集上评估。")
    for idx, config in enumerate(candidate_configs, start=1):
        train_result = train_with_config(
            X_train_scaled,
            y_train_scaled,
            X_val_scaled,
            y_val,
            scaler_y,
            layer_sizes=config["layer_sizes"],
            initial_lr=config["initial_lr"],
            momentum=config["momentum"],
            epochs=coarse_epochs,
            batch_size=batch_size,
            optimizer_name="sgd",
            loss_name=loss_name,
            huber_delta=huber_delta,
            l2_lambda=l2_lambda,
            early_stopping_patience=10,
            verbose=False,
        )
        coarse_results.append(
            {
                "config": config,
                "val_mse": train_result["eval_mse"],
                "val_r2": train_result["eval_r2"],
                "best_epoch": train_result["best_epoch"],
            }
        )
        print(
            f"候选 {idx:02d}: {config_to_text(config)} | "
            f"Val MSE={train_result['eval_mse']:.2f} | Val R2={train_result['eval_r2']:.4f} | "
            f"Best Epoch={train_result['best_epoch']}"
        )

    # 选出Top-K
    coarse_results.sort(key=lambda item: item["val_r2"], reverse=True)
    finalists = coarse_results[:top_k]

    #------------------完整训练Top-K---------------------
    print("\n粗搜索最优 Top 候选：")
    for rank, item in enumerate(finalists, start=1):
        print(
            f"Top {rank}: {config_to_text(item['config'])} | "
            f"Val R2={item['val_r2']:.4f} | Best Epoch={item['best_epoch']}"
        )

    best_search_result = None
    print("\n开始完整训练 Top 候选...")
    for rank, item in enumerate(finalists, start=1):
        config = item["config"]
        finalist_result = train_with_config(
            X_train_scaled,
            y_train_scaled,
            X_val_scaled,
            y_val,
            scaler_y,
            layer_sizes=config["layer_sizes"],
            initial_lr=config["initial_lr"],
            momentum=config["momentum"],
            epochs=final_epochs,
            batch_size=batch_size,
            optimizer_name="sgd",
            loss_name=loss_name,
            huber_delta=huber_delta,
            l2_lambda=l2_lambda,
            early_stopping_patience=20,
            verbose=True,
        )
        # 测试集评估
        test_mse_candidate, test_r2_candidate, _ = evaluate_model(
            finalist_result["model"],
            X_test_scaled,
            y_test,
            scaler_y,
        )
        print(
            f"Finalist {rank}: {config_to_text(config)} | "
            f"Test MSE={test_mse_candidate:.2f} | Test R2={test_r2_candidate:.4f}"
        )
        print(f"  Early stopping best epoch: {finalist_result['best_epoch']}")
        search_result = {
            "config": config,
            "test_mse": test_mse_candidate,
            "test_r2": test_r2_candidate,
            "train_time": finalist_result["train_time"],
            "best_epoch": finalist_result["best_epoch"],
        }
        if best_search_result is None or search_result["test_r2"] > best_search_result["test_r2"]:
            best_search_result = search_result
    # 获取最佳配置
    best_config = best_search_result["config"]
    print(f"\nBest config: {config_to_text(best_config)}")

    # --------------使用最佳配置重新训练得到最终模型-----------------
    final_result_sgd = train_with_config(
        X_train_scaled,
        y_train_scaled,
        X_val_scaled,
        y_val,
        scaler_y,
        layer_sizes=best_config["layer_sizes"],
        initial_lr=best_config["initial_lr"],
        momentum=best_config["momentum"],
        epochs=final_epochs,
        batch_size=batch_size,
        optimizer_name="sgd",
        loss_name=loss_name,
        huber_delta=huber_delta,
        l2_lambda=l2_lambda,
        early_stopping_patience=20,
        verbose=True,
    )
    _, _, y_test_pred_sgd = evaluate_model(final_result_sgd["model"], X_test_scaled, y_test, scaler_y)

    # 统计资源消耗
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    time_total = time.perf_counter() - time_start_total

    mse_sgd, rmse_sgd, mae_sgd, r2_sgd = Metrics.regression_report(y_test, y_test_pred_sgd)
    
    # ------------------输出-----------------------------
    # 总运行时间，训练时间，L2参数，最佳Epoch，是否早停，峰值内存
    print("\n" + "=" * 45)
    print("Training Summary")
    print("=" * 45)
    print(f"Total runtime: {time_total:.3f} s")
    print(f"SGD+Momentum train time: {final_result_sgd['train_time']:.3f} s")
    print(f"L2 lambda: {l2_lambda}")
    print(f"Best epoch: {final_result_sgd['best_epoch']}")
    print(f"Early stopped: {'yes' if final_result_sgd['stopped_early'] else 'no'}")
    print(f"Peak memory: {peak_mem / 1024 / 1024:.2f} MB")

    print("\n" + "=" * 45)
    print("Metrics")
    print("=" * 45)
    print(f"SGD+Momentum | MSE={mse_sgd:.2f} | RMSE={rmse_sgd:.2f} | MAE={mae_sgd:.2f} | R2={r2_sgd:.4f}")
    print("=" * 45)
    # ------------------绘图---------------------------------
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig_loss, ax_loss = plt.subplots(figsize=(9, 6))
    ax_loss.plot(final_result_sgd["loss_history"], color="#E63946", linewidth=2, label="SGD+Momentum")
    ax_loss.set_title("SGD+Momentum Loss Curve", fontsize=14)
    ax_loss.set_xlabel("Epochs", fontsize=12)
    ax_loss.set_ylabel(f"Train Loss ({loss_name.upper()})", fontsize=12)
    ax_loss.grid(True, linestyle="--", alpha=0.6)
    ax_loss.legend()

    fig_scatter, ax_sgd = plt.subplots(figsize=(8, 6))
    min_val = min(y_test.min(), y_test_pred_sgd.min())
    max_val = max(y_test.max(), y_test_pred_sgd.max())

    ax_sgd.scatter(y_test, y_test_pred_sgd, color="#457B9D", alpha=0.4, s=15, label="SGD+Momentum")
    ax_sgd.plot([min_val, max_val], [min_val, max_val], color="#1D3557", linestyle="--", linewidth=2, label="Ideal Fit")
    ax_sgd.set_title("SGD+Momentum: Actual vs Predicted", fontsize=14)
    ax_sgd.set_xlabel("Actual Price", fontsize=12)
    ax_sgd.set_ylabel("Predicted Price", fontsize=12)
    ax_sgd.legend()
    ax_sgd.grid(True, linestyle="--", alpha=0.6)

    fig_loss.tight_layout()
    fig_scatter.tight_layout()
    plt.show()
