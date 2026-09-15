import os
import time
import copy
import random
import typing
from collections import OrderedDict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image, UnidentifiedImageError

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

if not hasattr(typing, "OrderedDict"):
    typing.OrderedDict = OrderedDict

from torchvision import datasets, transforms
from torchvision.datasets.folder import IMG_EXTENSIONS


# =========================
# 1. 全局配置
# =========================

class Config:
    data_root = "./cnn图片"
    train_dir = "./cnn图片/train"
    test_dir = "./cnn图片/test"

    class_names = ["baihe", "dangshen", "gouqi", "huaihua", "jinyinhua"]
    num_classes = 5

    img_size = 128 # 越大捕捉细节越多
    batch_size = 32
    epochs = 30 # 太少欠拟合，太多过拟合
    lr = 1e-3
    weight_decay = 1e-4 # 权重衰减（常用于L2正则化）
    label_smoothing = 0.05 # 标签平滑，正则化

    num_workers = 4 # 子进程数量
    prefetch_factor = 2
    persistent_workers = True
    seed = 42

    output_dir = "./outputs"
    model_name = "best_herb_cnn.pth"
    do_feature_analysis = False
    feature_sample_per_class = 5
    feature_analysis_dir = "feature_analysis"

    use_amp = True  # GPU 上开启混合精度训练，可加速


cfg = Config()


# =========================
# 2. 固定随机种子
# =========================

def set_seed(seed=42):
    random.seed(seed) # 固定 random 模块的随机状态
    np.random.seed(seed) # 固定 NumPy 库的随机数生成器
    torch.manual_seed(seed) # 为 PyTorch 的 CPU 随机数生成器设置种子
    torch.cuda.manual_seed_all(seed) # 为 所有 GPU（CUDA） 设置随机数种子

    torch.backends.cudnn.benchmark = True # 自动寻找并选择最高效/最快的卷积计算算法

# =========================
# 3. 数据增强与加载
# =========================

def build_transforms(img_size):
    """
    训练集使用数据增强，提升泛化能力。
    测试集不使用随机增强，保证测试结果稳定。
    """

    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.15
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], # ImageNet 数据集的百万张图片的统计值
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_tfms, test_tfms

# 读取文件+关联规则+分批次给神经网络
class FlatImageFolder(Dataset):
    def __init__(self, root, class_names, transform=None):
        self.root = Path(root)
        self.class_names = class_names
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.transform = transform
        self.samples = self._make_samples()
        self.targets = [label for _, label in self.samples]

    def _make_samples(self):
        samples = []
        class_names = sorted(self.class_names, key=len, reverse=True)

        for path in sorted(self.root.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMG_EXTENSIONS:
                continue

            filename = path.name.lower()
            label_name = next((name for name in class_names if filename.startswith(name.lower())), None)
            if label_name is None:
                continue

            samples.append((str(path), self.class_to_idx[label_name]))

        if len(samples) == 0:
            raise RuntimeError(f"Found 0 valid images in {self.root}")

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def build_dataloaders(cfg):
    train_tfms, test_tfms = build_transforms(cfg.img_size)

    train_dataset = datasets.ImageFolder(cfg.train_dir, transform=train_tfms)
    train_eval_dataset = datasets.ImageFolder(cfg.train_dir, transform=test_tfms)
    test_has_class_dirs = any(
        path.is_dir() and path.name in cfg.class_names
        for path in Path(cfg.test_dir).iterdir()
    )
    if test_has_class_dirs:
        test_dataset = datasets.ImageFolder(cfg.test_dir, transform=test_tfms)
    else:
        test_dataset = FlatImageFolder(cfg.test_dir, cfg.class_names, transform=test_tfms)

    dataloader_kwargs = {
        "num_workers": cfg.num_workers,
        "pin_memory": True
    }
    if cfg.num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = cfg.prefetch_factor
        dataloader_kwargs["persistent_workers"] = cfg.persistent_workers

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        **dataloader_kwargs
    )
    train_eval_loader = DataLoader(
        train_eval_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        **dataloader_kwargs
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        **dataloader_kwargs
    )

    return train_loader, train_eval_loader, test_loader, train_dataset, test_dataset

# 统计数据集每个类别的图片数量
def get_class_counts(dataset, class_names):
    counts = [0 for _ in class_names]
    for _, label in dataset.samples:
        counts[label] += 1
    return counts

# 训练集特征分析：类别分布、尺寸、长宽比和颜色特征（RGB 均值），找出损坏或不合规的“异常”图像
def analyze_train_dataset_features(cfg):
    analysis_dir = Path(cfg.output_dir) / cfg.feature_analysis_dir
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # 初始化手机统计信息的字典和列表
    train_root = Path(cfg.train_dir)
    class_image_paths = {class_name: [] for class_name in cfg.class_names}
    class_counts = {class_name: 0 for class_name in cfg.class_names}
    widths = []
    heights = []
    aspect_ratios = []
    rgb_sums = {class_name: np.zeros(3, dtype=np.float64) for class_name in cfg.class_names}
    rgb_counts = {class_name: 0 for class_name in cfg.class_names}
    abnormal_images = []

    for class_name in cfg.class_names:
        class_dir = train_root / class_name
        if not class_dir.exists():
            abnormal_images.append(f"Missing class directory: {class_dir}")
            continue

        for path in sorted(class_dir.rglob("*")):
            if not path.is_file():
                continue

            if path.suffix.lower() not in IMG_EXTENSIONS:
                abnormal_images.append(f"Non-image file: {path}")
                continue

            try:
                with Image.open(path) as image:
                    image.verify() # 验证图片是否损坏
                with Image.open(path) as image:
                    image = image.convert("RGB") # 统一转为RGB 3通道
                    width, height = image.size
                    image_array = np.asarray(image, dtype=np.float32)
            except (OSError, UnidentifiedImageError) as exc:
                abnormal_images.append(f"Cannot open image: {path} | {exc}")
                continue
            
            # 检查图片尺寸
            if width < 64 or height < 64: 
                abnormal_images.append(f"Too small image: {path} | {width}x{height}")

            class_image_paths[class_name].append(path)
            class_counts[class_name] += 1
            widths.append(width)
            heights.append(height)
            aspect_ratios.append(width / height)
            rgb_sums[class_name] += image_array.reshape(-1, 3).mean(axis=0)
            rgb_counts[class_name] += 1

    print("\nTrain class counts")
    print(class_counts)

    class_names = cfg.class_names
    count_values = [class_counts[name] for name in class_names]

    plt.figure(figsize=(8, 5))
    plt.bar(class_names, count_values)
    plt.xticks(rotation=20)
    plt.ylabel("Number of Images")
    plt.title("Train Class Distribution")
    plt.tight_layout()
    plt.savefig(analysis_dir / "train_class_distribution.png", dpi=300)
    plt.close()

    if widths and heights:
        widths_array = np.array(widths)
        heights_array = np.array(heights)
        print(
            "Train image size stats | "
            f"min width: {widths_array.min():.0f}, "
            f"max width: {widths_array.max():.0f}, "
            f"mean width: {widths_array.mean():.2f}, "
            f"min height: {heights_array.min():.0f}, "
            f"max height: {heights_array.max():.0f}, "
            f"mean height: {heights_array.mean():.2f}"
        )

        plt.figure(figsize=(8, 5))
        plt.hist(widths, bins=30, alpha=0.7, label="Width")
        plt.hist(heights, bins=30, alpha=0.7, label="Height")
        plt.xlabel("Pixels")
        plt.ylabel("Number of Images")
        plt.title("Train Image Size Distribution")
        plt.legend()
        plt.tight_layout()
        plt.savefig(analysis_dir / "train_image_size_distribution.png", dpi=300)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.hist(aspect_ratios, bins=30)
        plt.xlabel("Width / Height")
        plt.ylabel("Number of Images")
        plt.title("Train Aspect Ratio Distribution")
        plt.tight_layout()
        plt.savefig(analysis_dir / "train_aspect_ratio_distribution.png", dpi=300)
        plt.close()
    else:
        widths_array = np.array([])
        heights_array = np.array([])

    sample_count = max(1, cfg.feature_sample_per_class)
    fig, axes = plt.subplots(
        len(class_names),
        sample_count,
        figsize=(sample_count * 2.2, len(class_names) * 2.2)
    )
    axes = np.asarray(axes).reshape(len(class_names), sample_count)
    rng = random.Random(cfg.seed)

    for row, class_name in enumerate(class_names):
        paths = class_image_paths[class_name]
        selected_paths = rng.sample(paths, min(sample_count, len(paths))) if paths else []
        for col in range(sample_count):
            ax = axes[row, col]
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(class_name, rotation=0, labelpad=35, va="center")
            if col >= len(selected_paths):
                continue
            try:
                with Image.open(selected_paths[col]) as image:
                    ax.imshow(image.convert("RGB"))
            except (OSError, UnidentifiedImageError):
                ax.set_title("Invalid")

    plt.tight_layout()
    plt.savefig(analysis_dir / "train_samples_by_class.png", dpi=300)
    plt.close(fig)

    mean_rgb_by_class = []
    print("\nTrain mean RGB by class")
    for class_name in class_names:
        if rgb_counts[class_name] > 0:
            mean_rgb = rgb_sums[class_name] / rgb_counts[class_name]
        else:
            mean_rgb = np.zeros(3, dtype=np.float64)
        mean_rgb_by_class.append(mean_rgb)
        print(f"{class_name:10s}: R={mean_rgb[0]:.2f}, G={mean_rgb[1]:.2f}, B={mean_rgb[2]:.2f}")

    x = np.arange(len(class_names))
    bar_width = 0.25
    mean_rgb_array = np.array(mean_rgb_by_class)

    plt.figure(figsize=(9, 5))
    plt.bar(x - bar_width, mean_rgb_array[:, 0], bar_width, label="R")
    plt.bar(x, mean_rgb_array[:, 1], bar_width, label="G")
    plt.bar(x + bar_width, mean_rgb_array[:, 2], bar_width, label="B")
    plt.xticks(x, class_names, rotation=20)
    plt.ylabel("Mean RGB Value")
    plt.title("Train Mean RGB by Class")
    plt.legend()
    plt.tight_layout()
    plt.savefig(analysis_dir / "train_mean_rgb_by_class.png", dpi=300)
    plt.close()

    abnormal_path = analysis_dir / "train_abnormal_images.txt"
    with open(abnormal_path, "w", encoding="utf-8") as f:
        if abnormal_images:
            f.write("\n".join(abnormal_images))
        else:
            f.write("No abnormal images found.")

    if widths_array.size > 0 and heights_array.size > 0:
        if widths_array.mean() > 160 and heights_array.mean() > 160:
            print("Images are large enough. img_size=160 is reasonable.")
        elif widths_array.mean() <= 128 or heights_array.mean() <= 128:
            print("Many images are small. Consider using img_size=128 to speed up training.")

    nonzero_counts = [count for count in count_values if count > 0]
    if nonzero_counts and max(nonzero_counts) / min(nonzero_counts) > 1.5:
        print("Class imbalance detected. Keeping class_weights is recommended.")
    else:
        print("Class distribution is relatively balanced.")

    if abnormal_images:
        print("Abnormal images detected. Please check train_abnormal_images.txt.")
    else:
        print("No abnormal images detected.")


# =========================
# 4. 模型设计：轻量 CNN
# =========================

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(  # 卷积提取特征
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False # 批归一化抵消偏置
            ),
            nn.BatchNorm2d(out_channels), # 批归一化：标准正态分布
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class HerbCNN(nn.Module):
    """
    “特征提取+分类器”双段式结构
    """

    def __init__(self, num_classes=5, dropout=0.35):
        super().__init__()

        # 空间尺寸减小，特征通道数增多
        # 得到形如[batch_size, 256, 10, 10] 的四维张量
        self.features = nn.Sequential(
            ConvBNReLU(3, 32),
            ConvBNReLU(32, 32),
            nn.MaxPool2d(2),   # 160 -> 80

            ConvBNReLU(32, 64),
            ConvBNReLU(64, 64),
            nn.MaxPool2d(2),   # 80 -> 40

            ConvBNReLU(64, 128),
            ConvBNReLU(128, 128),
            nn.MaxPool2d(2),   # 40 -> 20

            ConvBNReLU(128, 256),
            ConvBNReLU(256, 256),
            nn.MaxPool2d(2),   # 20 -> 10
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), # 全局平均池化 -> [batch_size, 256, 1, 1]
            nn.Flatten(), # 一维向量 -> [batch_size, 256],便于全连接层接受
            nn.Dropout(dropout), # 随机丢弃 35% 的神经元防止过拟合
            nn.Linear(256, 128), # 全连接层：从 256 降维到 128
            nn.ReLU(inplace=True), 
            nn.Dropout(dropout),
            nn.Linear(128, num_classes) # 最后一层全连接：输出 5 个类别的得分
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# =========================
# 5. 评测指标：包括准确率和损失
# 还包括更混淆矩阵、精确率、召回率和 F1 分数
# =========================

# 局部准确率
def accuracy_from_logits(logits, labels):
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct, total


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes=5):
    model.eval() # 切换评估模式，关闭随机训练行为

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True) # non_blocking配合pin_memory=True
        labels = labels.to(device, non_blocking=True) 

        outputs = model(images)
        loss = criterion(outputs, labels)

        correct, total = accuracy_from_logits(outputs, labels)

        total_loss += loss.item() * images.size(0)
        total_correct += correct
        total_samples += total

        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy().tolist()) 
        all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples

    cm = build_confusion_matrix(all_labels, all_preds, num_classes)
    metrics = classification_metrics_from_cm(cm)

    return avg_loss, acc, cm, metrics


def build_confusion_matrix(labels, preds, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(labels, preds):
        cm[true_label][pred_label] += 1
    return cm

# 基于混淆矩阵，计算核心指标：tp真正例，fp假正例，fn假负例
def classification_metrics_from_cm(cm):
    num_classes = cm.shape[0]

    precisions = []
    recalls = []
    f1s = []

    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        precision = tp / (tp + fp + 1e-8) # 精确率
        recall = tp / (tp + fn + 1e-8) # 召回率
        f1 = 2 * precision * recall / (precision + recall + 1e-8) # F1分数——调和平均数

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    metrics = {
        "precision_per_class": precisions,
        "recall_per_class": recalls,
        "f1_per_class": f1s,
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s))
    }

    return metrics


# =========================
# 6. 训练函数
# =========================

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train() # 调到训练模式

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad() # 清空梯度

        # AMP 自动混合精度分支
        if scaler is not None: # NVIDIA 显卡+use_amp=True
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward() # 防止下溢出，放大loss
            scaler.step(optimizer) # 把放大后的梯度缩回正常比例
            scaler.update()
        else: # Intel 核显/ CPU 
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        correct, total = accuracy_from_logits(outputs, labels)

        total_loss += loss.item() * images.size(0)
        total_correct += correct
        total_samples += total

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples

    return avg_loss, acc


def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, device, cfg):
    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "lr": []
    }

    best_model_wts = copy.deepcopy(model.state_dict())
    best_train_acc = 0.0

    start_time = time.time()

    scaler = None
    if device.type == "cuda" and cfg.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(cfg.epochs):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler
        )

        scheduler.step() # 动态学习率调整

        current_lr = optimizer.param_groups[0]["lr"]
        test_loss, test_acc, _, _ = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            num_classes=cfg.num_classes
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["lr"].append(current_lr)

        if train_acc > best_train_acc:
            best_train_acc = train_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        epoch_time = time.time() - epoch_start

        print(
            f"Epoch [{epoch + 1:03d}/{cfg.epochs}] "
            f"Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc:.4f} | "
            f"LR: {current_lr:.6f} | "
            f"Time: {epoch_time:.2f}s"
        )

    total_time = time.time() - start_time

    model.load_state_dict(best_model_wts)

    return model, history, total_time


# =========================
# 7. 模型性能统计：参量数、磁盘占用和推理速度
# =========================

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model_path):
    size_bytes = os.path.getsize(model_path)
    return size_bytes / 1024 / 1024 # 转MB


@torch.no_grad()
def benchmark_inference(model, loader, device):
    model.eval()

    total_samples = 0

    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.time()

    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        _ = model(images)
        total_samples += images.size(0)

    if device.type == "cuda":
        torch.cuda.synchronize()

    total_time = time.time() - start_time

    ms_per_image = total_time / total_samples * 1000 # 每张图片耗时
    fps = total_samples / total_time # 每秒传输帧数

    return total_time, ms_per_image, fps


# =========================
# 8. 可视化函数
# =========================

def plot_training_curves(history, output_dir):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    plt.plot(epochs, history["test_loss"], marker="s", label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train/Test Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "train_loss_curve.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["train_acc"], marker="o", label="Train Accuracy")
    plt.plot(epochs, history["test_acc"], marker="s", label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train/Test Accuracy Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "train_accuracy_curve.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["lr"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "learning_rate_curve.png"), dpi=300)
    plt.close()


def plot_class_distribution(train_counts, test_counts, class_names, output_dir):
    x = np.arange(len(class_names))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, train_counts, width, label="Train")
    plt.bar(x + width / 2, test_counts, width, label="Test")
    plt.xticks(x, class_names, rotation=20)
    plt.ylabel("Number of Images")
    plt.title("Class Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "class_distribution.png"), dpi=300)
    plt.close()


def plot_confusion_matrix(cm, class_names, output_dir):
    plt.figure(figsize=(7, 6))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
    plt.close()


def plot_final_accuracy_bar(train_acc, test_acc, output_dir):
    names = ["Train Accuracy", "Test Accuracy"]
    values = [train_acc, test_acc]

    plt.figure(figsize=(6, 5))
    plt.bar(names, values)
    plt.ylim(0, 1.0)
    plt.ylabel("Accuracy")
    plt.title("Final Train/Test Accuracy")
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f"{v:.4f}", ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "final_accuracy_bar.png"), dpi=300)
    plt.close()


# =========================
# 9. 结果打印
# =========================

def print_metrics(metrics, class_names):
    print("\n========== Classification Metrics ==========")

    for i, name in enumerate(class_names):
        print(
            f"{name:10s} | "
            f"Precision: {metrics['precision_per_class'][i]:.4f} | "
            f"Recall: {metrics['recall_per_class'][i]:.4f} | "
            f"F1: {metrics['f1_per_class'][i]:.4f}"
        )
    # Macro 宏平均
    print("--------------------------------------------")
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro Recall   : {metrics['macro_recall']:.4f}")
    print(f"Macro F1       : {metrics['macro_f1']:.4f}")


# =========================
# 10. 主函数
# =========================

def print_image_backend_info():
    print("\n========== Image Backend ==========")
    try:
        pil_version = getattr(PIL, "__version__", "unknown")
        backend_name = "Pillow-SIMD" if "post" in pil_version.lower() else "Pillow"
        print(f"PIL version: {pil_version}")
        print(f"Image backend: {backend_name}")
    except Exception as exc:
        print("PIL version: unknown")
        print(f"Image backend: unknown ({exc})")


def main():
    # -------------环境初始化-------------------------
    set_seed(cfg.seed)
    print_image_backend_info()

    os.makedirs(cfg.output_dir, exist_ok=True)

    if cfg.do_feature_analysis:
        analyze_train_dataset_features(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, train_eval_loader, test_loader, train_dataset, test_dataset = build_dataloaders(cfg)

    print("\nClass to index:")
    print(train_dataset.class_to_idx)

    # --------------数据集分析与可视化--------------------
    train_counts = get_class_counts(train_dataset, cfg.class_names)
    test_counts = get_class_counts(test_dataset, cfg.class_names)

    print("\nTrain class counts:", dict(zip(cfg.class_names, train_counts)))
    print("Test class counts :", dict(zip(cfg.class_names, test_counts)))

    plot_class_distribution(
        train_counts=train_counts,
        test_counts=test_counts,
        class_names=cfg.class_names,
        output_dir=cfg.output_dir
    )
    # ------------------模型构建与优化组件配置-------------
    model = HerbCNN(num_classes=cfg.num_classes).to(device)
    if device.type == "cuda" and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("torch.compile enabled for GPU training.")
        except Exception as exc:
            print(f"torch.compile is unavailable in this environment: {exc}")
    elif device.type == "cuda":
        print("torch.compile is not supported by this PyTorch version.")

    num_params = count_parameters(model)
    print(f"\nTrainable parameters: {num_params:,}")

    # 类别不平衡时，使用 class weight 可以让小样本类别更受重视
    class_weights = 1.0 / (torch.tensor(train_counts, dtype=torch.float32) + 1e-8)
    class_weights = class_weights / class_weights.sum() * cfg.num_classes
    class_weights = class_weights.to(device)

    # 标准的交叉熵损失
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=cfg.label_smoothing
    )
    # 优化器（自带L2正则化）
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )
    # 余弦退火调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs
    )
    # ---------------开启训练与模型封存------------------
    print("\n========== Start Training ==========")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model, history, train_time = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        cfg=cfg
    )

    model_path = os.path.join(cfg.output_dir, cfg.model_name)
    model_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save(model_to_save.state_dict(), model_path)
    # ----------------多维评估与性能基准测试------------------
    print("\n========== Final Evaluation ==========")

    # 训练集准确率：用不带随机增强的 train_eval_loader 重新评估
    train_loss, train_acc, train_cm, train_metrics = evaluate(
        model=model,
        loader=train_eval_loader,
        criterion=criterion,
        device=device,
        num_classes=cfg.num_classes
    )

    # 测试集准确率：训练结束后最终评测
    test_loss, test_acc, test_cm, test_metrics = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        num_classes=cfg.num_classes
    )
    # 硬件性能测速
    infer_time, ms_per_image, fps = benchmark_inference(
        model=model,
        loader=test_loader,
        device=device
    )
    # 硬盘占用+显存峰值
    model_size = get_model_size_mb(model_path)

    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        peak_memory = 0.0

    print("\n========== Main Results ==========")
    print(f"Final Train Loss: {train_loss:.4f}")
    print(f"Final Train Acc : {train_acc:.4f}")
    print(f"Final Test Loss : {test_loss:.4f}")
    print(f"Final Test Acc  : {test_acc:.4f}")

    print("\n========== Code / Model Performance ==========")
    print(f"Training Time        : {train_time:.2f} s")
    print(f"Inference Time       : {infer_time:.4f} s")
    print(f"ms / image           : {ms_per_image:.4f} ms")
    print(f"FPS                  : {fps:.2f} images/s")
    print(f"Trainable Parameters : {num_params:,}")
    print(f"Model Size           : {model_size:.2f} MB")
    if device.type == "cuda":
        print(f"Peak GPU Memory      : {peak_memory:.2f} MB")

    print_metrics(test_metrics, cfg.class_names)

    plot_training_curves(history, cfg.output_dir)
    plot_confusion_matrix(test_cm, cfg.class_names, cfg.output_dir)
    plot_final_accuracy_bar(train_acc, test_acc, cfg.output_dir)

    print("\nAll figures and model have been saved to:", cfg.output_dir)


if __name__ == "__main__":
    main()
