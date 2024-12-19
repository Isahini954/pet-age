import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import joblib
import logging

# 设置matplotlib显示中文和负号
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置日志记录
logging.basicConfig(filename='data_cleaning.log',
                    level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# 设置随机种子以确保结果可复现
def set_random_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 列出目录下所有有效格式的图片文件
def list_images(directory):
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    return [f for f in os.listdir(directory) if os.path.splitext(f)[1].lower() in valid_extensions]

# 加载标签文件
def load_annotations(annotation_path):
    try:
        return pd.read_csv(annotation_path, sep='\t', header=None, names=['image_name', 'age_month'], encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(annotation_path, sep='\t', header=None, names=['image_name', 'age_month'], encoding='gbk')

# 自定义数据集类
class PetAgeDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, scaler=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.scaler = scaler

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, 'image_name']
        img_path = os.path.join(self.img_dir, img_name)
        if not os.path.exists(img_path):
            logging.warning(f'Image not found: {img_path}. Returning black image.')
            return torch.zeros(3, 224, 224, dtype=torch.float32), torch.tensor(self.df.loc[idx, 'age_month_scaled'], dtype=torch.float32)
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                img_np = np.array(img)
                if self.transform:
                    transformed = self.transform(image=img_np)
                    image = transformed['image']
                else:
                    # 如果没有定义transform，手动转换为Tensor
                    image = torch.tensor(img_np).permute(2, 0, 1).float() / 255.0
            age = self.df.loc[idx, 'age_month_scaled']
            return image, torch.tensor(age, dtype=torch.float32)
        except Exception as e:
            logging.warning(f'Unable to load image {img_path}. Returning black image. Error: {e}')
            return torch.zeros(3, 224, 224, dtype=torch.float32), torch.tensor(self.df.loc[idx, 'age_month_scaled'], dtype=torch.float32)

# 定义模型
class EfficientNetRegressor(nn.Module):
    def __init__(self, pretrained=True, dropout_prob=0.5):
        super(EfficientNetRegressor, self).__init__()
        self.model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None)
        num_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(num_features, 1)
        )

    def forward(self, x):
        return self.model(x).squeeze(1)  # 返回形状为 (batch_size,)

# 定义组合损失
class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def forward(self, preds, targets):
        return self.l1_loss(preds, targets) + self.mse_loss(preds, targets)

# 训练一个epoch的函数
def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    for images, ages in tqdm(loader, desc='Training', leave=False):
        images, ages = images.to(device), ages.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, ages)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)

# 验证一个epoch的函数
def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, ages in tqdm(loader, desc='Validation', leave=False):
            images = images.to(device, non_blocking=True)
            ages = ages.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, ages)
            running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)

# 评估模型
def evaluate_model(model, loader, device):
    model.eval()
    preds = []
    true_vals = []
    with torch.no_grad():
        for images, ages in tqdm(loader, desc='Evaluating', leave=False):
            images = images.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                outputs = model(images)
            preds.extend(outputs.cpu().numpy())
            true_vals.extend(ages.cpu().numpy())
    return preds, true_vals

# 清理注释，移除不存在的图像
def clean_annotations(df, img_dir, dataset_type='Dataset'):
    existing_images = set(list_images(img_dir))
    df_clean = df[df['image_name'].isin(existing_images)].copy()
    missing_images = set(df['image_name']) - set(df_clean['image_name'])
    if missing_images:
        logging.warning(f'The following images are missing in {img_dir}: {missing_images}')
        print(f'Warning: {len(missing_images)} images in {dataset_type} are missing in {img_dir}. Check "data_cleaning.log" for details.')
    return df_clean

# 匹配图像与标签
def match_images_to_annotations(image_list, annotation_df):
    annotation_df['image_name_clean'] = annotation_df['image_name'].str.strip().str.lower()
    annotation_dict = dict(zip(annotation_df['image_name_clean'], annotation_df['age_month']))
    matched = []
    missing = []
    for img in image_list:
        img_clean = img.strip().lower()
        if img_clean in annotation_dict:
            matched.append({'image_name': img, 'age_month': annotation_dict[img_clean]})
        else:
            missing.append(img)
    if missing:
        logging.warning(f'No corresponding labels for images: {missing}')
    return pd.DataFrame(matched)

# 主函数
def main():
    # 设置随机种子
    set_random_seed()

    # 确定使用的设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 数据路径
    train_img_dir = "C:\\Users\\24942\\Desktop\\深度学习竞赛\\trainset"
    val_img_dir = "C:\\Users\\24942\\Desktop\\深度学习竞赛\\valset"
    train_label_path = "C:\\Users\\24942\\Desktop\\深度学习竞赛\\annotations\\annotations\\train.txt"
    val_label_path = "C:\\Users\\24942\\Desktop\\深度学习竞赛\\annotations\\annotations\\val.txt"

    # 加载标签文件
    train_annotations = load_annotations(train_label_path)
    val_annotations = load_annotations(val_label_path)

    # 列出图像目录中的所有图像
    train_images = list_images(train_img_dir)
    val_images = list_images(val_img_dir)

    # 匹配图像与标签
    train_matched = match_images_to_annotations(train_images, train_annotations)
    val_matched = match_images_to_annotations(val_images, val_annotations)

    # 清理匹配后的数据，移除不存在的图像
    train_matched = clean_annotations(train_matched, train_img_dir, dataset_type='Train')
    val_matched = clean_annotations(val_matched, val_img_dir, dataset_type='Validation')

    # 检查缺失的标签
    missing_train = set(train_annotations['image_name'].str.strip().str.lower()) - set(train_matched['image_name'].str.strip().str.lower())
    missing_val = set(val_annotations['image_name'].str.strip().str.lower()) - set(val_matched['image_name'].str.strip().str.lower())
    if missing_train:
        logging.warning(f'The following training images are missing in {train_img_dir}: {missing_train}')
    if missing_val:
        logging.warning(f'The following validation images are missing in {val_img_dir}: {missing_val}')

    # 拆分训练集和验证集
    train_data, val_data = train_test_split(train_matched, test_size=0.2, random_state=42, shuffle=True)

    # 定义数据增强和预处理
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomBrightnessContrast(p=0.3),
        A.Rotate(limit=20, p=0.7),
        A.RandomResizedCrop(height=224, width=224, scale=(0.7, 1.0), p=0.7),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3, p=0.4),
        A.CoarseDropout(max_holes=2, max_height=32, max_width=32, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    # 标准化年龄
    age_scaler = StandardScaler()
    train_data['age_month_scaled'] = age_scaler.fit_transform(train_data[['age_month']])
    val_data['age_month_scaled'] = age_scaler.transform(val_data[['age_month']])
    val_matched['age_month_scaled'] = age_scaler.transform(val_matched[['age_month']])

    # 保存标准化器
    joblib.dump(age_scaler, 'age_scaler.pkl')

    # 创建数据集和数据加载器
    train_dataset = PetAgeDataset(train_data, train_img_dir, transform=train_transform, scaler=age_scaler)
    val_dataset = PetAgeDataset(val_data, train_img_dir, transform=val_transform, scaler=age_scaler)
    test_dataset = PetAgeDataset(val_matched, val_img_dir, transform=val_transform, scaler=age_scaler)  # 使用 val_matched

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # 初始化模型
    model = EfficientNetRegressor(pretrained=True).to(device)
    for param in model.parameters():
        param.requires_grad = True

    # 定义损失函数和优化器
    criterion = CombinedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    # 设置TensorBoard日志目录
    log_dir = "C:\\Users\\24942\\Desktop\\1\\pet_age_prediction_new"
    if os.path.exists(log_dir) and os.path.isfile(log_dir):
        os.remove(log_dir)
    elif not os.path.exists(log_dir):
        os.makedirs(log_dir)

    writer = SummaryWriter(log_dir=log_dir)

    # 训练参数
    num_epochs = 10
    patience = 5
    best_val_loss = float('inf')
    early_stop_counter = 0

    # 混合精度训练
    amp_scaler = torch.cuda.amp.GradScaler()

    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        train_loss = train_epoch(model, train_loader, criterion, optimizer, amp_scaler, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Current Learning Rate: {current_lr}')

        writer.add_scalars('Loss', {'Train': train_loss, 'Validation': val_loss}, epoch + 1)
        writer.add_scalar('Learning Rate', current_lr, epoch + 1)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print('Best model saved.')
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print('Early stopping triggered. Training stopped.')
                break

    writer.close()

    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pth'))

    # 在验证集上评估模型
    val_preds_scaled, val_true_scaled = evaluate_model(model, val_loader, device)
    val_preds = age_scaler.inverse_transform(np.array(val_preds_scaled).reshape(-1, 1)).flatten()
    val_true = age_scaler.inverse_transform(np.array(val_true_scaled).reshape(-1, 1)).flatten()
    val_mae = mean_absolute_error(val_true, val_preds)
    val_mse = mean_squared_error(val_true, val_preds)
    print(f'\nValidation MAE: {val_mae:.2f} months')
    print(f'Validation MSE: {val_mse:.2f} months^2')

    # 在测试集上进行预测
    test_preds_scaled, _ = evaluate_model(model, test_loader, device)
    test_preds = age_scaler.inverse_transform(np.array(test_preds_scaled).reshape(-1, 1)).flatten()
    test_preds = np.clip(test_preds, 0, 191)
    test_preds = np.round(test_preds).astype(int)

    assert len(test_preds) == len(val_matched), "Prediction results do not match the number of test images."

    # 创建提交文件
    submission = pd.DataFrame({
        'image_name': val_matched['image_name'],
        'age_month': test_preds
    })

    # 检查提交文件中的图像是否存在于 valset 目录
    missing_in_valset = set(submission['image_name']) - set(val_images)
    if missing_in_valset:
        print(f'Warning: The following images were not found in the valset folder: {missing_in_valset}')
        submission = submission[~submission['image_name'].isin(missing_in_valset)]
    else:
        print('All prediction result images exist in the valset folder.')

    submission.to_csv('pred_result.txt', sep='\t', index=False, header=False)
    print('\nPrediction results have been saved as pred_result.txt')

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png')
    plt.show()

    # 绘制真实值与预测值的对比
    plt.figure(figsize=(6, 6))
    plt.scatter(val_true, val_preds, alpha=0.5)
    plt.xlabel('True Age (months)')
    plt.ylabel('Predicted Age (months)')
    plt.title('True vs Predicted')
    plt.plot([0, 200], [0, 200], 'r--')
    plt.grid(True)
    plt.savefig('pred_vs_true.png')
    plt.show()

if __name__ == '__main__':
    main()
