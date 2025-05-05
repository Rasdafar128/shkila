# Работа с данными
import numpy as np

# Torch
import torch
from torch.utils.data import Dataset, random_split

# Визуализация
import matplotlib.pyplot as plt

# Остальное
import os
import random
from PIL import Image
from tqdm import tqdm

# Config
from config import *


def set_all_seeds(seed=42):
    # Устанавливаем seed для встроенного генератора Python
    random.seed(seed)
    # Устанавливаем seed для хэш-функции Python (опция для контроля поведения хэшей)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Устанавливаем seed для NumPy
    np.random.seed(seed)

    # Устанавливаем seed для PyTorch
    torch.manual_seed(seed)
    # Устанавливаем seed для генератора на CUDA
    torch.cuda.manual_seed(seed)
    # Отключаем недетерминированное поведение в алгоритмах CUDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def denormalize(image_tensor):
    # Преобразуем mean и std в тензоры и переносим их на то же устройство, что и image
    tensor_mean = torch.tensor(mean).view(-1, 1, 1).to(image_tensor.device)
    tensor_std = torch.tensor(std).view(-1, 1, 1).to(image_tensor.device)

    # Денормализация: (тензор * std) + mean
    denormalize_image = image_tensor * tensor_std + tensor_mean

    # Преобразуем в диапазон [0, 255] и к типу uint8
    return (denormalize_image * 255).clamp(0, 255).byte()





def train_val_split(dataset, val_size=0.2, seed=42):

    train_size = int((1 - val_size) * len(dataset))
    val_size = len(dataset) - train_size
    generator1 = torch.Generator().manual_seed(seed)

    train_set, valid_set = random_split(dataset, [train_size, val_size], generator1)
    return train_set, valid_set

def show_images(dataset, k=5):
    """
    Визуализация изображений и соответствующих масок из датасета.
    
    :param dataset: PyTorch Dataset, содержащий изображения и маски
    :param k: Количество изображений для визуализации
    """
    # Убедимся, что k не превышает длину датасета
    k = min(k, len(dataset))
    
    # Устанавливаем размер сетки
    fig, axs = plt.subplots(k, 2, figsize=(10, 5 * k))
    
    for i in range(k):
        # Получаем i-ый элемент из датасета
        image, mask = dataset[i]
        
        # Преобразуем изображения и маски в numpy (если это тензоры)
        if hasattr(image, 'numpy'):
            image = image.numpy()
        if hasattr(mask, 'numpy'):
            mask = mask.numpy()
        
        # Убираем канал, если он есть (для изображений с формой (1, H, W))
        if image.ndim == 3 and image.shape[0] == 1:
            image = image.squeeze(0)  # Убираем первый канал
        elif image.ndim == 3 and image.shape[0] == 3:
            image = image.transpose(1, 2, 0)  # Переставляем каналы (C, H, W) → (H, W, C)
        
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)  # Убираем первый канал
        
        # Отображаем изображение
        axs[i, 0].imshow(image, cmap='gray' if image.ndim == 2 else None)
        axs[i, 0].set_title(f"Image {i + 1}")
        axs[i, 0].axis("off")
        
        # Отображаем маску
        axs[i, 1].imshow(mask, cmap='gray')
        axs[i, 1].set_title(f"Mask {i + 1}")
        axs[i, 1].axis("off")
    
    plt.tight_layout()
    plt.show()




def label_encoding(dataset):
    classes = []
    for x in tqdm(dataset):
        for v in x[1].unique():
            if v not in classes:
                classes.append(v)
    return classes



class ImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Считываем изображение
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        # Применяем трансформации
        image = self.transform(image)

        return image.to(device)


class ImageSegmentationDataset(ImageDataset):
    def __init__(self, image_paths, mask_paths, image_transform, mask_transform):
        super().__init__(image_paths, image_transform)

        self.mask_transform = mask_transform
        self.mask_paths = mask_paths

    def __getitem__(self, idx):
        # Считываем маску
        mask_path = self.mask_paths[idx]
        mask = Image.open(mask_path).convert("RGB")

        # Применяем трансформации
        mask = self.mask_transform(mask)

        return super().__getitem__(idx), mask.to(device)
