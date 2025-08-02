import os
import random
import numpy as np
import cv2
import torch

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class RandomHFlip:
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, img):
        if random.random() < self.p:
            return cv2.flip(img, 1)
        return img

class RandomVFlip:
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, img):
        if random.random() < self.p:
            return cv2.flip(img, 0)
        return img

class ResizeAndCrop:
    """随机放大到指定范围，然后随机裁剪到目标尺寸"""
    def __init__(self, crop_size, rel_scale_range=(1.0, 1.5), p=1.0):
        # crop_size: (width, height)
        self.crop_size = crop_size
        self.rel_scale_range = rel_scale_range
        self.p = p
    
    def __call__(self, img):
        if random.random() > self.p:
            # 如果不启用变换，直接resize到目标尺寸
            return cv2.resize(img, self.crop_size, interpolation=cv2.INTER_LINEAR)
        
        h, w = img.shape[:2]
        target_w, target_h = self.crop_size

        # 随机选择放大倍数
        scale = random.uniform(self.rel_scale_range[0], self.rel_scale_range[1])
        scale = max(scale, 1.0)
        scale = max(self.crop_size[0] / w * scale, self.crop_size[1] / h * scale)

        # 计算放大后的尺寸
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # 随机选择插值方法并放大图像
        interpolation = random.choice([cv2.INTER_NEAREST, cv2.INTER_LINEAR])
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
        
        # 随机选择裁剪位置
        max_x = new_w - target_w
        max_y = new_h - target_h
        
        start_x = random.randint(0, max_x) if max_x > 0 else 0
        start_y = random.randint(0, max_y) if max_y > 0 else 0
        
        # 执行裁剪
        cropped_img = resized_img[start_y:start_y + target_h, start_x:start_x + target_w]
        
        return cropped_img

class ToTensor:
    """将numpy数组转换为torch tensor，并从HWC转为CHW"""
    def __call__(self, img):
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img)

class Normalize:
    """标准化tensor"""
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32).reshape(-1, 1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(-1, 1, 1)
    
    def __call__(self, tensor):
        # 转换为numpy进行计算，然后转回tensor
        img_np = tensor.numpy()
        img_np = (img_np - self.mean) / self.std
        return torch.from_numpy(img_np)

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

class CofiredDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_root):
        super(CofiredDataset, self).__init__()
        self.images = []
        for root, _, files in os.walk(dataset_root):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".bmp"):
                    self.images.append(os.path.join(root, file))
        print(f"Found {len(self.images)} images in {dataset_root}")
        
        self.transform_img = Compose([
            ResizeAndCrop((224, 224), rel_scale_range=(1.0, 1.8), p=0),
            # RandomHFlip(0.5),
            RandomVFlip(0.5),
            ToTensor(),
            Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        self.imagesize = (3, 224, 224)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]

        # Load and transform image
        image = cv2.imread(image_path)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform_img(image)

        # Since all samples are normal, create zero mask
        mask = torch.zeros([1, *self.imagesize[1:]])
        if "abnormal" in image_path.lower():
            anomaly_type = "dahuan"
            is_anomaly = 1
        else:
            anomaly_type = "good"
            is_anomaly = 0
        return {
            "image": image,
            "mask": mask,
            "classname": "cofired",
            "anomaly": anomaly_type,
            "is_anomaly": is_anomaly,
            "image_name": os.path.basename(image_path),
            "image_path": image_path,
        }

if __name__ == '__main__':
    da = CofiredDataset(
        dataset_root="/home/kohill/aoidev/datasets/中瓷/3D下料/train/cropped"
    )
    import matplotlib.pyplot as plt
    plt.ion()
    fig, ax = plt.subplots(1, 1, squeeze=False)
    ax = ax.reshape(-1)
    plt.show()
    for item in da:
        image_tensor = item["image"].data.numpy()
        image_tensor = np.transpose(image_tensor, (1, 2, 0))
        image_tensor = image_tensor * np.array(IMAGENET_STD)[np.newaxis, np.newaxis] + np.array(IMAGENET_MEAN)[np.newaxis, np.newaxis]
        image_tensor = np.clip(image_tensor, 0, 1)
        ax[0].imshow(image_tensor)
        plt.waitforbuttonpress()