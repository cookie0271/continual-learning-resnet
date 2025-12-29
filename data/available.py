import os
import urllib.request
from urllib.error import URLError
import zipfile
import shutil  # 新增：用于复制图片
from torchvision import datasets, transforms
from data.manipulate import UnNormalize


class TinyImageNet(datasets.ImageFolder):
    """Dataset wrapper for Tiny-ImageNet (supports optional download)."""

    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    filename = "tiny-imagenet-200.zip"

    def __init__(self, root, train=True, download=False, **kwargs):
        if download:
            self._download(root)

        # If the archive already exists locally (manual download), extract it once.
        self._ensure_local_archive_extracted(root)

        dataset_root = self._resolve_root(root)
        split = "train" if train else "val"  # 测试集用 val 而非 test
        split_root = os.path.join(dataset_root, split)

        # 新增：处理 val 目录，生成类别文件夹
        if split == "val":
            split_root = self._prepare_val_dir(dataset_root)

        if not os.path.isdir(split_root):
            raise RuntimeError(
                f"TinyImageNet data not found at '{split_root}'. "
                "Run with download=True or place the extracted dataset at this path."
            )

        super().__init__(split_root, **kwargs)

    def _resolve_root(self, root):
        direct_root = os.path.join(root, "train")
        nested_root = os.path.join(root, "tiny-imagenet-200", "train")
        if os.path.isdir(direct_root):
            return root
        if os.path.isdir(nested_root):
            return os.path.join(root, "tiny-imagenet-200")
        return root

    def _ensure_local_archive_extracted(self, root):
        """Extract a pre-downloaded archive if present and not yet unpacked."""

        if os.path.isdir(os.path.join(root, "tiny-imagenet-200")) or os.path.isdir(os.path.join(root, "train")):
            return

        candidate_archives = [
            os.path.join(root, self.filename),
            os.path.join(os.path.dirname(root), self.filename),
        ]

        archive_path = next((path for path in candidate_archives if os.path.isfile(path)), None)
        if archive_path is None:
            return

        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(root)

    def _download(self, root):
        os.makedirs(root, exist_ok=True)

        archive_path = os.path.join(root, self.filename)
        extract_root = root

        # Skip download if archive already extracted.
        if os.path.isdir(os.path.join(root, "tiny-imagenet-200")) or os.path.isdir(os.path.join(root, "train")):
            return

        # Download archive if missing.
        if not os.path.isfile(archive_path):
            try:
                urllib.request.urlretrieve(self.url, archive_path)
            except URLError as err:
                raise RuntimeError(
                    "TinyImageNet download failed (network unavailable or URL blocked). "
                    "Place the extracted 'tiny-imagenet-200' folder in '{root}' or "
                    "retry with a working connection."
                    .format(root=root)) from err

        # Extract archive.
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(extract_root)

    # 新增：处理 val 目录，生成 ImageFolder 兼容的类别文件夹
    def _prepare_val_dir(self, dataset_root):
        """
        解析 val_annotations.txt，为 val 目录生成类别子文件夹，适配 ImageFolder 加载逻辑
        """
        val_dir = os.path.join(dataset_root, "val")
        val_images_dir = os.path.join(val_dir, "images")
        val_anno_file = os.path.join(val_dir, "val_annotations.txt")
        val_processed_dir = os.path.join(val_dir, "val_processed")  # 处理后的 val 目录

        # 若已生成，直接返回（避免重复处理）
        if os.path.exists(val_processed_dir):
            return val_processed_dir

        # 创建处理后的根目录
        os.makedirs(val_processed_dir, exist_ok=True)

        # 读取标注文件，按类别分类图片
        with open(val_anno_file, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if not line:
                    continue
                # 标注格式：文件名 类别ID 其他（忽略）
                parts = line.split()
                img_name, class_id = parts[0], parts[1]

                # 创建类别子文件夹
                class_dir = os.path.join(val_processed_dir, class_id)
                os.makedirs(class_dir, exist_ok=True)

                # 复制图片到对应类别文件夹
                src_img_path = os.path.join(val_images_dir, img_name)
                dst_img_path = os.path.join(class_dir, img_name)
                if os.path.exists(src_img_path) and not os.path.exists(dst_img_path):
                    shutil.copy(src_img_path, dst_img_path)

        return val_processed_dir


# specify available data-sets.
AVAILABLE_DATASETS = {
    'MNIST': datasets.MNIST,
    'CIFAR100': datasets.CIFAR100,
    'CIFAR10': datasets.CIFAR10,
    'TinyImageNet': TinyImageNet,
}

# specify available transforms.
AVAILABLE_TRANSFORMS = {
    'MNIST': [
        transforms.ToTensor(),
    ],
    'MNIST32': [
        transforms.Pad(2),
        transforms.ToTensor(),
    ],
    'CIFAR10': [
        transforms.ToTensor(),
    ],
    'CIFAR100': [
        transforms.ToTensor(),
    ],
    'TinyImageNet': [
        transforms.Resize(64),
        transforms.ToTensor(),
    ],
    'CIFAR10_norm': [
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ],
    'CIFAR100_norm': [
        transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2761])
    ],
    'TinyImageNet_norm': [
        transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2770, 0.2691, 0.2821])
    ],
    'CIFAR10_denorm': UnNormalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
    'CIFAR100_denorm': UnNormalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2761]),
    'TinyImageNet_denorm': UnNormalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2770, 0.2691, 0.2821]),
    'augment_from_tensor': [
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4, padding_mode='symmetric'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ],
    'augment': [
        transforms.RandomCrop(32, padding=4, padding_mode='symmetric'),
        transforms.RandomHorizontalFlip(),
    ],
    'TinyImageNet_augment': [
        transforms.RandomCrop(64, padding=4, padding_mode='symmetric'),
        transforms.RandomHorizontalFlip(),
    ],
}

# specify configurations of available data-sets.
DATASET_CONFIGS = {
    'MNIST': {'size': 28, 'channels': 1, 'classes': 10},
    'MNIST32': {'size': 32, 'channels': 1, 'classes': 10},
    'CIFAR10': {'size': 32, 'channels': 3, 'classes': 10},
    'CIFAR100': {'size': 32, 'channels': 3, 'classes': 100},
    'TinyImageNet': {'size': 64, 'channels': 3, 'classes': 200},
}