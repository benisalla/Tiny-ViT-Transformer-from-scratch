import os
from torchvision import transforms
from torchvision.transforms import InterpolationMode, AutoAugment, AutoAugmentPolicy
from core.config import Config
from data.vit_custom_dataset import VitCustomDataset
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    """
    Prepares datasets and data loaders for a machine learning model based on configurations provided.

    Attributes:
        train_transforms (transforms.Compose): Transformations applied to training images.
        val_transforms (transforms.Compose): Transformations applied to validation and test images.
    """
    def __init__(self, config: Config):
        """
        Initializes the DataPreprocessor with data directory, batch size, and transformations.
        """
        self.data_dir = config.data_dir
        self.max_img_cls = config.max_img_cls
        self.max_cls = config.max_cls
        self.classes = config.classes[:self.max_cls]
        self.batch_size = config.batch_size
        self.valid_size = config.valid_size
        self.num_workers = config.num_workers
        self.pin_memory = config.pin_memory
        self.shuffle = config.shuffle
        self.im_size = config.im_size

        # Training data transformations
        self.train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(self.im_size, scale=(0.5, 1.0), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10, interpolation=InterpolationMode.BICUBIC),
            AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Validation and test data transformations
        self.val_transforms = transforms.Compose([
            transforms.Resize((self.im_size, self.im_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def filter_images_per_class(self):
        """
        Filters images from directories based on class limits and organizes them for dataset creation.
        """
        class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        filtered_data = []
        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            if os.path.isdir(class_dir):
                images = os.listdir(class_dir)[:self.max_img_cls]
                for img_name in images:
                    img_path = os.path.join(class_dir, img_name)
                    item = (img_path, class_to_idx[class_name])
                    filtered_data.append(item)
        return filtered_data

    def split_datasets(self):
        """
        Splits the data into training, validation, and testing datasets using stratified sampling.
        """
        dataset_dict = self.filter_images_per_class()
        total_size = len(dataset_dict)
        train_size = int((1 - 2 * self.valid_size) * total_size)
        val_size = int(self.valid_size * total_size)
        test_size = total_size - train_size - val_size

        labels = [label for _, label in dataset_dict]
        train_idx, temp_idx, _, temp_labels = train_test_split(
            range(total_size), labels, stratify=labels, test_size=(val_size + test_size)
        )
        val_idx, test_idx = train_test_split(
            temp_idx, stratify=temp_labels, test_size=test_size/(val_size + test_size)
        )

        train_dataset = VitCustomDataset([dataset_dict[i] for i in train_idx], transform=self.train_transforms)
        val_dataset = VitCustomDataset([dataset_dict[i] for i in val_idx], transform=self.val_transforms)
        test_dataset = VitCustomDataset([dataset_dict[i] for i in test_idx], transform=self.val_transforms)

        return train_dataset, val_dataset, test_dataset

    def create_dataloaders(self):
        """
        Creates data loaders for the training, validation, and test datasets.
        """
        train_data, val_data, test_data = self.split_datasets()
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers, pin_memory=self.pin_memory)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)

        return train_loader, val_loader, test_loader
