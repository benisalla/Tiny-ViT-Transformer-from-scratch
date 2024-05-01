from torch.utils.data import Dataset
from PIL import Image

class VitCustomDataset(Dataset):
    def __init__(self, dataset_dict, transform=None):
        self.dataset_dict = dataset_dict
        self.transform = transform

    def __len__(self):
        return len(self.dataset_dict)

    def __getitem__(self, idx):
        img_path, label = self.dataset_dict[idx]
        image = Image.open(img_path).convert('RGB') # to make sure
        if self.transform:
            image = self.transform(image)
        return image, label