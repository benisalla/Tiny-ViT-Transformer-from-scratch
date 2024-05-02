from torch.utils.data import Dataset
from PIL import Image

class VitCustomDataset(Dataset):
    """
    A custom dataset class for handling image data specifically tailored for Vision Transformer models. 
    It supports image loading and applying transformations.

    Attributes:
        dataset_dict (list of tuples): A list where each tuple contains an image path and its corresponding label.
        transform (callable, optional): A function/transform that takes in an image and returns a transformed version.
    """
    def __init__(self, dataset_dict, transform=None):
        """
        Initializes the VitCustomDataset with dataset information and optional transformations.

        Args:
            dataset_dict (list of tuples): Each tuple contains the file path of an image and its corresponding label.
            transform (callable, optional): Transform to be applied on a sample. Default is None.
        """
        self.dataset_dict = dataset_dict
        self.transform = transform

    def __len__(self):
        """
        Returns the size of the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return len(self.dataset_dict)

    def __getitem__(self, idx):
        """
        Fetches the image and label based on the index provided and applies the transformation to the image.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            tuple: Tuple containing the transformed image and its label.
        """
        img_path, label = self.dataset_dict[idx]
        image = Image.open(img_path).convert('RGB')  # Ensure image is in RGB format

        if self.transform:
            image = self.transform(image)

        return image, label
