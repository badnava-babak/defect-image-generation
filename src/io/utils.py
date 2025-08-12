from typing import List

from torchvision import transforms

from src.config import DATA_PATH
from src.io.dataset_loader import DefectDataset, FewShotDefectDataset


def load_few_shot_dataset(class_name: str, allowed_defects: List[str], num_samples: int=10) -> "DefectDataset":
    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    dataset = FewShotDefectDataset(
        root_dir=DATA_PATH,
        class_name=class_name,
        transform=image_transform,
        mask_transform=mask_transform,
        allowed_defects=allowed_defects,
        metadata_file="captions.csv",  # optional
        return_caption=True,
        nb_samples=num_samples
    )
    return dataset
