import os
from typing import Optional, List, Callable, Tuple
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

import random


class DefectDataset(Dataset):
    def __init__(
            self,
            root_dir: str,
            class_name: str,
            image_dir: str = "image",
            mask_dir: str = "mask",
            rgb_mask_dir: str = "rgb_mask",
            transform: Optional[Callable] = None,
            mask_transform: Optional[Callable] = None,
            allowed_defects: Optional[List[str]] = None,
            metadata_file: Optional[str] = None,
            return_caption: bool = False
    ):
        """
        Args:
            root_dir: Root directory of the dataset.
            image_dir: Sub-directory for images.
            mask_dir: Sub-directory for segmentation masks.
            transform: Transformations to apply to the images.
            mask_transform: Transformations to apply to the masks.
            allowed_defects: Optional list of class labels to filter.
            metadata_file: Optional CSV or JSON for captions or class filtering.
            return_caption: If True, return captions from metadata.
        """
        self.root_dir = root_dir
        self.class_name = class_name
        self.image_dir = os.path.join(root_dir, class_name, image_dir)
        self.mask_dir = os.path.join(root_dir, class_name, mask_dir)
        self.rgb_mask_dir = os.path.join(root_dir, class_name, rgb_mask_dir)
        self.transform = transform
        self.mask_transform = mask_transform
        self.return_caption = return_caption

        self.samples = self._load_samples(metadata_file, allowed_defects)

    def _load_samples(
            self, metadata_file: Optional[str], allowed_defects: Optional[List[str]]
    ) -> List[dict]:
        """
        Loads sample paths and optionally filters based on class metadata.
        Returns a list of dicts: {image_path, mask_path, class, caption}
        """
        samples = []

        if metadata_file:
            import pandas as pd

            metadata_path = os.path.join(self.root_dir, metadata_file)
            meta_df = pd.read_csv(metadata_path)

            for _, row in meta_df.iterrows():
                class_name, defect_name, file_name = row['Path'].split('/')
                if class_name != self.class_name: continue
                if defect_name not in allowed_defects: continue
                defect_path = os.path.join(self.image_dir, defect_name)
                image_path = os.path.join(defect_path, file_name)
                mask_path = os.path.join(self.mask_dir, defect_name, file_name.replace('.', '_mask.'))
                rgb_mask_path = os.path.join(self.rgb_mask_dir, defect_name, file_name.replace('.', '_rgb_mask.'))
                samples.append({
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "rgb_mask_path": rgb_mask_path,
                    "class": defect_name,
                    "object_desc": row['object description'],
                    "defect_desc": row['defect description']
                })
        else:
            # Fallback: no metadata, match image and mask by filename
            for defect_name in allowed_defects:
                defect_path = os.path.join(self.image_dir, defect_name)
                for file_name in os.listdir(defect_path):
                    if not file_name.lower().endswith((".jpg", ".png", ".jpeg")):
                        continue
                    image_path = os.path.join(defect_path, file_name)
                    mask_path = os.path.join(self.mask_dir, defect_name, file_name.replace('.', '_mask.'))
                    rgb_mask_path = os.path.join(self.rgb_mask_dir, defect_name, file_name.replace('.', '_rgb_mask.'))
                    samples.append({
                        "image_path": image_path,
                        "mask_path": mask_path,
                        "rgb_mask_path": rgb_mask_path,
                        "class": defect_name,
                        "object_desc": '',
                        "defect_desc": ''
                    })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple:
        sample = self.samples[idx]

        image = Image.open(sample["image_path"]).convert("RGB")
        mask = Image.open(sample["mask_path"]).convert("L")  # binary mask
        rgb_mask = Image.open(sample["rgb_mask_path"]).convert("RGB")

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
            rgb_mask = self.mask_transform(rgb_mask)

        output = {
            "image": image,
            "mask": mask,
            "rgb_mask": rgb_mask,
        }

        if sample["class"]:
            output["label"] = sample["class"]
        if self.return_caption:
            output["object_desc"] = sample["object_desc"]
            output["defect_desc"] = sample["defect_desc"]

        return output


# Few-shot filter: 10 samples of each damage type in 'pill'
class FewShotDefectDataset(DefectDataset):

    def __init__(self, nb_samples: int = 10, *args, **kwargs):
        self.nb_samples = nb_samples
        super(FewShotDefectDataset, self).__init__(*args, **kwargs)

    def _load_samples(self, metadata_file, allowed_classes):
        all_samples = super()._load_samples(metadata_file, allowed_classes)

        # Few-shot filter
        few_shot_samples = []
        damage_type_groups = {}

        for sample in all_samples:
            key = (sample['class'])  # group by class + damage type
            damage_type_groups.setdefault(key, []).append(sample)

        for group in damage_type_groups.values():
            few_shot_samples.extend(random.sample(group, min(self.nb_samples, len(group))))

        return few_shot_samples



