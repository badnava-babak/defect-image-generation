import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import PIL
from datasets import load_dataset
from src.config import ALLOWED_DEFECTS, DATA_PATH
from src.io.dataset_loader import DefectDataset, FewShotDefectDataset
from src.io.utils import load_few_shot_dataset
from src.utils.plots import plot_sample
from torchvision.utils import save_image


def sample(args):
    # Download the dataset if is it not downloaded yet
    if int(args.dl_defect_spectrum):
        load_dataset("DefectSpectrum/Defect_Spectrum", cache_dir="/scratch/b502b586")

    # Load the few shot dataset and prepare the data for fine tunning
    dataset = load_few_shot_dataset("pill", ALLOWED_DEFECTS, int(args.num_sample))

    captions = {s: [] for s in ALLOWED_DEFECTS}

    for i, sample in enumerate(dataset):
        defect_type = sample["label"]
        image = sample["image"]
        caption = f"{sample['defect_desc']}\n{sample['defect_desc']}"

        dir_path = f"{args.out_dir}/ft/{defect_type}"
        file_name = f"{i:03d}.jpg"
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        save_image(image, os.path.join(dir_path, file_name))

        # Save a copy of the files in a seperate folder for textual inversion fine tunning
        if bool(args.ti_dataset):
            dir_path = f"{args.out_dir}/ti/{defect_type}"
            file_name = f"{i:03d}.jpg"
            if not os.path.isdir(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            save_image(image, os.path.join(dir_path, file_name))

        captions[defect_type].append((file_name, caption))

    for k, caption in captions.items():
        if caption:
            dir_path = f"{args.out_dir}/ft/{k}"
            pd.DataFrame(caption, columns=["file_name", "caption"]).to_csv(
                os.path.join(dir_path, f"metadata.csv")
            )


def parse_args():
    p = argparse.ArgumentParser(
        description="Run a simulation or post-process an EpisodeStats pickle "
        "and log the metrics to a CSV file."
    )
    p.add_argument(
        "--num_sample",
        required=False,
        default=5,
        help="Number of instances per each class to sample",
    )

    p.add_argument(
        "--out_dir",
        required=False,
        default="/scratch/b502b586/SiemensEnergy/dataset/",
        help="Number of instances per each class to sample",
    )

    p.add_argument(
        "--ti_dataset",
        required=False,
        default=True,
        help="Wheather to generate a version of the dataset for Textual Inversion",
    )

    p.add_argument(
        "--dl_defect_spectrum",
        required=False,
        default=False,
        help="Download the Defect Spectrum dataset. You don't need to download it everytime, after the first download the dataset is cached.",
    )

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sample(args)
