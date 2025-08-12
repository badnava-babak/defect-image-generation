import argparse

import pandas as pd
from torch_fidelity import calculate_metrics


def evaluate(args):
    metrics = calculate_metrics(
        input1=args.orig_data_dir,  # folder of real images for this defect
        input2=args.synthetic_data_dir,  # folder of generated images
        cuda=True,
        kid_subset_size=10,
        isc=True, fid=True, kid=True, prc=True, verbose=True
    )
    df = pd.DataFrame(metrics, index=list(range(len(metrics))))
    print(df)  # dict with 'frechet_inception_distance', 'kernel_inception_distance_mean', 'precision', 'recall'
    df.to_csv(f"{args.out_dir}/{args.file_name}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Run a evaluation on quality of generated images."
    )

    p.add_argument("--out_dir", required=False, default='/home/b502b586/scratch/SiemensEnergy/evaluations',
                   help="Number of instances per each class to sample")

    p.add_argument("--orig_data_dir", required=False,
                   default='/home/b502b586/scratch/SiemensEnergy/dataset/ti/color',
                   help="Directory of the original images")

    p.add_argument("--synthetic_data_dir", required=False,
                   default='/home/b502b586/scratch/SiemensEnergy/dataset/synthetic/color/img',
                   help="Directory of the synthetic generated images")

    p.add_argument("--file_name", required=False, default='results.csv',
                   help="Name of the file in which the results of the evaluation will be stored.")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
