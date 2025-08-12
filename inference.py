import argparse

import torch
from diffusers import StableDiffusionXLPipeline

from src.config import ALLOWED_DEFECTS
from src.io.utils import load_few_shot_dataset


def generate_samples(args, dataset, pipe, img_postfix):
    """
    Generate exactly `args.num_samples` images for `args.defect_type` from `dataset` using `pipe`.
    Saves PNG + sidecar TXT (prompt/object_desc). If a mask exists in the sample, use inpaint call.
    """
    from pathlib import Path
    from PIL import Image
    from tqdm import tqdm
    import torch
    import numpy as np

    def _to_pil(x):
        """Best-effort convert (PIL / torch tensor CHW / numpy HWC / path) -> PIL.Image."""
        if isinstance(x, Image.Image):
            return x
        if torch.is_tensor(x):
            t = x.detach().cpu()
            if t.ndim == 3 and t.shape[0] in (1, 3):
                t = t.clamp(0, 1).mul(255).byte()
                if t.shape[0] == 1:
                    t = t.repeat(3, 1, 1)
                t = t.permute(1, 2, 0).numpy()
                return Image.fromarray(t)
            raise ValueError("Unsupported tensor shape for image.")
        if isinstance(x, np.ndarray):
            arr = x
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 1)
                arr = (arr * 255).astype(np.uint8)
            return Image.fromarray(arr)
        # assume path-like
        return Image.open(x)

    defect = str(args.defect_type)
    out_root = Path(args.out_dir)
    out_dir = out_root / defect
    out_dir.mkdir(parents=True, exist_ok=True)

    img_dir = out_dir / "img"
    caption_dir = out_dir / "captions"
    img_dir.mkdir(parents=True, exist_ok=True)
    caption_dir.mkdir(parents=True, exist_ok=True)

    produced = 0
    idx = 0
    total = len(dataset)

    pbar = tqdm(total=int(args.num_samples), desc=f"Generating '{defect}'")

    while produced < int(args.num_samples) and idx < total:
        sample = dataset[idx]
        idx += 1

        if sample.get("label") != defect:
            continue

        object_desc = sample.get("object_desc", "")
        defect_desc = sample.get("defect_desc", "")
        prompt = f"{object_desc}\n{defect_desc}"

        init_image = sample.get("image", None)
        if init_image is None:
            continue
        init_pil = _to_pil(init_image)

        # Decide if we should inpaint (mask present) or do plain txt2img
        mask = sample.get("rgb_mask", None)
        mask_pil = _to_pil(mask)

        gen_kwargs = dict(prompt=prompt,
                          num_inference_steps=250,
                          guidance_scale=1.,
                          image=init_pil,
                          mask_image=mask_pil,
                          strength=1.
                          )

        try:
            result = pipe(**gen_kwargs).images[0]
        except Exception as e:
            print(f"[WARN] Generation failed at idx={idx - 1}: {e}")
            continue

        produced += 1
        pbar.update(1)

        # Save outputs and sidecars
        stem = f"{produced:03d}{img_postfix}"
        out_png = out_dir / "img" / f"{stem}.png"
        out_txt = out_dir / "captions" / f"{produced:03d}.txt"  # keep your original naming for txt

        result.save(out_png)

        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(prompt.strip())

    pbar.close()
    if produced < int(args.num_samples):
        print(f"[WARN] Only produced {produced} samples (requested {args.num_samples}). Not enough matching items.")


def inference(args):
    dataset = load_few_shot_dataset("pill", ALLOWED_DEFECTS)

    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        safety_checker=None,
        cache_dir='/scratch/b502b586'
    ).to("cuda")

    lora_weights = f"{args.lora_dir}/ft/{args.defect_type}-defect-pill-{args.lora_samples}-samples/"
    ti_weights = f"{args.lora_dir}/ti/{args.defect_type}-defect-pill-{args.lora_samples}-samples/"

    print('Loading LORA weights!')
    pipe.load_lora_weights(lora_weights)

    if bool(args.enable_ti):
        print('Loading Textual Inversion weights!')
        pipe.load_textual_inversion(ti_weights, token=f"<sks_{args.defect_type}_defect>")

    generate_samples(args, dataset, pipe, '')


def parse_args():
    p = argparse.ArgumentParser(
        description="Run a simulation or post-process an EpisodeStats pickle "
                    "and log the metrics to a CSV file."
    )
    p.add_argument("--num_samples", required=False, default=10,
                   help="Number of samples to generate")

    p.add_argument("--out_dir", required=False,
                   default='/home/b502b586/scratch/SiemensEnergy/dataset/synthetic',
                   help="Directory in which the generated images will be saved")

    p.add_argument("--lora_dir", required=False,
                   default='/home/b502b586/scratch/SiemensEnergy/lora-weights',
                   help="Directory in which LORA weights are saved")

    p.add_argument("--lora_samples", required=False,
                   default=20,
                   help="Number of samples used to fine tune LORA weights")

    p.add_argument("--enable_ti", required=False,
                   default=True,
                   help="If True, loads textual inversion LORA weights")

    p.add_argument("--defect_type", required=False, default='color',
                   help="Defect type that you want to generate images: [color, scratch]")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    inference(args)
