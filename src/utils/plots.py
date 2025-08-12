import matplotlib.pyplot as plt


def plot_sample(sample):
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(sample['image'].T)
    ax[0].title.set_text("Original Image")
    ax[1].imshow(sample['mask'].T)
    ax[1].title.set_text("Image Mask")
    ax[2].imshow(sample['rgb_mask'].T)
    ax[2].title.set_text("RGB Mask")
    # fig.suptitle(f"{sample['object_desc']}\n{sample['defect_desc']}")
    fig.suptitle(f"{sample['object_desc']}")
    plt.tight_layout()
    plt.show()
