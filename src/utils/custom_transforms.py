import torch
import torchvision.transforms as transforms


def get_transforms(args):
    train_transform = transforms.Compose([
        transforms.ToTensor(),                      # from HxWxC to CxHxW
        SelfNormalize,                              # normalize
        # ---------- Augmentations -----------
        transforms.RandomCrop(args.patch_size),     # random crop
        transforms.RandomHorizontalFlip(),          # random horizontal flip
        transforms.RandomVerticalFlip(),            # random vertical flip
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),                      # from HxWxC to CxHxW
        SelfNormalize,                              # normalize
        SplitPatchTransform(args.patch_size),       # split image in patches
    ])

    return train_transform, test_transform

class SplitPatchTransform:

    def __init__(self, patch_size):
        self.patch_size = patch_size

    # Split current image into N patches
    def __call__(self, image):
        img_h, img_w = image.shape[1:]
        n_patches_h = img_h // self.patch_size
        n_patches_w = img_w // self.patch_size
        patches = []
        for i in range(n_patches_h):
            for j in range(n_patches_w):
                patch = image[:, i * self.patch_size:(i + 1) * self.patch_size, j * self.patch_size:(j + 1) * self.patch_size]
                patches.append(patch)
        return torch.stack(patches)


def SelfNormalize(image):
    # normalize each channel independently
    for ch in range(image.shape[0]):
        image[ch] = (image[ch] - image[ch].mean()) / image[ch].std()
    return image
