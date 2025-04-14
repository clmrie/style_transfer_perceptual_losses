

from torchvision import transforms

def get_augmentation_transforms(image_size):
    augmentation = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
    ])
    return augmentation

