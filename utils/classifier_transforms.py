from torchvision import transforms


# Training transforms
def get_train_transform(CLASS_INFERENCE_SIZE, pretrained):
    train_transform = transforms.Compose([
        transforms.Resize((CLASS_INFERENCE_SIZE, CLASS_INFERENCE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        # transforms.ToTensor(),
        # normalize_transform(pretrained)
    ])
    return train_transform


def get_resize_transform(INFERENCE_SIZE):
    resize_transform = transforms.Compose([transforms.Resize((CLASS_INFERENCE_SIZE, CLASS_INFERENCE_SIZE))])
    return resize_transform


# Validation transforms
def get_valid_transform(CLASS_INFERENCE_SIZE, pretrained):
    valid_transform = transforms.Compose([
        transforms.Resize((CLASS_INFERENCE_SIZE, CLASS_INFERENCE_SIZE)),
        # transforms.ToTensor(),
        # normalize_transform(pretrained)
    ])
    return valid_transform


# Image normalization transforms.
def normalize_transform(pretrained):
    if pretrained:  # Normalization for pre-trained weights.
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
    else:  # Normalization when training from scratch.
        normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    return normalize
