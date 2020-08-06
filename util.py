import torchvision.transforms as transforms

def image_transformer():
    return {
        'train': transforms.Compose(
            [transforms.Resize((224, 224), interpolation=2),
             transforms.RandomHorizontalFlip(),
             transforms.RandomRotation(30),
             transforms.ToTensor(),
             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
        ),
        'test': transforms.Compose(
            [transforms.Resize((224, 224), interpolation=2),
             transforms.ToTensor(),
             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
        )
    }