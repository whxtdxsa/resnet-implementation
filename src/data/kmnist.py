from torchvision import datasets, transforms
import os
base_dir = os.path.dirname(os.path.abspath(__file__))
img_height = img_width = 224

def load_kmnist():
    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.KMNIST(
        root=base_dir,  
        train=True, 
        download=True,  
        transform=transform
    )

    test_dataset = datasets.KMNIST(
        root=base_dir,
        train=False,
        download=True,
        transform=transform
    )
    return train_dataset, test_dataset
