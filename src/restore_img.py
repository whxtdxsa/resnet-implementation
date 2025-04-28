from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage

dataset = datasets.KMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

to_pil = ToPILImage()

tensor_image, label = dataset[1]  # (1, 28, 28) Tensor, 정답 레이블

pil_image = to_pil(tensor_image)

pil_image.show()

pil_image.save(f"kmnist_label{label}.png")
