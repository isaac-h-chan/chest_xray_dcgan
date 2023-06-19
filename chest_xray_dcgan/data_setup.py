import torchvision
from torchvision import datasets, transforms
import torch.utils.data


def load_data(dir, batch_size, num_workers):

    dataset = datasets.ImageFolder(
        root=dir, 
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))]))
    
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers)

    return dataloader