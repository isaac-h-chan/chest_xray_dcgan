import torchvision
from torchvision import datasets, transforms
import torch.utils.data

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def load_data(dir, batch_size, num_workers):

    dataset = datasets.ImageFolder(
        root=dir, 
        transform=transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            #AddGaussianNoise(0, 0.05),
            transforms.Normalize((0.5), (0.5))]))
    
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers)

    return dataloader