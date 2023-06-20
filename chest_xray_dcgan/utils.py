
import torch
import torch.nn as nn

# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def save_models(generator, discriminator, gen_path, dis_path):
    torch.save(generator.state_dict(), gen_path)
    torch.save(discriminator.state_dict(), dis_path)

def load_models(generator, discriminator,gen_path, dis_path):
    generator.load_state_dict(torch.load(gen_path))
    discriminator.load_state_dict(torch.load(dis_path))
    return generator, discriminator
    