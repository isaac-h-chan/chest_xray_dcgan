
import torch

def train(generator, discriminator, train_dataloader, loss_fn, dis_optimizer, gen_optimizer, device="cpu"):

    dis_fake_loss_total = 0
    dis_real_loss_total = 0
    gen_loss_total = 0

    for batch, real_images in enumerate(train_dataloader):
        # TRAINING DISCRIMINATOR
        # Creating labels tensor for real_images and assigning every label as 1 for real
        discriminator.zero_grad()
        batch_size = real_images[0].size(0)
        labels = torch.full((batch_size,), 1, dtype=torch.float, device=device)

        # Using real images from dataset
        preds_on_real = discriminator(real_images[0].to(device)).view(-1)

        loss_on_real = loss_fn(preds_on_real, labels)
        loss_on_real.backward()

        dis_real_loss_total += loss_on_real
        # Using fake images from generator
        noise = torch.randn(batch_size, 128, 1, 1, device=device)

        # Generating fake images from generator and filling labels with 0's for fake
        fake_images = generator(noise)
        labels.fill_(0)

        # fake_images are detached so autograd does not clear it from the graph (search retains_graph for more details)
        preds_on_fake = discriminator(fake_images.detach()).view(-1)

        loss_on_fake = loss_fn(preds_on_fake, labels)
        loss_on_fake.backward()
        dis_fake_loss_total += loss_on_fake

        dis_optimizer.step()


        # TRAINING GENERATOR
        generator.zero_grad()

        # Generator's objective is to generate images that are classified as real by discriminator, thus its all its labels are 1
        labels.fill_(1)

        preds_on_fake = discriminator(fake_images).view(-1)

        gen_loss = loss_fn(preds_on_fake, labels)
        gen_loss.backward()
        gen_loss_total += gen_loss

        gen_optimizer.step()

        print(f"fake loss: {loss_on_fake} | real loss: {loss_on_real} | gen loss: {gen_loss}")

    return dis_fake_loss_total/len(train_dataloader), dis_real_loss_total/len(train_dataloader), gen_loss_total/len(train_dataloader)


#def test(generator, discriminator, test_dataloader, batch_size, loss_fn, dis_optimizer, gen_optimizer, device="cpu"):


