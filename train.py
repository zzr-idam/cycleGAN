import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import torchvision
import torchvision.transforms as Transforms
from torch.utils.tensorboard import SummaryWriter
import os
from torch.utils.data import Dataset, dataloader
import time
from dataset import HorseZebraDataset
from discriminator import Discriminator
from generator import Generator
from torchvision.utils import save_image

lr = 2e-5
batch_size = 4
epoch = 100
horse_root = "/home/wenwen/dataset/UIEB/reference-890/"
#horse_root = "/home/wenwen/dataset/4Kdehaze/test/inputs/"
zebra_root = "/home/wenwen/dataset/UIEB/raw-890/"
image_channels = 3

D_H_save_path = '\logs\D_H'
D_Z_save_path = '\logs\D_Z'
G_H_save_path = '\logs\G_H'
G_Z_save_path = '\logs\G_Z'

transforms = Transforms.Compose(
    [
        Transforms.Resize(384),
        #Transforms.RandomVerticalFlip(p=0.5),
        Transforms.ToTensor()
    ]
)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

dataset = HorseZebraDataset(horse_root=horse_root, zebra_root=zebra_root, transforms=transforms)
mydataloader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True)

D_H = Discriminator(in_channels=image_channels).to(device)
D_Z = Discriminator(in_channels=image_channels).to(device)

G_H = Generator(img_channels=image_channels).to(device)
G_Z = Generator(img_channels=image_channels).to(device)

opt_dic = torch.optim.Adam(list(D_H.parameters()) + list(D_Z.parameters()), lr=lr, betas=(0.5, 0.999))
opt_gen = torch.optim.Adam(list(G_H.parameters()) + list(G_Z.parameters()), lr=lr, betas=(0.5, 0.999))

cretrion = nn.MSELoss()

loss_f = nn.L1Loss()

writer_horse_path = "fake_horse"
writer_zebra_path = "fake_zebra"
writer_horse = SummaryWriter(writer_horse_path)
writer_zebra = SummaryWriter(writer_zebra_path)

D_H.train()
D_Z.train()
G_H.train()
G_Z.train()
step = 0
for i in range(epoch):
    for index, data in enumerate(mydataloader, 1):
        horse_img, zebra_img = data
        horse_img = horse_img.to(device)
        zebra_img = zebra_img.to(device)
        # horse
        fake_horse = G_H(zebra_img)
        #if epoch % 1 == 0 and index % 10 == 0:
            #save_image(fake_horse, 'image/fake_horse_{}.jpg'.format(i + 1))
        D_H_real = D_H(horse_img)
        D_H_fake = D_H(fake_horse.detach())

        D_H_real_loss = cretrion(D_H_real, torch.ones_like(D_H_real))
        D_H_fake_loss = cretrion(D_H_fake, torch.zeros_like(D_H_fake))
        D_H_loss = D_H_real_loss + D_H_fake_loss


       # Zebra
        fake_zebra = G_Z(horse_img)
        if epoch % 1 == 0 and index % 10 == 0:
            save_image(fake_zebra,'image/fake_zebra{}.jpg'.format(i + 1))
            save_image(horse_img,'image/horse{}.jpg'.format(i + 1))
        D_Z_real = D_Z(zebra_img)
        D_Z_fake = D_Z(fake_zebra.detach())
        D_Z_real_loss = cretrion(D_Z_real, torch.ones_like(D_Z_real))
        D_Z_fake_loss = cretrion(D_Z_fake, torch.zeros_like(D_Z_fake))
        D_Z_loss = D_Z_real_loss + D_Z_fake_loss

        # 总损失
        D_loss = (D_H_loss + D_Z_loss) / 2
        opt_dic.zero_grad()
        D_loss.backward()
        opt_dic.step()


        # adversarial loss for both generators
        D_H_fake = D_H(fake_horse)
        D_Z_fake = D_Z(fake_zebra)
        loss_G_H = cretrion(D_H_fake, torch.ones_like(D_H_fake))
        loss_G_Z = cretrion(D_Z_fake, torch.ones_like(D_Z_fake))

        # cycle loss
        cycle_zebra = G_Z(fake_horse)
        cycle_horse = G_H(fake_zebra)
        #print(zebra_img)
        #print(cycle_zebra)
        cycle_zebra_loss = loss_f(zebra_img, cycle_zebra)
        cycle_horse_loss = loss_f(horse_img, cycle_horse)

        # total loss
        G_loss = (
            loss_G_Z
            + loss_G_H
            + 10 * cycle_zebra_loss
            + 10 * cycle_horse_loss
        )

        opt_gen.zero_grad()
        G_loss.backward()
        opt_gen.step()

        if index % 1 == 0:
            with torch.no_grad():
                D_H.eval()
                D_Z.eval()
                G_H.eval()
                G_Z.eval()
                image_grad_horse = torchvision.utils.make_grid(
                    fake_zebra, normalize=True
                )
                writer_zebra.add_image("fake_zebra", image_grad_horse, global_step=step)

                step+=1
                D_H.train()
                D_Z.train()
                G_H.train()
                G_Z.train()
        
        print(G_loss)
        print("[%d/epoch]" %(i))
    torch.save(G_Z.state_dict(), 'model' + '/G_Z_{}.pth'.format(i))
