import torch
import timeit
import torchvision
import datetime
import os
import sys
import pickle

import matplotlib.animation as animation
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sc

from torchvision import transforms, datasets
from torch import nn
from torch.optim import Adam
from torch.autograd import Variable


ngpu=1

DEVICE = torch.device("cuda" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

print(f'Running on {DEVICE}')


MODEL_NAME = 'Conditional-DCGAN'

IMAGE_SIZE = 64
BATCH_SIZE = 100

def images_to_vector(images):
    return images.view(images.size(0), (IMAGE_SIZE*IMAGE_SIZE))

def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 1, IMAGE_SIZE, IMAGE_SIZE)


dataroot =  './dataset'


dataset = datasets.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Grayscale(num_output_channels=1),
                               transforms.Resize(IMAGE_SIZE),
                               transforms.CenterCrop(IMAGE_SIZE),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,)),
                           ]))
# Create the dataloader
DATA_LOADER = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                         shuffle=True, num_workers=0)


NUM_BATCHES = len(DATA_LOADER)


DATAITER = iter(DATA_LOADER)
IMAGES, LABELS = DATAITER.next()








def to_onehot(x, num_classes=5):
    assert isinstance(x, int) or isinstance(x, (torch.LongTensor, torch.cuda.LongTensor))
    if isinstance(x, int):
        c = torch.zeros(1, num_classes).long()
        c[0][x] = 1
    else:
        x = x.cpu()
        c = torch.LongTensor(x.size(0), num_classes)
        c.zero_()
        c.scatter_(1, x, 1) # dim, index, src value
    return c

def get_sample_image(G, n_noise=100):
    """
        save sample 100 images
    """
    img = np.zeros([280, 280])
    for j in range(10):
        c = torch.zeros([10, 10]).to(DEVICE)
        c[:, j] = 1
        z = torch.randn(10, n_noise).to(DEVICE)
        y_hat = G(z,c).view(10, 28, 28)
        result = y_hat.cpu().data.numpy()
        img[j*28:(j+1)*28] = np.concatenate([x for x in result], axis=-1)
    return img

class Discriminator(nn.Module):
    """
        Convolutional Discriminator for MNIST
    """
    def __init__(self, in_channel=1, input_size=784, condition_size=10, num_classes=1):
        super(Discriminator, self).__init__()
        
        self.transform = nn.Sequential(
            nn.Linear(input_size+condition_size, 784),
            nn.LeakyReLU(0.2),
        )
        self.conv1 = nn.Sequential(
            # 28 -> 14
            nn.Conv2d(in_channel, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        
        
        self.conv2 = nn.Sequential(
            # 14 -> 7
            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        
        self.conv3 = nn.Sequential(
            # 7 -> 4
            nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )
         
        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 1024, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2)
            ,
        )
        
        self.fc = nn.Sequential(
            # reshape input, 128 -> 1
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x, c=None):
        # x: (N, 1, 28, 28), c: (N, 10)
        x, c = x.view(x.size(0), -1), c.float() # may not need
        
      
        
        v = torch.cat((x, c), 1) # v: (N, 794)
       
        y_ = self.transform(v) # (N, 784)
        
        
        y_ = y_.view(y_.shape[0], 1, 28, 28) # (N, 1, 28, 28)
        
        
        y_ = self.conv1(y_)
        
        y_ = self.conv2(y_)
        
        y_ = self.conv3(y_)
        
        y_ = self.conv4(y_)
        
       
        y_ = y_.view(y_.size(0), -1)
       
        y_ = self.fc(y_)
        
        return y_

class Generator(nn.Module):
    """
        Convolutional Generator for MNIST
    """
    def __init__(self, input_size=100, condition_size=10):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size+condition_size, 4*4*512),
            nn.ReLU(),
        )
        self.conv = nn.Sequential(
            # input: 4 by 4, output: 7 by 7
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # input: 7 by 7, output: 14 by 14
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # input: 14 by 14, output: 28 by 28
            nn.ConvTranspose2d(128, 1, 4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )
        
    def forward(self, x, c):
        # x: (N, 100), c: (N, 10)
        x, c = x.view(x.size(0), -1), c.float() # may not need
        v = torch.cat((x, c), 1) # v: (N, 110)
        y_ = self.fc(v)
        y_ = y_.view(y_.size(0), 512, 4, 4)
        y_ = self.conv(y_) # (N, 28, 28)
        return y_

D = Discriminator().to(DEVICE)
G = Generator().to(DEVICE)
# D.load_state_dict('D_dc.pkl')
# G.load_state_dict('G_dc.pkl')



CRITERION = nn.BCELoss()
D_OPT = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
G_OPT = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))

EPOCHS =  50# need more than 20 epochs for training generator

N_CRITIC = 1 # for training more k steps about Discriminator
N_NOISE = 100

D_LABELS = torch.ones([BATCH_SIZE, 1]).to(DEVICE) # Discriminator Label to real
D_FAKES = torch.zeros([BATCH_SIZE, 1]).to(DEVICE) # Discriminator Label to fake

IMG_LIST = []
GENERATOR_ERROR = []
DISCRIMINATOR_ERROR = []


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    plt.imshow(img, cmap='gray')
    plt.show()


def image_save(img):
    img = img / 2 + 0.5     # unnormalize
    plt.title("Training Images")
    plt.imshow(img,cmap='gray')
    plt.imsave('./images/cGAN_GENERATED_IMAGE', img, cmap='gray')
    
    


for epoch in range(EPOCHS):
    STEP = 0
    print(f'Epoch: {epoch +1}')
    for idx, (images, labels) in enumerate(DATA_LOADER):
        # Training Discriminator
        x = images.to(DEVICE)
        y = labels.view(BATCH_SIZE, 1)
        y = to_onehot(y).to(DEVICE)
        x_outputs = D(x, y)
        D_x_loss = CRITERION(x_outputs, D_LABELS)

        z = torch.randn(BATCH_SIZE, N_NOISE).to(DEVICE)
        
        z_outputs = D(G(z, y), y)
        D_z_loss = CRITERION(z_outputs, D_FAKES)
        D_loss = D_x_loss + D_z_loss
        DISCRIMINATOR_ERROR.append(D_loss)
        D.zero_grad()

        # d_writer.add_scalar('Discriminator_Loss', DISCRIMINATION_ERROR.item(), idx)

        D_loss.backward()
        D_OPT.step()

        if STEP % N_CRITIC == 0:
            # Training Generator
            z = torch.randn(BATCH_SIZE, N_NOISE).to(DEVICE)
            z_outputs = D(G(z, y), y)
            G_loss = CRITERION(z_outputs, D_LABELS)
            GENERATOR_ERROR.append(G_loss)
            
            # g_writer.add_scalar('Generator_Loss', GENERATOR_ERROR.item(), idx)

            D.zero_grad()
            G.zero_grad()
            G_loss.backward()
            G_OPT.step()
            
            
        
        if STEP % 100 == 0:
            print('Step: {}, D Loss: {}, G Loss: {}'.format(STEP, D_loss.item(), G_loss.item()))
            img = get_sample_image(G, N_NOISE)
            imshow(img)
            # image_writer.add_image('image', img, idx)
            IMG_LIST.append(img)
            G.train()
        STEP += 1
        
        
    if (epoch == EPOCHS-1):
        G.eval()
        img = get_sample_image(G, N_NOISE)
        #imshow(img)
        image_save(img)
            

# d_writer.close()
# g_writer.close()
# image_writer.close()






fig = plt.figure(figsize=(8,8))
plt.title('cGAN - MNIST')
plt.axis("off")
ims = [[plt.imshow(i, animated=True, cmap='gray')] for i in IMG_LIST]
ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=1000, blit=True)




ani.save('./images/cGAN.gif', writer='imagemagick')



torch.save(G, './models/CGAN_GENERATOR.pt')
torch.save(D, './models/CGAN_DISCRIMINATOR.pt')





def get_errors(error_list):
    new_error = []
    for i in error_list:
        detached = i.detach()# Removes the requires grad
        value = detached.item()
        new_error.append(value)
    return(new_error)


generator_error_values = get_errors(GENERATOR_ERROR)
discriminator_error_values = get_errors(DISCRIMINATOR_ERROR)

def smooth_line(value_list, smoothing_value): # Smoothing value - number of images seen to smooth on
    smoothed = sc.savgol_filter(value_list, smoothing_value, 3)
    return smoothed


smooth_generator = smooth_line(GENERATOR_ERROR, 1001)
smooth_discriminator = smooth_line(DISCRIMINATOR_ERROR, 1001)

def plot_errors(error_list, smoothed, name, colour):
    fig = plt.figure()
    plt.plot(error_list, color=colour)
    plt.plot(smoothed)
    fig.suptitle(name, fontsize=20)
    plt.xlabel('Images Seen', fontsize=18)
    plt.ylabel('Error', fontsize=16)
    fig.savefig(f'./graphs/{name}.jpg')

plot_errors(GENERATOR_ERROR, smooth_generator, 'cGAN_G_ERROR', 'orange')
plot_errors(DISCRIMINATOR_ERROR, smooth_discriminator, 'cGAN_D_ERROR', 'orange')




with open('./output/cGAN_generator.txt', 'wb') as fp:
    pickle.dump(GENERATOR_ERROR, fp)

with open('./output/cGAN_generator_smooth.txt', 'wb') as fp:
    pickle.dump(smooth_generator, fp)


with open('./output/cGAN_discriminator.txt', 'wb') as fp:
    pickle.dump(DISCRIMINATOR_ERROR, fp)

with open('./output/cGAN_discriminator_smooth.txt', 'wb') as fp:
    pickle.dump(smooth_discriminator, fp)