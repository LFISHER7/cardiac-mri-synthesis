import torch
from torch import nn
from torch.optim import Adam
from torch.autograd import Variable
import timeit
import matplotlib.animation as animation
import torchvision.utils as vutils
from torchvision import transforms, datasets
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sc


import torch
import torch.nn.functional as F
from torch.autograd import Variable


def resize2d(img, size):
    return (F.adaptive_avg_pool2d(Variable(img), size)).data



ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

print(f'Running on {device}')

image_directory = './dataset'

BATCH_SIZE=100

IMAGE_SIZE = 64

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
dataroot =  './ACDC_DATA_CONDITION'


dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Grayscale(num_output_channels=1),
                               transforms.Resize(IMAGE_SIZE),
                               transforms.CenterCrop(IMAGE_SIZE),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,)),
                           ]))
# Create the dataloader
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=0)


# subset_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
# data_loader = torch.utils.data.DataLoader(cifar_data, batch_size=BATCH_SIZE, sampler=torch.utils.data.sampler.SubsetRandomSampler(subset_indices))


number_of_batches = len(data_loader)


print(f'Number of images: {number_of_batches * 100}')
print(f'Number of batches: {number_of_batches}')

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()



def image_save(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    images = np.transpose(npimg, (1, 2, 0))
    plt.title("Training Images")
    plt.imshow(images)
    plt.imsave('./images/DCGAN_GENERATED_IMAGES.jpg', images, cmap='gray')

def images_to_vector(images):
    return images.view(images.size(0), (64*64))

def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 3, 64, 64)
# Plot some training images
# training_batch = next(iter(data_loader))

# plt.figure(figsize=(8,8))
# plt.axis("off")
# plt.title("Training Images")
# image_to_plot = np.transpose(vutils.make_grid(training_batch[0].to(device)[:16], padding=2, normalize=True).cpu(),(1,2,0))
# plt.imshow(image_to_plot)
# plt.imsave('./images/TRAINING_IMAGES.jpg', image_to_plot)

class DiscriminativeNet(torch.nn.Module):
    
    def __init__(self):
        super(DiscriminativeNet, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=128, kernel_size=4, 
                stride=2, padding=1, bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=512, out_channels=1024, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.out = nn.Sequential(
            nn.Linear(1024*4*4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # Flatten and apply sigmoid
        x = x.view(-1, 1024*4*4)
        x = self.out(x)
        return x

class GenerativeNet(torch.nn.Module):
    
    def __init__(self):
        super(GenerativeNet, self).__init__()
        
        self.linear = torch.nn.Linear(100, 1024*4*4)
        
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=1024, out_channels=512, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=512, out_channels=256, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256, out_channels=128, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128, out_channels=3, kernel_size=4,
                stride=2, padding=1, bias=False
            )
        )
        self.out = torch.nn.Tanh()

    def forward(self, x):
        # Project and reshape
        x = self.linear(x)
        x = x.view(x.shape[0], 1024, 4, 4)
        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # Apply Tanh
        return self.out(x)
    
# Noise
def noise(size):
    n = Variable(torch.randn(size, 100))
    if torch.cuda.is_available(): return n.cuda()
    return n

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        
        m.weight.data.normal_(0.00, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        

# Create Network instances and init weights
generator = GenerativeNet()
generator.apply(init_weights)

discriminator = DiscriminativeNet()
discriminator.apply(init_weights)

# Enable cuda if available
if torch.cuda.is_available():
    generator.cuda()
    discriminator.cuda()

# Optimizers
d_optimizer = Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
g_optimizer = Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Loss function
loss = nn.BCELoss()

# Number of epochs
num_epochs = 50

def real_data_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = Variable(torch.ones(size, 1))
    
    if torch.cuda.is_available(): return data.cuda()
    return data

def fake_data_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size, 1))
    
    if torch.cuda.is_available(): return data.cuda()
    return data

def train_discriminator(optimizer, real_data, fake_data):
    # Reset gradients
    optimizer.zero_grad()
    
    # 1.1 Train on Real Data
    prediction_real = discriminator(real_data)
    # Calculate error and backpropagate
    
    error_real = loss(prediction_real, real_data_target(real_data.size(0)))
    error_real.backward()

    # 1.2 Train on Fake Data
    prediction_fake = discriminator(fake_data)
    # Calculate error and backpropagate
    error_fake = loss(prediction_fake, fake_data_target(real_data.size(0)))
    error_fake.backward()
    
    # 1.3 Update weights with gradients
    optimizer.step()
    
    # Return error
    return error_real + error_fake, prediction_real, prediction_fake

def train_generator(optimizer, fake_data):
    # 2. Train Generator
    # Reset gradients
    optimizer.zero_grad()
    # Sample noise and generate fake data
    prediction = discriminator(fake_data)
    # Calculate error and backpropagate
    error = loss(prediction, real_data_target(prediction.size(0)))
    error.backward()
    # Update weights with gradients
    optimizer.step()
    # Return error
    return error

num_test_samples = 16
test_noise = noise(num_test_samples)

discriminator_error_dcgan = []
generator_error_dcgan = []
img_list=[]









for epoch in range(num_epochs):
    STEP=0
    
    print(f'Epoch: {epoch + 1}')
    
    for n_batch, (real_batch,_) in enumerate(data_loader):
        
        # 1. Train Discriminator
        real_data = Variable(real_batch)
        if torch.cuda.is_available(): real_data = real_data.cuda()
        # Generate fake data
        fake_data = generator(noise(real_data.size(0))).detach()
        # Train D
        
        d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer, 
                                                                real_data, fake_data)

        # 2. Train Generator
        # Generate fake data
        fake_data = generator(noise(real_batch.size(0)))
        # Train G
        g_error = train_generator(g_optimizer, fake_data)
        
        discriminator_error_dcgan.append(d_error)
        generator_error_dcgan.append(g_error)
        #print(f'Generator error: {g_error}')
        #print(f'Discriminator error: {d_error}')
        
        
        if (n_batch % 100 == 0):
            test_images = vectors_to_images(generator(test_noise))
            test_images = test_images.data.cpu()     
            img_list.append(vutils.make_grid(test_images, padding=2, normalize=True))
    
            
            image_grid = torchvision.utils.make_grid(test_images)
            print(STEP)
            
        STEP+=1

    if (epoch == num_epochs-1):
        test_images = vectors_to_images(generator(test_noise))
        test_images = test_images.data.cpu()  
        image_grid = torchvision.utils.make_grid(test_images)
        image_save(image_grid)



    

    
torch.save(generator, './models/DCGAN/GENERATOR.pt')
torch.save(discriminator, './models/DCGAN/DISCRIMINATOR.pt')

#%%capture
fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)


ani.save('./images/animation_dcgan.gif', writer='imagemagick')





def get_errors(error_list):
    new_error = []
    for i in error_list:
        detached = i.detach()# Removes the requires grad
        value = detached.item()
        new_error.append(value)
    return(new_error)


generator_error_values = get_errors(generator_error_dcgan)
discriminator_error_values = get_errors(discriminator_error_dcgan)

def smooth_line(value_list, smoothing_value): # Smoothing value - number of images seen to smooth on
    smoothed = sc.savgol_filter(value_list, smoothing_value, 3)
    return smoothed


smooth_generator = smooth_line(generator_error_dcgan, 1001)
smooth_discriminator = smooth_line(discriminator_error_dcgan, 1001)


fig = plt.figure()
plt.plot(generator_error_dcgan, color='orange')

plt.plot(smooth_generator)
fig.suptitle('Generator Error', fontsize=20)
plt.xlabel('Images Seen', fontsize=18)
plt.ylabel('Error', fontsize=16)
fig.savefig('./graphs/DCGAN_G_ERROR.jpg')




fig = plt.figure()
plt.plot(discriminator_error_dcgan)
plt.plot(smooth_discriminator, color='orange')
fig.suptitle('Discriminator Error', fontsize=20)
plt.xlabel('Images Seen', fontsize=18)
plt.ylabel('Error', fontsize=16)
fig.savefig('./graphs/DCGAN_D_ERROR.jpg')
