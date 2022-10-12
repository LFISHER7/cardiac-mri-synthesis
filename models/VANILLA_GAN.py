import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.autograd.variable import Variable


def mnist_data():
    compose = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(.5,.5, .5), (.5,.5, .5)
        ])
    out_dir = './dataset'
    return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)

ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Load data
data = mnist_data()
# Create loader with data, so that we can iterate over it
data_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)
# Num batches
num_batches = len(data_loader)

### Discriminator

The discriminator take in a flattened images as input (28 * 28 for MNIST).

def images_to_flatened_vector(images):
    return images.view(images.size(0), 784)

def flattened_vector_to_image(vectors):
    return vectors.view(vectors.size(0), 1, 28, 28)

class DiscriminatorNet(nn.Module):
    
    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        
        self.hidden0 = nn.Sequential(
            nn.Linear(28*28, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        
        self.hidden1 = nn.Sequential(
            nn.Linear(1024, 512), 
            nn.LeakyReLU(0.2), 
            nn.Dropout(0.3),
        )
        
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        
        self.out = nn.Sequential(
            nn.Linear(256, 1), 
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x
    
discriminator = DiscriminatorNet()
        
        

### Generator


def noise(size):
    n = Variable(torch.rand(size, 100))
    if torch.cuda.is_available(): return..cudauda()
    return n

class GeneratorNet(nn.Module):
    
    def __init__(self):
        super(GeneratorNet, self).__init__()
        
        self.hidden0 = nn.Sequential(
            nn.Linear(100, 256), 
            nn.LeakyReLU(0.2)
        )
        
        self.hidden1 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2)
        )
        
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2)
        )
        
        self.out = nn.Sequential(
            nn.Linear(1024, 784),
            nn.Tanh())
    
    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x
    
generator = GeneratorNet() 


if torch.cuda.is_available():
    discriminator.cuda()
    generator.cuda()

## Optimization



discriminator_opt = optim.Adam(discriminator.parameters(), lr = 0.0002)
generator_opt = optim.Adam(generator.parameters(), lr=0.0002)

loss = nn.BCELoss() # Use Binary cross entropy loass as it resembles log loss.

#### Labels



def one_labels(size):
    data = Variable(torch.ones(size, 1))
    if torch.cuda.is_available(): return data.cuda()
    return data

def zero_labels(size):
    data = Variable(torch.zeros(size, 1))
    if torch.cuda.is_available(): return data.cuda()
    return data


## Training

def train_discriminator(optimizer, real_data, fake_data):
    N = real_data.size(0) # Tensor that is 1 dimensional but has no elements.
    optimizer.zero_grad()
    
    # TAKE IN REAL DATA
    predictions_real = discriminator(real_data)
    error_real = loss(predictions_real, zero_labels(N))
    error_real.backward()
    
    # TAKE IN FAKE DATA
    predictions_fake = discriminator(fake_data)
    error_fake = loss(predictions_fake, one_labels(N))
    error_fake.backward()
    
    # UPDATE WEIGHTS
    optimizer.step()
    
    return error_real + error_fake, predictions_real, predictions_fake


def train_generator(optimizer, fake_data):
    N = fake_data.size(0)
    
    optimizer.zero_grad()
    
    # SAMPLE NOISE AND GENERATE FAKE DATA
    generated = discriminator(fake_data)
    
    # CALCULATE ERROR
    error = loss(generated, one_labels(N))
    error.backward()
    
    # UPDATE WEIGHTS WITH GRADIENTS
    optimizer.step()
    
    return error
    

### Testing Generator


num_test_samples = 16
test_noise = noise(num_test_samples)

## Running

import timeit
import matplotlib.animation as animation
import torchvision.utils as vutils


start = timeit.default_timer()

epochs = 200

img_list = []
generator_error = []
discriminator_error = []

for epoch in range(epochs):
    print(f'Epoch: {epoch + 1}')
    for i,(data,_) in enumerate(data_loader):
        N = data.size(0)
        
        # TRAIN DISCRIMINATOR
        
        real_data = Variable(images_to_flatened_vector(data))
        if torch.cuda.is_available(): real_data = real_data.cuda()
        # Generate fake data and detach - dont need to calculate gradients for generator
        
        fake_data = generator(noise(N)).detach()
        
        # Train D
        
        d_error, d_pred_real, d_pred_fake = train_discriminator(discriminator_opt, real_data, fake_data)
        
        # Train G
        
        fake_data = generator(noise(N))
        g_error = train_generator(generator_opt, fake_data)
        
        discriminator_error.append(d_error)
        generator_error.append(g_error)
        #print(f'Generator error: {g_error}')
        #print(f'Discriminator error: {d_error}')
        
        
        if (i % 100 == 0):
            test_images = vectors_to_images(generator(test_noise))
            test_images = test_images.data.cpu()     
            img_list.append(vutils.make_grid(test_images, padding=2, normalize=True))
    
            
            image_grid = torchvision.utils.make_grid(test_images)
            
            imshow(image_grid)

#%%capture
fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)


ani.save('animation.gif', writer='imagemagick')


stop = timeit.default_timer()

print('Time: ', stop - start)  