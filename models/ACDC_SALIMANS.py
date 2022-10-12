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

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

print(f'Running on {device}')

image_directory = './dataset'

BATCH_SIZE=100

IMAGE_SIZE = 64

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
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
data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
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
    plt.imsave('./images/SALIMANS_GENERATED_IMAGES.jpg', images, cmap='gray')

def images_to_vector(images):
    return images.view(images.size(0), (64*64))

def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 1, 64, 64)
# Plot some training images
# training_batch = next(iter(data_loader))

# plt.figure(figsize=(8,8))
# plt.axis("off")
# plt.title("Training Images")
# image_to_plot = np.transpose(vutils.make_grid(training_batch[0].to(device)[:16], padding=2, normalize=True).cpu(),(1,2,0))
# plt.imshow(image_to_plot)
# plt.imsave('./images/TRAINING_IMAGES.jpg', image_to_plot)

class DiscriminativeNet(torch.nn.Module):
    
    def __init__(self, nb=128, nc=16):
        super(DiscriminativeNet, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=128, kernel_size=4, 
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



        T_ten_init = torch.randn(1024 * 4 * 4, 128 * 16) * 0.1
        self.T_tensor = nn.Parameter(T_ten_init, requires_grad=True)
        
        self.out = nn.Sequential(
            nn.Linear(1024*4*4+128, 1),
            nn.Sigmoid(),
        )

        self.aux_layer = nn.Sequential(nn.Linear(1024 * 4 *4, 5 + 1), nn.Softmax(dim=1))

    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # Flatten and apply sigmoid
        y = x.view(-1, 1024*4*4)
        
        # #### Minibatch Discrimination ###
        T_tensor = self.T_tensor
        
        Ms = torch.matmul(y, T_tensor)
        Ms = Ms.view(-1, 128, 16)

        out_tensor = []
        for i in range(Ms.size()[0]):

            out_i = None
            for j in range(Ms.size()[0]):
                o_i = torch.sum(torch.abs(Ms[i, :, :] - Ms[j, :, :]), 1)
                o_i = torch.exp(-o_i)
                if out_i is None:
                    out_i = o_i
                else:
                    out_i = out_i + o_i

            out_tensor.append(out_i)

        out_T = torch.cat(tuple(out_tensor)).view(Ms.size()[0], 128)
        x = torch.cat((y, out_T), 1)
        # #### Minibatch Discrimination ###

        validity = self.out(x)
        label = self.aux_layer(y)

        # x = self.out(x)
        return validity, label

class GenerativeNet(torch.nn.Module):
    
    def __init__(self):
        super(GenerativeNet, self).__init__()
        
        self.label_emb = nn.Embedding(5, 100)

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
                in_channels=128, out_channels=1, kernel_size=4,
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
adversarial_loss = torch.nn.BCELoss()
auxiliary_loss = torch.nn.CrossEntropyLoss()

# Number of epochs
num_epochs = 20

def real_data_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = Variable(torch.ones(size-round(size*0.3), 1))
    data.new_full((size, 1), 0.9)
    

    
    data_0 = Variable(torch.zeros(round(size*0.3), 1))
    data = torch.cat((data, data_0))


    r=torch.randperm(size)    
    data=data[r]


    if torch.cuda.is_available(): return data.cuda()
    return data

def fake_data_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size, 1))
    
    if torch.cuda.is_available(): return data.cuda()
    return data


num_test_samples = 16
test_noise = noise(num_test_samples)

discriminator_error_dcgan = []
generator_error_dcgan = []
img_list=[]

inception_score_list = []



FID_LIST= []



# inception_network = torch.load('./models/INCEPTION_DCGAN_TRANSFERING_0.pt').cpu()
# inception_network.eval()


ss_accuracy_list = []


for epoch in range(num_epochs):
    STEP=0
    
    print(f'Epoch: {epoch + 1}')
    
    # if epoch == 0:
    #     images_before_training = generate_image(100, 500, generator, 64)

    #     for i, image in enumerate(images_before_training):

        
    #         real_image = image / 2 + 0.5
    #         npimg = real_image.numpy()
    #         npimg = np.reshape(npimg, (3, 64, 64))
    #         images = np.transpose(npimg, (1, 2, 0))
    #         plt.imshow(images)

    #         plt.imsave(f'./FID_IMAGES_FAKE/{i}.jpg', images)

    #         images_before_training[i] = resize2d(image, (299, 299))


    #     scoring = (inception_score(images_before_training, inception_network,  num_classes=10, num_splits=10, split=False))
    #     inception_score_list.append(scoring)

    #     print(f'INCEPTION SCORE START OF TRAINING: {scoring}')


    #     FID = calculate_fid_given_paths(paths = ['./FID_IMAGES_REAL', './FID_IMAGES_FAKE'], batch_size=50, dims=2048, cuda= '', model = 0)


    #     print(f'FID SCORE START OF TRAINING: {FID}')

    #     FID_LIST.append(FID)





        ## 1. Train Discriminator
       
        batch_size = real_batch.shape[0]
        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)
        fake_aux_gt = Variable(LongTensor(batch_size).fill_(10), requires_grad=False)

        # Configure input
        real_imgs = Variable(real_batch.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))






        # if epoch == 0 and n_batch <500:
        #     ## FID
        #     for i in range(real_data.shape[0]):
        #         real_image = real_data[i]
        #         real_image = real_image / 2 + 0.5
        #         npimg = real_image.numpy()
        #         images = np.transpose(npimg, (1, 2, 0))
        #         plt.imshow(images)

        #         plt.imsave(f'./FID_IMAGES_REAL/{n_batch}.jpg', images)


        ## FID


        g_optimizer.zero_grad()

       
        # Generate fake data
        fake_data = generator(noise(real_imgs.size(0))).detach()
        # Train D


        validity, _ = discriminator(fake_data)
        g_loss = adversarial_loss(validity, valid)
        g_loss.backward()
        g_optimizer.step()


        d_optimizer.zero_grad()

        real_pred, real_aux = discriminator(real_imgs)
        d_real_loss = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, labels)) / 2

        # Loss for fake images
        fake_pred, fake_aux = discriminator(fake_data.detach())
        d_fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, fake_aux_gt)) / 2

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        # Calculate discriminator accuracy
        pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
        gt = np.concatenate([labels.data.cpu().numpy(), fake_aux_gt.data.cpu().numpy()], axis=0)
        d_acc = np.mean(np.argmax(pred, axis=1) == gt)

        d_loss.backward()
        d_optimizer.step()




        

        
        discriminator_error_dcgan.append(d_loss)
        generator_error_dcgan.append(g_loss)
        #print(f'Generator error: {g_error}')
        #print(f'Discriminator error: {d_error}')
        
        
        if (n_batch % 100 == 0):
            test_images = vectors_to_images(generator(test_noise))
            test_images = test_images.data.cpu()     
            img_list.append(vutils.make_grid(test_images, padding=2, normalize=True))
    
            
            image_grid = torchvision.utils.make_grid(test_images)
            print(STEP)
            
        STEP+=1




    images_for_inception = generate_image(100, 500, generator, 64)

    for i, image in enumerate(images_for_inception):

        
        real_image = image / 2 + 0.5
        npimg = real_image.numpy()
        npimg = np.reshape(npimg, (3, 64, 64))
        images = np.transpose(npimg, (1, 2, 0))
        plt.imshow(images)

        plt.imsave(f'./FID_IMAGES_FIXED/{i}.jpg', images)

        images_for_inception[i] = resize2d(image, (299, 299))
    

    scoring = (inception_score(images_for_inception, inception_network,  num_classes=10, num_splits=10, split=False))
    inception_score_list.append(scoring)

    print(f'INCEPTION SCORE: {scoring}')




    FID = calculate_fid_given_paths(paths = ['./FID_IMAGES_REAL', './FID_IMAGES_FIXED'], batch_size=50, dims=2048, cuda= '', model=0)


    print(f'FID: {FID}')

    FID_LIST.append(FID)


    best_fid_score = max(FID_LIST)
    best_inception_score = max(inception_score_list)

    if scoring == best_inception_score or FID == best_fid_score:

        torch.save(generator, './models/model_history/DCGAN/GENERATOR_SALIMANS.pt')
        torch.save(discriminator, './models/model_history/DCGAN/DISCRIMINATOR_SALIMANS.pt')


    correct = 0
    total = 0
    for n_batch, (real_batch,labels) in enumerate(data_loader_test):
        real_imgs = Variable(real_batch.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))

        validity, label_pred = discriminator(real_imgs)
        _, predicted = torch.max(label_pred, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()           
    acc  = 100 * correct/total
    ss_accuracy_list.append(acc)
    print(f'Semi-supervised test accuracy: {acc}%')


    

#%%capture
fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)


ani.save('./images/animation_dcgan.gif', writer='imagemagick')


generator = torch.load('./models/model_history/DCGAN/GENERATOR_SALIMANS.pt')
discriminator = torch.load('./models/model_history/DCGAN/SALIMANS.pt')


test_images = vectors_to_images(generator(test_noise))
test_images = test_images.data.cpu()  
image_grid = torchvision.utils.make_grid(test_images)
image_save(image_grid)


print('GENERATING IMAGES USING GENERATOR')

final_inception_scores = []
final_generated_images = generate_image(100, 10000, generator, 64)

for i, image in enumerate(final_generated_images):

        final_generated_images[i] = resize2d(image, (299, 299))

print('CALCULATING FINAL INCEPTION SCORE')

for i in range(5):
    inception_network = torch.load(f'./models/INCEPTION_DCGAN_TRANSFERING_{i}.pt').cpu()
    inception_network.eval()
    

    inception_network.eval()
    scoring = (inception_score(final_generated_images, inception_network,  num_classes=10, num_splits=10, split=False))
    final_inception_scores.append(scoring)

print(f'FINAL INCEPTION SCORE: {np.mean(final_inception_scores)}')
print(f'SD: {np.std(final_inception_scores)}')

print('CALCULATING FINAL FID...')

final_fid_scores = []
for t in range(5):

    images_for_fid = generate_image(100, 500, generator, 64)

    for i, image in enumerate(images_for_fid):

        
        real_image = image / 2 + 0.5
        npimg = real_image.numpy()
        npimg = np.reshape(npimg, (3, 64, 64))
        images = np.transpose(npimg, (1, 2, 0))
        plt.imshow(images)

        plt.imsave(f'./FID_IMAGES_FAKE/{i}.jpg', images)



    
    for i in range(5):

        FID = calculate_fid_given_paths(paths = ['./FID_IMAGES_REAL', './FID_IMAGES_FAKE'], batch_size=50, dims=2048, cuda= '', model = i)
        final_fid_scores.append(FID)




print(f'FINAL FID SCORE: {np.mean(final_fid_scores)}')
print(f'SD: {np.std(final_fid_scores)}')




x_pos = [0, 400, 800, 1200, 1600, 2000, 2400, 2800, 3200, 3600, 4000]
x_lab = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


fig = plt.figure()
plt.plot(ss_accuracy_list[0:num_epochs])
fig.suptitle('Semi-Supervised Accuracy - DCGAN', fontsize=20)
plt.xlabel('Epoch', fontsize=18)
plt.ylabel('Accuracy', fontsize=16)
fig.savefig('./graphs/SALIMANS_ACC.jpg')



fig = plt.figure()
plt.plot(inception_score_list[0:num_epochs])
fig.suptitle('Inception Score - DCGAN', fontsize=20)
plt.xlabel('Epoch', fontsize=18)
plt.ylabel('IS', fontsize=16)
fig.savefig('./graphs/SALIMANS_IS.jpg')


fig = plt.figure()
plt.plot(FID_LIST[0:num_epochs])
fig.suptitle('FID - DCGAN', fontsize=20)
plt.xlabel('Epoch', fontsize=18)
plt.ylabel('FID', fontsize=16)
fig.savefig('./graphs/SALIMANS_FID.jpg')


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
plt.xticks(x_pos, x_lab)
fig.suptitle('Generator Error', fontsize=20)
plt.xlabel('Epoch', fontsize=18)
plt.ylabel('Error', fontsize=16)
fig.savefig('./graphs/SALIMANS_G_ERROR.jpg')




fig = plt.figure()
plt.plot(discriminator_error_dcgan)
plt.plot(smooth_discriminator, color='orange')
plt.xticks(x_pos, x_lab)
fig.suptitle('Discriminator Error', fontsize=20)
plt.xlabel('Epoch', fontsize=18)
plt.ylabel('Error', fontsize=16)
fig.savefig('./graphs/SALIMANS_D_ERROR.jpg')

print(f'BEST ACCURACY = {max(ss_accuracy_list)}')

# with open('./output/DCGAN_generator.txt', 'wb') as fp:
#     pickle.dump(generator_error_dcgan, fp)

# with open('./output/DCGAN_generator_smooth.txt', 'wb') as fp:
#     pickle.dump(smooth_generator, fp)


# with open('./output/DCGAN_discriminator.txt', 'wb') as fp:
#     pickle.dump(discriminator_error_dcgan, fp)

# with open('./output/DCGAN_discriminator_smooth.txt', 'wb') as fp:
#     pickle.dump(smooth_discriminator, fp)

# with open('./output/DCGAN_IS.txt', 'wb') as fp:
#     pickle.dump(inception_score_list[0:num_epochs], fp)

        