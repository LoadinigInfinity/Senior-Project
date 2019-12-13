"""@author: Josh Pollock
#Code adapted from msattir Vanilla GAN @ "https://github.com/msattir/gan/blob/master/gan.py
"""
import torch
from torch import optim, nn
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
from utils import Logger
import tkinter.filedialog

def faces():
    root = tkinter.Tk()
    root.withdraw()
    print("Select folder with the images")
    out_dir = tkinter.filedialog.askdirectory()
    #    fp = Image.open(file_path).convert('RGB') #attempt to fix grayscale error
    #    Normalize = "image = (image - mean) / std"    
#    compose = transforms.Compose(
#        [transforms.Resize([256, 256]),
#         transforms.RandomCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor()])
    compose = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])
#    return  datasets.ImageFolder(root=out_dir, transform=compose)
    return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)

#Loading Data
data = faces()

# Create loader with data, so that we can iterate over it
data_loader = torch.utils.data.DataLoader(data, batch_size=100,shuffle=True)
num_batches = len(data_loader)

class DiscriminatorNet(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    """
    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        '''
        # The image size = 256
        '''
        n_features = 784 # The image size = 28 x 28 = 784 #for MNIST dataset
#        n_features = 50176
        n_out = 1
        
        self.net0 = nn.Sequential(nn.Linear(n_features, 1024), nn.LeakyReLU(0.2), nn.Dropout(0.3))
        self.net1 = nn.Sequential(nn.Linear(1024, 512), nn.LeakyReLU(0.2), nn.Dropout(0.3))
        self.net2 = nn.Sequential(nn.Linear(512, 256), nn.LeakyReLU(0.2), nn.Dropout(0.3))
        self.out = nn.Sequential(torch.nn.Linear(256, n_out), torch.nn.Sigmoid())

    def forward(self, x):
        x = self.net0(x)
        x = self.net1(x)
        x = self.net2(x)
        x = self.out(x)
        return x
discriminator = DiscriminatorNet()


class GeneratorNet(torch.nn.Module):
    """
    A three hidden-layer generative neural network
    """
    def __init__(self):
        super(GeneratorNet, self).__init__()
        n_features = 100
#        n_out = 50176
        n_out = 784
        
        self.net0 = nn.Sequential(nn.Linear(n_features, 256), nn.LeakyReLU(0.2))
        self.net1 = nn.Sequential(nn.Linear(256, 512), nn.LeakyReLU(0.2))
        self.net2 = nn.Sequential(nn.Linear(512, 1024), nn.LeakyReLU(0.2))
        self.out = nn.Sequential(nn.Linear(1024, n_out), nn.Tanh())

    def forward(self, x):
        x = self.net0(x)
        x = self.net1(x)
        x = self.net2(x)
        x = self.out(x)
        return x
generator = GeneratorNet()

# Optimizers
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)
loss = nn.BCELoss()

def ones_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = Variable(torch.ones(size, 1))
    return data

def zeros_target(size):
    ''' 
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size, 1))
    return data

def images_to_vectors(images):
    return images.view(images.size(0), 784)
#    return images.view(images.size(0), 50176)

def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 1, 28, 28)
#    return vectors.view(vectors.size(0), 1, 224, 224)
    

def noise(size):
    '''
    Generates a 1-d vector of gaussian sampled random values
    '''
    n = Variable(torch.randn(size, 100))
    return n


def train_discriminator(optimizer, real_data, fake_data):
    # Reset gradients
   optimizer.zero_grad()
   prediction_real = discriminator(real_data)
   error_real = loss(prediction_real, ones_target(real_data.size(0)))
   error_real.backward()
   prediction_fake = discriminator(fake_data)
   # Calculate error and backpropagate
   error_fake = loss(prediction_fake, zeros_target(real_data.size(0)))
   error_fake.backward()
   optimizer.step()
   return error_real + error_fake, prediction_real, prediction_fake


def train_generator(optimizer, fake_data):
   optimizer.zero_grad()
   predictions = discriminator(fake_data)
   error = loss(predictions, ones_target(fake_data.size(0)))
   error.backward()
   optimizer.step()
   return error

num_test_samples = 25
test_noise = noise(num_test_samples)

# Create logger instance
logger = Logger(model_name='VGAN', data_name='faces')
num_epochs = 500
for epoch in range(num_epochs):
    for n_batch, (real_batch,_) in enumerate(data_loader):
       
           #Train Disc
        real_data = Variable(images_to_vectors(real_batch))
        fake_data = generator(noise(real_batch.size(0))).detach()
        d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer, real_data, fake_data)
        
        #Train Gen
        fake_data = generator(noise(real_batch.size(0)))
        g_error = train_generator(g_optimizer, fake_data)
        
        #print (d_error, g_error, epoch)
        logger.log(d_error, g_error, epoch, n_batch, num_batches)
        
        if (n_batch) % 100 == 0: 
            test_images = vectors_to_images(generator(test_noise))
            test_images = test_images.data
            logger.log_images(test_images, num_test_samples, epoch, n_batch, num_batches);
            # Display status Logs
            logger.display_status(epoch, num_epochs, n_batch, num_batches,d_error, g_error, d_pred_real, d_pred_fake)
            logger.save_models(generator, discriminator, epoch)