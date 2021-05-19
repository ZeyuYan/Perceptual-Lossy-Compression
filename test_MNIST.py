import os, time
import matplotlib.pyplot as plt
import itertools
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from networks import generator1, generator2, discriminator

def show_result(img, show = False, save = False, path = 'result/'):

    G1.eval()
    G2.eval()
    img_mse, bitstream = G1(img)

    z_ = torch.randn((100, 100-rate)).view(-1, 100-rate, 1, 1)
    z_ = Variable(z_.cuda())
    z_ = torch.cat([bitstream.data, z_], 1)
    img_p = G2(z_)

    size_figure_grid = 10
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(10*10):
        i = k // 10
        j = k % 10
        ax[i, j].cla()
        ax[i, j].imshow(img_mse[k, 0].cpu().data.numpy(), cmap='gray')

    label = 'P=+∞'
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path+'withoutP.png')

    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)
    for k in range(10*10):
        i = k // 10
        j = k % 10
        ax[i, j].cla()
        ax[i, j].imshow(img[k, 0].cpu().data.numpy(), cmap='gray')

    label = 'input'
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path+'input.png')

    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)
    for k in range(10*10):
        i = k // 10
        j = k % 10
        ax[i, j].cla()
        ax[i, j].imshow(img_p[k, 0].cpu().data.numpy(), cmap='gray')

    label = 'P=0'
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path+'p=0.png')


    if show:
        plt.show()
    else:
        plt.close()


# testing parameters
batch_size = 100
rate = 4
img_size = 32

# results save folder
root = 'result/'
model = 'MNIST_'
if not os.path.isdir(root):
    os.mkdir(root)
if not os.path.isdir(root + 'samples'):
    os.mkdir(root + 'samples')
    
# data_loader
transform = transforms.Compose([
        transforms.Scale(img_size),
        transforms.ToTensor()
])
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('/path/to/dataset/', train=False, download=True, transform=transform),
    batch_size=batch_size, shuffle=False)

# network
G1 = generator1(64, rate)
G2 = generator2(32)
D = discriminator(16, rate)
G1.load_state_dict(torch.load(root + model + 'generator1_param.pkl'))
G2.load_state_dict(torch.load(root + model + 'generator2_param.pkl'))
D.load_state_dict(torch.load(root + model + 'discriminator_param.pkl'))
G1.cuda()
G2.cuda()
D.cuda()

train_hist = {}
train_hist['G1_mse'] = []
train_hist['G2_mse'] = []

print('training start!')
start_time = time.time()
G1_mse = []
G2_mse = []

# learning rate decay

epoch_start_time = time.time()
for x_, y_ in train_loader:
    mini_batch = x_.size()[0]
        
    x_ = Variable(x_.cuda())
    x_mse_, bitstream = G1(x_)
    G1_loss = torch.mean((x_mse_ - x_)**2)
    G1_mse.append(G1_loss.data)

    z_ = torch.randn((mini_batch, 100 - rate)).view(-1, 100 - rate, 1, 1)
    z_ = Variable(z_.cuda())
    G2_input = torch.cat([bitstream.data, z_], 1)

    G_result = G2(G2_input)
    g2_mse = torch.mean((G_result.data - x_)**2)
    G2_mse.append(g2_mse.data)

print('mse without P constraint: %.4f, mse with P=0: %.4f' % (torch.mean(torch.FloatTensor(G1_mse)), torch.mean(torch.FloatTensor(G2_mse))))
fixed_p = root + 'samples/' + model
show_result(x_[0:100,:,:,:], save=True, path=fixed_p)


