import os, time
import matplotlib.pyplot as plt
import itertools
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.autograd import Variable
from networks import generator1, generator2, discriminator

def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

def show_train_mse(hist, show = False, save = False, path = 'Train_mse.png'):
    x = range(len(hist['G1_mse']))

    y1 = hist['G1_mse']
    y2 = hist['G2_mse']
    y3 = hist['2G1_mse']

    plt.plot(x, y1, label='G1_mse')
    plt.plot(x, y2, label='G2_mse')
    plt.plot(x, y3, label='2*G1_mse')

    plt.xlabel('Epoch')
    plt.ylabel('MSE')

    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

# training parameters
batch_size = 128
lr = 0.001
train_epoch = 100
pretrain_epoch = 1
lambda_gp = 10
pretrained = False
rate = 4
img_size = 32

# results save folder
root = 'result/'
model = 'MNIST_'
if not os.path.isdir(root):
    os.mkdir(root)
    
# data_loader
transform = transforms.Compose([
        transforms.Scale(img_size),
        transforms.ToTensor()
])
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('/path/to/dataset/', train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True)

# network
G1 = generator1(64, rate)
G2 = generator2(32)
D = discriminator(16, rate)
if pretrained == True:
    G1.load_state_dict(torch.load(root + model + 'generator1_param.pkl'))
    G2.load_state_dict(torch.load(root + model + 'generator2_param.pkl'))
    D.load_state_dict(torch.load(root + model + 'discriminator_param.pkl'))
else:
    G1.weight_init(mean=0.0, std=0.02)
    G2.weight_init(mean=0.0, std=0.02)
    D.weight_init(mean=0.0, std=0.02)
G1.cuda()
G2.cuda()
D.cuda()

G1_optimizer = optim.RMSprop(G1.parameters(), lr=lr/10)
G2_optimizer = optim.RMSprop(G2.parameters(), lr=lr/10)
D_optimizer = optim.RMSprop(D.parameters(), lr=lr)

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['G1_mse'] = []
train_hist['G2_mse'] = []
train_hist['2G1_mse'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

print('training start!')
start_time = time.time()
for epoch in range(train_epoch):
    D_losses = []
    G_losses = []
    G1_mse = []
    G1_2_mse = []
    G2_mse = []

    # learning rate decay
    if (epoch+1) == 50:
        G1_optimizer.param_groups[0]['lr'] /= 10
        G2_optimizer.param_groups[0]['lr'] /= 10
        D_optimizer.param_groups[0]['lr'] /= 10
        print("learning rate change!")

    epoch_start_time = time.time()
    for x_, y_ in train_loader:
        # train generator1 G1
        G1.zero_grad()
        mini_batch = x_.size()[0]
        
        x_ = Variable(x_.cuda())
        x_mse_, bitstream = G1(x_)
        G1_loss = torch.mean((x_mse_ - x_)**2)

        G1_loss.backward()
        G1_optimizer.step()

        G1_mse.append(G1_loss.data)
        G1_2_mse.append(2*G1_loss.data)

        if (epoch+1) > pretrain_epoch:
            # train discriminator D
            D.zero_grad()

            D_result = D(x_, bitstream.data).squeeze()
            D_real_loss = -D_result.mean()

            z_ = torch.randn((mini_batch, 100 - rate)).view(-1, 100 - rate, 1, 1)
            z_ = Variable(z_.cuda())
            G2_input = torch.cat([bitstream.data, z_], 1)

            G_result = G2(G2_input)
            D_result = D(G_result.data, bitstream.data).squeeze()

            D_fake_loss = D_result.mean()
            D_fake_score = D_result.data.mean()

            D_train_loss = D_real_loss + D_fake_loss

            D_train_loss.backward()
            D_optimizer.step()

            D_losses.append(D_train_loss.data)

            #gradient penalty
            D.zero_grad()
            alpha = torch.rand(x_.size(0), 1, 1, 1)
            alpha1 = alpha.cuda().expand_as(x_)
            interpolated1 = Variable(alpha1 * x_.data + (1 - alpha1) * G_result.data, requires_grad=True)
            interpolated2 = Variable(bitstream.data, requires_grad=True)

            out = D(interpolated1, interpolated2).squeeze()

            grad = torch.autograd.grad(outputs=out,
                                       inputs=[interpolated1, interpolated2],
                                       grad_outputs=torch.ones(out.size()).cuda(),
                                       retain_graph=True,
                                       create_graph=True,
                                       only_inputs=True)[0]

            grad = grad.view(grad.size(0), -1)
            grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
            d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

            # Backward + Optimize
            gp_loss = lambda_gp * d_loss_gp

            gp_loss.backward()
            D_optimizer.step()

            # train generator G2
            D.zero_grad()
            G2.zero_grad()

            z_ = torch.randn((mini_batch, 100 - rate)).view(-1, 100 - rate, 1, 1)
            z_ = Variable(z_.cuda())
            G2_input = torch.cat([bitstream.data, z_], 1)

            G_result = G2(G2_input)
            D_result = D(G_result, bitstream.data).squeeze()

            G_train_loss = - D_result.mean()

            G_train_loss.backward()
            G2_optimizer.step()

            G_losses.append(G_train_loss.data)
            g2_mse = torch.mean((G_result.data - x_)**2)
            G2_mse.append(g2_mse.data)
        else:
            #pre-training
            arr = [i for i in range(0, mini_batch)]
            np.random.shuffle(arr)
            bit_f = bitstream[arr,:,:,:].data
            D_result = D(x_, bitstream.data).squeeze()
            D_real_loss = -D_result.mean()

            D_result = D(x_, bit_f.data).squeeze()
            D_fake_loss = D_result.mean()
            D_fake_score = D_result.data.mean()

            D_train_loss = D_real_loss + D_fake_loss

            D_train_loss.backward()
            D_optimizer.step()

            D_losses.append(D_train_loss.data)

            #gradient penalty
            D.zero_grad()
            alpha = torch.rand(x_.size(0), 1, 1, 1)
            alpha1 = alpha.cuda().expand_as(x_)
            interpolated1 = Variable(x_.data, requires_grad=True)
            #interpolated2 = Variable(bit2.data, requires_grad=True)

            alpha2 = alpha.cuda().expand_as(bitstream)
            interpolated2 = Variable(alpha2 * bitstream.data + (1 - alpha2) * bit_f.data, requires_grad=True)
            out = D(interpolated1, interpolated2).squeeze()

            grad = torch.autograd.grad(outputs=out,
                                       inputs=[interpolated1, interpolated2],
                                       grad_outputs=torch.ones(out.size()).cuda(),
                                       retain_graph=True,
                                       create_graph=True,
                                       only_inputs=True)[0]

            grad = grad.view(grad.size(0), -1)
            grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
            d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

            # Backward + Optimize
            gp_loss = lambda_gp * d_loss_gp

            gp_loss.backward()
            D_optimizer.step()

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time

    print('[%d/%d] - ptime: %.2f, loss_d: %.3f, G1_mse: %.4f, G2_mse: %.4f' % ((epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
                                                              torch.mean(torch.FloatTensor(G1_mse)), torch.mean(torch.FloatTensor(G2_mse))))
    
    fixed_p = root + 'Fixed_results/' + model + str(epoch + 1) + '.png'
    train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
    train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
    train_hist['G1_mse'].append(torch.mean(torch.FloatTensor(G1_mse)))
    train_hist['2G1_mse'].append(torch.mean(torch.FloatTensor(G1_2_mse)))
    train_hist['G2_mse'].append(torch.mean(torch.FloatTensor(G2_mse)))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print("Avg one epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), train_epoch, total_ptime))
print("Training finish!... save training results")
torch.save(G1.state_dict(), root + model + 'generator1_param.pkl')
torch.save(G2.state_dict(), root + model + 'generator2_param.pkl')
torch.save(D.state_dict(), root + model + 'discriminator_param.pkl')
with open(root + model + 'train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)

show_train_hist(train_hist, save=True, path=root + model + 'train_hist.png')
show_train_mse(train_hist, save=True, path=root + model + 'train_mse.png')
np.savetxt(root+model+'G1_mse.txt', train_hist['G1_mse'], delimiter=" ")
np.savetxt(root+model+'G2_mse.txt', train_hist['G2_mse'], delimiter=" ")

