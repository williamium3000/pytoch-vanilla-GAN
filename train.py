import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import copy
import os
import time
import json
import logging
import torchvision
from torchvision import models, transforms
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from network import Discriminator
from network import Generator
np.random.seed(0)
torch.manual_seed(0)# 为CPU设置种子用于生成随机数，以使得结果是确定的
torch.cuda.manual_seed_all(0)# 为所有的GPU设置种子，以使得结果是确定的

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

def sample(model, device, epoch):
    generator = model[0]
    generator.eval()
    if mutil_gpu:
        generator = generator.module
    with torch.no_grad():
        sampled_latent = torch.tensor(np.random.normal(0, 1, (16, latent_dim)), dtype=torch.float32)
        sampled_latent = sampled_latent.to(device=device)
            
        samples = generator(sampled_latent)
        # print(samples)
        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(28, 28) * 255, cmap='gray')

        if not os.path.exists('out/'):
            os.makedirs('out/')

        plt.savefig('out/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')

        plt.close(fig)



def train(model, optimizer, lr_scheduler, dataloaders, device, epochs):
    generator = model[0]
    discriminator = model[1]
    optimizer_G = optimizer[0]
    optimizer_D = optimizer[1]

    for e in range(epochs):
        for x, y in tqdm(dataloaders['train']):
            generator.train()
            discriminator.train()
            
            valid = torch.ones((x.shape[0], 1))
            fake = torch.zeros((x.shape[0], 1))
            sampled_latent = torch.tensor(np.random.normal(0, 1, (x.shape[0], latent_dim)), dtype=torch.float32).to(device=device)
            x = x.to(device=device)
            valid = valid.to(device=device)
            fake = fake.to(device=device)
            

            
            
            
            generated_imgs = generator(sampled_latent)
            ge_ = discriminator(generated_imgs)
            gt_ = discriminator(x)

            gen_loss = nn.BCELoss()(ge_, valid)
            optimizer_G.zero_grad()
            gen_loss.backward()
            optimizer_G.step()

            
    

            

            dis_loss = (nn.BCELoss()(discriminator(generated_imgs.detach()), fake) + nn.BCELoss()(gt_, valid)) / 2

            optimizer_D.zero_grad()
            dis_loss.backward()
            optimizer_D.step()
            if lr_scheduler:
                lr_scheduler.step()
        print('epoche %d, gen loss = %f, dis loss = %f' % (e, gen_loss.item(), dis_loss.item()))
        logging.info('epoche %d, gen loss = %f, dis loss = %f' % (e, gen_loss.item(), dis_loss.item()))

        sample(model, device, e)


        
        writer.add_scalars("loss", {"GEN":gen_loss.item(), "DIS":dis_loss.item()}, e)
        
        save_model(save_dir='model_checkpoint', file_name="check_point_G", model=generator, optimizer = optimizer_G, lr_scheduler = lr_scheduler)
        save_model(save_dir='model_checkpoint', file_name="check_point_D", model=discriminator, optimizer = optimizer_D, lr_scheduler = lr_scheduler)
    
    save_model(save_dir='model_checkpoint', file_name=task_name + "_G", model=generator, optimizer = optimizer_G, lr_scheduler = lr_scheduler)
    save_model(save_dir='model_checkpoint', file_name=task_name + "_D", model=discriminator, optimizer = optimizer_D, lr_scheduler = lr_scheduler)
    
    return model 


def save_model(save_dir, model, optimizer, lr_scheduler, file_name=None):
    if mutil_gpu:
        model = model.module
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if file_name:
        save_path = os.path.join(save_dir, file_name)
    else:
        save_path = os.path.join(save_dir, str(int(time.time())))
    if lr_scheduler:
        state_dicts = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": lr_scheduler.state_dict()
        }
    else:
        state_dicts = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

    torch.save(state_dicts, save_path + '.pkl')

def load_model(file_path, model, optimizer = None, lr_scheduler = None):
    state_dicts = torch.load(file_path, map_location="cpu")
    model.load_state_dict(state_dicts["model"])
    if optimizer:
        optimizer.load_state_dict(state_dicts["optimizer"])
    if lr_scheduler:
        lr_scheduler.load_state_dict(state_dicts["scheduler"])

task_name = "Vanila_GAN_on_MNIST"
model_name = "Vanila_GAN"
optimizer_name = 'Adam'
lr = 0.0002
weight_decay = 1e-4
step_size = 200
gamma = 0.5
batch_size = 256
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
load_checkpoint = False
mutil_gpu = False
device_ids = ["cuda:0", "cuda:1"]
epochs = 500

logging.basicConfig(filename="{}.log".format(task_name), level=logging.INFO)

logging.info(
    """{}:
    - model name: {}
    - optimizer: {}
    - learning rate: {}
    - weight_decay: {}
    - step_size: {}
    - gamma: {}
    - batch size: {}
    - device : {}
    - epochs: {}
    - load_checkpoint: {}
    - mutil_gpu: {}
    - gpus: {}
 """.format(
        task_name, 
        model_name, 
        optimizer_name, 
        lr, 
        weight_decay,
        step_size,
        gamma,
        batch_size,
        device, 
        epochs,
        load_checkpoint,
        mutil_gpu,
        device_ids)
)
print("""{}:
    - model name: {}
    - optimizer: {}
    - learning rate: {}
    - weight_decay: {}
    - step_size: {}
    - gamma: {}
    - batch size: {}
    - device : {}
    - epochs: {}
    - load_checkpoint: {}
    - mutil_gpu: {}
    - gpus: {}
 """.format(
        task_name, 
        model_name, 
        optimizer_name, 
        lr, 
        weight_decay,
        step_size,
        gamma,
        batch_size,
        device, 
        epochs,
        load_checkpoint,
        mutil_gpu,
        device_ids))

if __name__ == "__main__":
    img_size = (28, 28, 1)
    latent_dim = 128

    writer = SummaryWriter()

    discriminator = Discriminator(img_size = img_size)
    generator = Generator(latent_dim = latent_dim, output_size = img_size)

    optimizer_G = getattr(optim, optimizer_name)(generator.parameters(), lr=lr, betas = (0.5, 0.999), weight_decay=weight_decay)
    optimizer_D = getattr(optim, optimizer_name)(discriminator.parameters(), lr=lr, betas = (0.5, 0.999), weight_decay=weight_decay)
    # lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma) 
    # if load_checkpoint:
    #     load_model("model_checkpoint/check_point.pkl", model, optimizer, lr_scheduler)

    if mutil_gpu:
        discriminator = nn.DataParallel(discriminator, device_ids, device)
        generator = nn.DataParallel(generator, device_ids, device)
    discriminator = discriminator.to(device=device)
    generator = generator.to(device=device)
    


    mnist = torchvision.datasets.MNIST(root = "mnist", train=True, download=True, transform=transforms.Compose(
            [
                transforms.ToTensor(), 
                transforms.Normalize([0.5], [0.5])
            ]
        ))
    dataLoaders = {"train":torch.utils.data.DataLoader(mnist,
            batch_size=batch_size, shuffle=True, num_workers= 0, pin_memory=True, drop_last=False)}


    train(
        model=(generator, discriminator), 
        optimizer=(optimizer_G, optimizer_D), 
        lr_scheduler=None, 
        dataloaders=dataLoaders, 
        device=device,
        epochs=epochs
        )






    
    

