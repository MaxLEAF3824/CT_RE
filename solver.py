import os
import numpy as np
import time
import datetime
import torch
import torchvision
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from evaluation import *
from network import U_Net,R2U_Net,AttU_Net,R2AttU_Net
import csv
from tqdm import tqdm
import wandb

wandb.init(project="ct-reconstruction")

class Solver(object):
    def __init__(self, config, train_loader, valid_loader, test_loader):

        # Data loader
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        # Models
        self.unet = None
        self.optimizer = None
        self.img_ch = config.img_ch
        self.output_ch = config.output_ch
        self.criterion = torch.nn.MSELoss()
        # self.criterion = torch.nn.BCELoss()
        self.augmentation_prob = config.augmentation_prob

        # Hyper-parameters
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        # Training settings
        self.num_epochs = config.num_epochs
        self.num_epochs_decay = config.num_epochs_decay
        self.batch_size = config.batch_size

        # Step size
        self.log_step = config.log_step
        self.val_step = config.val_step

        # Path
        self.model_path = config.model_path
        self.result_path = config.result_path
        self.mode = config.mode

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = config.model_type
        self.t = config.t
        self.dp = config.dp
        self.build_model()

    def build_model(self):
        """Build generator and discriminator."""
        if self.model_type =='U_Net':
            self.unet = U_Net(img_ch=self.img_ch,output_ch=1)
        elif self.model_type =='R2U_Net':
            self.unet = R2U_Net(img_ch=self.img_ch,output_ch=1,t=self.t)
        elif self.model_type =='AttU_Net':
            self.unet = AttU_Net(img_ch=self.img_ch,output_ch=1)
        elif self.model_type == 'R2AttU_Net':
            self.unet = R2AttU_Net(img_ch=self.img_ch,output_ch=1,t=self.t)
            

        self.optimizer = optim.Adam(list(self.unet.parameters()),
                                      self.lr, [self.beta1, self.beta2])
        self.unet.to(self.device)
        if self.dp:
            self.unet = torch.nn.DataParallel(self.unet)
        # self.print_network(self.unet, self.model_type)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def to_data(self, x):
        """Convert variable to tensor."""
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data

    def update_lr(self, g_lr, d_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def reset_grad(self):
        """Zero the gradient buffers."""
        self.unet.zero_grad()

    def compute_accuracy(self,SR,GT):
        SR_flat = SR.view(-1)
        GT_flat = GT.view(-1)

        acc = GT_flat.data.cpu()==(SR_flat.data.cpu()>0.5)

    def tensor2img(self,x):
        img = (x[:,0,:,:]>x[:,1,:,:]).float()
        img = img*255
        return img

    def train(self):
        """Train encoder, generator and discriminator."""

        #====================================== Training ===========================================#
        unet_path = os.path.join(self.model_path, f'{self.model_type}-{self.num_epochs}-{self.lr:.4f}-{self.num_epochs_decay}-{self.augmentation_prob}.pkl')

        # Train for Encoder
        lr = self.lr
        best_unet_score = 0.
        
        for epoch in range(self.num_epochs):
            self.unet.train(True)
            epoch_loss = 0
            
            psnr = 0.    # PSNR
            len_dl = len(self.train_loader)
            for i, (images, GT) in tqdm(enumerate(self.train_loader),total=len_dl):
                # GT : Ground Truth
                images = images.to(self.device)
                GT = GT.to(self.device)

                # SR : Segmentation Result
                SR = self.unet(images)
                # SR = F.sigmoid(SR)
                SR_flat = SR.view(SR.size(0),-1)
                GT_flat = GT.view(GT.size(0),-1)
                loss = self.criterion(SR_flat,GT_flat)
                epoch_loss += loss.item()

                # Backprop + optimize
                self.reset_grad()
                loss.backward()
                self.optimizer.step()

                psnr += get_psnr(SR,GT)

            psnr = psnr/len_dl
            
            # Print the log info
            train_psnr = psnr
            print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {epoch_loss:.4f} \n[Training] PSNR: {train_psnr:.4f}')
            
            # Decay learning rate
            if (epoch+1) > (self.num_epochs - self.num_epochs_decay):
                lr -= (self.lr / float(self.num_epochs_decay))
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                print (f'Decay learning rate to lr: {lr}.')
            
            
            #===================================== Validation ====================================#
            self.unet.train(False)
            self.unet.eval()

            psnr=0.
            ssim=0.
            len_dl = len(self.valid_loader)
            for i, (images, GT) in tqdm(enumerate(self.valid_loader),total=len_dl):
                images = images.to(self.device)
                GT = GT.to(self.device)
                SR = self.unet(images)
                # SR = F.sigmoid(SR)
                psnr += get_psnr(SR,GT)
                ssim += get_ssim(SR,GT)
                
            psnr = psnr/len_dl
            ssim = ssim/len_dl
            unet_score = psnr
            
            print(f'[Validation] PSNR: {psnr:.4f}, SSIM: {ssim:.4f}')
            wandb.log({"epoch": epoch, "loss": epoch_loss, "train_psnr": train_psnr,"valid_psnr": psnr, "valid_ssim": ssim, "lr": lr})

            # Save Best U-Net model
            if unet_score > best_unet_score:
                best_unet_score = unet_score
                best_unet = self.unet.state_dict()
                if self.dp:
                    best_unet = self.unet.module.state_dict()
                print('Best %s model score : %.4f'%(self.model_type,best_unet_score))
                torch.save(best_unet,unet_path)
        
    def test(self):
        self.unet.train(False)
        self.unet.eval()
        psnr = 0.
        ssim = 0.
        len_dl = len(self.test_loader)
        for i, (images, GT) in tqdm(enumerate(self.test_loader),total=len_dl):
            images = images.to(self.device)
            GT = GT.to(self.device)
            SR = self.unet(images)
            # SR = F.sigmoid(SR)
            psnr += get_psnr(SR,GT)
            ssim += get_ssim(SR,GT)
                
        psnr = psnr/len_dl
        ssim = ssim/len_dl

        print(f"[Test] PSNR: {psnr:.4f}, SSIM: {ssim:.4f}")