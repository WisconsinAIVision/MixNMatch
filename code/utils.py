from config import cfg
import numpy as np
import os
import time
import torch.backends.cudnn as cudnn
import torch
import torchvision.utils as vutils
import sys
import shutil
import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    
    if classname.find('Conv') != -1:
        nn.init.orthogonal_(m.weight.data, 1.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def make_output_dir():
    root_path = '../output'
    args = sys.argv[1:]
    if len(args)==0:
        raise RuntimeError('output folder must be specified')
    new_output = args[0]
    path = os.path.join(root_path, new_output)
    if os.path.exists(path):
        if len(args)==2 and args[1]=='-f': 
            print('WARNING: experiment directory exists, it has been erased and recreated') 
            shutil.rmtree(path)
            os.makedirs(path)
        else:
            print('WARNING: experiment directory exists, it will be erased and recreated in 3s')
            time.sleep(3)
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)
    return path


def child_to_parent(child_c_code, classes_child, classes_parent):

    ratio = classes_child / classes_parent
    arg_parent = torch.argmax(child_c_code,  dim=1) / ratio
    parent_c_code = torch.zeros([child_c_code.size(0), classes_parent]).cuda()
    for i in range(child_c_code.size(0)):
        parent_c_code[i][ int(arg_parent[i]) ] = 1
    return parent_c_code



def save_img_results(imgs_tcpu, fake_imgs, count, image_dir, nrow=8):

    num = cfg.TRAIN.VIS_COUNT*8888

    if imgs_tcpu is not None:
        real_img = imgs_tcpu[:][0:num]
        vutils.save_image(
            real_img, '%s/real_samples%09d.png' % (image_dir, count),
            scale_each=True, normalize=True, nrow=nrow)
        real_img_set = vutils.make_grid(real_img).numpy()
        real_img_set = np.transpose(real_img_set, (1, 2, 0))
        real_img_set = real_img_set * 255
        real_img_set = real_img_set.astype(np.uint8)

    if fake_imgs is not None:
        
        for i in range(len(fake_imgs)):
            fake_img = fake_imgs[i][0:num]

            vutils.save_image(
                fake_img.data, '%s/count_%09d_fake_samples%d.png' %
                (image_dir, count, i), scale_each=True, normalize=True, nrow=nrow)

            fake_img_set = vutils.make_grid(fake_img.data).cpu().numpy()

            fake_img_set = np.transpose(fake_img_set, (1, 2, 0))
            fake_img_set = (fake_img_set + 1) * 255 / 2
            fake_img_set = fake_img_set.astype(np.uint8)
    
def zero_last_layer(encoder):
    encoder.model_z[4].weight.data.fill_(0.0)
    encoder.model_z[4].bias.data.fill_(0.0)
    return encoder



def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0):
    # adapted from cyclegan 
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
     

    Returns the gradient penalty loss
    """

    if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
        interpolatesv = real_data
    elif type == 'fake':
        interpolatesv = fake_data
    elif type == 'mixed':
        interpolatesv = []
        for i in range( len(real_data) ):
            alpha = torch.rand(real_data[i].shape[0], 1)
            alpha = alpha.expand(real_data[i].shape[0], real_data[i].nelement() // real_data[i].shape[0]).contiguous().view(*real_data[i].shape)
            alpha = alpha.to(device)
            interpolatesv.append(  alpha*real_data[i] + ((1-alpha)*fake_data[i])  )
    else:
        raise NotImplementedError('{} not implemented'.format(type))
    
    # require grad
    for i in range( len(interpolatesv) ):
        interpolatesv[i].requires_grad_(True)
    
    # feed into D
    disc_interpolates = netD(*interpolatesv)

    # cal penalty

    gradient_penalty = 0
    for i in range( len(disc_interpolates) ):
        for j in range( len(interpolatesv) ):
            gradients = torch.autograd.grad(outputs=disc_interpolates[i], inputs=interpolatesv[j],
                                            grad_outputs=torch.ones(disc_interpolates[i].size()).to(device),
                                            create_graph=True, retain_graph=True, only_inputs=True, allow_unused=True)
            if gradients[0] is not None:  # it will return None if input is not used in this output (allow unused)
                gradients = gradients[0].view(real_data[j].size(0), -1)  # flat the data
                gradient_penalty += (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean()        # added eps
        
    return gradient_penalty








