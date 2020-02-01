from config import cfg
import os
import time
from PIL import Image
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from model_train import G_NET, BACKGROUND_D, PARENT_D, CHILD_D, Encoder, Bi_Dis
from datasets import get_dataloader
import random
from utils import *
from itertools import chain
from copy import deepcopy
cudnn.benchmark = True
device = torch.device("cuda:" + cfg.GPU_ID)




################### Useful functions ###################



def define_optimizers( netG, netsD, BD, encoder ):

    # define optimizer for D
    optimizersD = []  
    for i in range(3):
        if i == 0 or i==2:
            optimizersD.append( optim.Adam(netsD[i].parameters(), lr=2e-4, betas=(0.5, 0.999)) )   
        else:
            optimizersD.append(None)
  
    optimizerBD = optim.Adam( BD.parameters(), lr=2e-4, betas=(0.5, 0.999))
  
    params = chain( netG.parameters(), encoder.parameters(), netsD[1].parameters(), netsD[2].module.code_logits.parameters() )       
    optimizerGE = optim.Adam(  params , lr=2e-4, betas=(0.5, 0.999) ) 
 
   
    return optimizersD, optimizerBD, optimizerGE



def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def copy_G_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten


class CrossEntropy():
    def __init__(self):
        self.code_loss = nn.CrossEntropyLoss() 
    def __call__(self, prediction, label):
        # check label if hard (onehot)
        if label.max(dim=1)[0].min() == 1:
            return self.code_loss(prediction, torch.nonzero( label.long() )[:, 1] )
        else:        
            log_prediction = torch.log_softmax(prediction, dim=1)    
            return (- log_prediction*label).sum(dim=1).mean(dim=0)



def load_network():

    gpus = [int(ix) for ix in cfg.GPU_ID.split(',')]
 
    netG = G_NET()
    netG.apply(weights_init)
    netG = torch.nn.DataParallel(netG, device_ids=gpus)
  
    netsD = [ BACKGROUND_D(), PARENT_D(), CHILD_D() ]
    for i in range(len(netsD)):
        netsD[i].apply(weights_init)
        netsD[i] = torch.nn.DataParallel(netsD[i], device_ids=gpus)       

    BD = Bi_Dis()
    BD = torch.nn.DataParallel(BD, device_ids=gpus)     

    encoder = Encoder()
    encoder.apply(weights_init)
    encoder = torch.nn.DataParallel(encoder, device_ids=gpus)
   
    netG.to(device)  
    encoder.to(device)  
    BD.to(device)
    for i in range(3):
        netsD[i].to(device)

    return netG, netsD, BD, encoder
  


def save_model( encoder, myG, D0, D1, D2, BD, epoch, model_dir):
    torch.save(encoder.state_dict(), '%s/E_%d.pth' % (model_dir, epoch))
    torch.save(myG.state_dict(), '%s/G_%d.pth' % (model_dir, epoch))
    torch.save(D0.state_dict(), '%s/D0_%d.pth' % (model_dir, epoch))
    torch.save(D1.state_dict(), '%s/D1_%d.pth' % (model_dir, epoch))
    torch.save(D2.state_dict(), '%s/D2_%d.pth' % (model_dir, epoch))
    torch.save(BD.state_dict(), '%s/BD_%d.pth' % (model_dir, epoch))



def save_opt( optimizerGE,  optimizerD0,  optimizerD2, optimizerBD, epoch, opt_dir):
    torch.save(optimizerGE.state_dict(), '%s/GE_%d.pth' % (opt_dir, epoch))
    torch.save(optimizerD0.state_dict(), '%s/D0_%d.pth' % (opt_dir, epoch))
    torch.save(optimizerD2.state_dict(), '%s/D2_%d.pth' % (opt_dir, epoch))
    torch.save(optimizerBD.state_dict(), '%s/BD_%d.pth' % (opt_dir, epoch))

    

############################### Trainer ############################


class Trainer(object):
    def __init__(self, output_dir):

        # make dir for all kinds of output 
        self.model_dir = os.path.join(output_dir , 'Model')
        os.makedirs(self.model_dir)
        self.image_dir = os.path.join(output_dir , 'Image')
        os.makedirs(self.image_dir)
        self.opt_dir = os.path.join(output_dir , 'Opt')
        os.makedirs(self.opt_dir)

        # make dataloader and code buffer 
        self.dataloader = get_dataloader()
      
        # other variables
        self.batch_size = cfg.TRAIN.BATCH_SIZE 
        self.patch_stride = 4.0 
        self.n_out = 24
        self.recp_field = 34       

        # get fixed images used for comparison for each epoch 
        self.fixed_image =  self.prepare_data(  next(iter(self.dataloader)) )[1]
        save_img_results( self.fixed_image.cpu(), None, -1, self.image_dir )

 
    def prepare_code(self):

        free_z = torch.FloatTensor( self.batch_size, cfg.GAN.Z_DIM ).normal_(0, 1).to(device)

        free_c = torch.zeros( self.batch_size, cfg.FINE_GRAINED_CATEGORIES ).to(device)
        idxs = torch.LongTensor( self.batch_size ).random_(0, cfg.FINE_GRAINED_CATEGORIES)
        for i, idx in enumerate(idxs):
            free_c[i,idx] = 1
        free_p = torch.zeros( self.batch_size, cfg.SUPER_CATEGORIES ).to(device)
        idxs = torch.LongTensor( self.batch_size ).random_(0, cfg.SUPER_CATEGORIES)
        for i, idx in enumerate(idxs):
            free_p[i,idx] = 1
        free_b = torch.zeros( self.batch_size, cfg.FINE_GRAINED_CATEGORIES ).to(device)
        idxs = torch.LongTensor( self.batch_size ).random_( 0, cfg.FINE_GRAINED_CATEGORIES )
        for i, idx in enumerate(idxs):
            free_b[i,idx] = 1

        return free_z, free_b, free_p, free_c

    

    def prepare_data(self, data):

        real_img126, real_img, real_c, _, warped_bbox = data 
        real_img126 = real_img126.to(device)
        real_img = real_img.to(device)
        for i in range(len(warped_bbox)):
            warped_bbox[i] = warped_bbox[i].float().to(device)

        real_p = child_to_parent(real_c, cfg.FINE_GRAINED_CATEGORIES, cfg.SUPER_CATEGORIES ).to(device)
        real_z = torch.FloatTensor( self.batch_size, cfg.GAN.Z_DIM ).normal_(0, 1).to(device) 
        real_c = real_c.to(device)
        real_b = real_c             

        return  real_img126, real_img, real_z, real_b, real_p, real_c, warped_bbox


    def train_Dnet(self, idx):

        assert(idx == 0 or idx ==2)
  
        # choose net and opt  
        netD, optD = self.netsD[idx], self.optimizersD[idx]
        netD.zero_grad()

        # choose real and fake images
        if idx == 0:
            real_img = self.real_img126
            fake_img = self.fake_imgs[0]
        elif idx == 2:
            real_img = self.real_img
            fake_img = self.fake_imgs[2]   

        # # # # # # # #for background stage now  # # # # # # #
        if idx == 0:

            # go throung D net to get prediction
            class_prediction, real_prediction = netD(real_img) 
            _, fake_prediction = netD( fake_img.detach() )   

            real_label = torch.ones_like(real_prediction)
            fake_label = torch.zeros_like(fake_prediction)     
            weights_real = torch.ones_like(real_prediction)
            
            for i in range( self.batch_size ):

                x1 = self.warped_bbox[0][i]
                x2 = self.warped_bbox[2][i]
                y1 = self.warped_bbox[1][i]
                y2 = self.warped_bbox[3][i]

                a1 = max(torch.tensor(0).float().to(device), torch.ceil((x1 - self.recp_field)/self.patch_stride))
                a2 = min(torch.tensor(self.n_out - 1).float().to(device), torch.floor((self.n_out - 1) - ((126 - self.recp_field) - x2)/self.patch_stride)) + 1
                b1 = max(torch.tensor(0).float().to(device), torch.ceil( (y1 - self.recp_field)/self.patch_stride))
                b2 = min(torch.tensor(self.n_out - 1).float().to(device), torch.floor((self.n_out - 1) - ((126 - self.recp_field) - y2)/self.patch_stride)) + 1

                if (x1 != x2 and y1 != y2):
                    weights_real[i, :, a1.type(torch.int): a2.type(torch.int), b1.type(torch.int): b2.type(torch.int)] = 0.0

            norm_fact_real = weights_real.sum()
            norm_fact_fake = weights_real.shape[0]*weights_real.shape[1]*weights_real.shape[2]*weights_real.shape[3]

            # Real/Fake loss for 'real background' (on patch level)
            real_prediction_loss = self.RF_loss_un( real_prediction, real_label )
            # Masking output units which correspond to receptive fields which lie within the bounding box
            real_prediction_loss = torch.mul(real_prediction_loss, weights_real).mean()
            # Normalizing the real/fake loss for background after accounting the number of masked members in the output.
            if (norm_fact_real > 0):
                real_prediction_loss = real_prediction_loss * ((norm_fact_fake * 1.0) / (norm_fact_real * 1.0))

            # Real/Fake loss for 'fake background' (on patch level)
            fake_prediction_loss = self.RF_loss_un(fake_prediction, fake_label).mean()        
          
            # Background/foreground classification loss
            class_prediction_loss = self.RF_loss_un( class_prediction, weights_real ).mean()  

            # add three losses together 
            D_loss = cfg.TRAIN.BG_LOSS_WT*(real_prediction_loss + fake_prediction_loss) + class_prediction_loss
      

        # # # # # # # #for child stage now (only real/fake discriminator)  # # # # # # # 
        if idx == 2:

            # go through D net to get data
            _, real_prediction = netD(real_img) 
            _, fake_prediction = netD( fake_img.detach() )

            # get real/fake lables
            real_label = torch.ones_like(real_prediction)
            fake_label = torch.zeros_like(fake_prediction) 
 
            # get loss 
            real_prediction_loss = self.RF_loss(real_prediction, real_label)         
            fake_prediction_loss = self.RF_loss(fake_prediction, fake_label)
            D_loss = real_prediction_loss+fake_prediction_loss

        D_loss.backward()
        optD.step()



    def train_BD(self):

        self.optimizerBD.zero_grad()

        # make prediction on pairs 
        pred_enc_z, pred_enc_b, pred_enc_p, pred_enc_c = self.BD(  self.real_img, self.fake_z.detach(), self.fake_b.detach(), self.fake_p.detach(), self.fake_c.detach() )
        pred_gen_z, pred_gen_b, pred_gen_p, pred_gen_c = self.BD(  self.fake_imgs[2].detach(), self.real_z, self.real_b, self.real_p, self.real_c )
       
        real_data = [ self.real_img, self.fake_z.detach(), self.fake_b.detach(), self.fake_p.detach(), self.fake_c.detach() ]
        fake_data = [ self.fake_imgs[2].detach(), self.real_z, self.real_b, self.real_p, self.real_c ]
        penalty = cal_gradient_penalty( self.BD, real_data, fake_data, device, type='mixed', constant=1.0)

        D_loss =  -( pred_enc_z.mean()+pred_enc_b.mean()+pred_enc_p.mean()+pred_enc_c.mean()  ) + ( pred_gen_z.mean()+pred_gen_b.mean()+pred_gen_p.mean()+pred_gen_c.mean() ) + penalty*10
        D_loss.backward()
        self.optimizerBD.step()
      


    def train_EG(self):

        self.optimizerGE.zero_grad()

        # reconstruct code and calculate loss 
        self.rec_p, _ = self.netsD[1]( self.fg_mk[0])
        self.rec_c, _ = self.netsD[2]( self.fg_mk[1])
        p_code_loss = self.CE( self.rec_p , self.real_p )
        c_code_loss = self.CE( self.rec_c,  self.real_c )

        # pred code and calculate loss (here no code constrain)
        free_z, free_b, free_p, free_c = self.prepare_code()
        with torch.no_grad():
            free_fake_imgs, _, _, _ = self.netG( free_z, free_c, free_p, free_b, 'code'  )                   
        pred_z, pred_b, pred_p, pred_c = self.encoder( free_fake_imgs[2].detach(),   'logits' )
        z_pred_loss = self.L1( pred_z , free_z )
        b_pred_loss = self.CE( pred_b , free_b )
        p_pred_loss = self.CE( pred_p , free_p )
        c_pred_loss = self.CE( pred_c,  free_c )        
    
    
        # aux and backgroud real/fake loss
        self.bg_class_pred, self.bg_rf_pred = self.netsD[0]( self.fake_imgs[0] ) 
        bg_rf_loss = self.RF_loss( self.bg_rf_pred, torch.ones_like( self.bg_rf_pred ) )*cfg.TRAIN.BG_LOSS_WT
        bg_class_loss = self.RF_loss( self.bg_class_pred, torch.ones_like( self.bg_class_pred ) )

        # child image real/fake loss  
        _, self.child_rf_pred = self.netsD[2]( self.fake_imgs[-1] )
        child_rf_loss = self.RF_loss( self.child_rf_pred, torch.ones_like(self.child_rf_pred) )
  
        # fool BD loss
        pred_enc_z, pred_enc_b, pred_enc_p, pred_enc_c = self.BD(  self.real_img, self.fake_z, self.fake_b, self.fake_p, self.fake_c )
        pred_gen_z, pred_gen_b, pred_gen_p, pred_gen_c = self.BD(  self.fake_imgs[2], self.real_z, self.real_b, self.real_p, self.real_c )
        fool_BD_loss = ( pred_enc_z.mean()+pred_enc_b.mean()+pred_enc_p.mean()+pred_enc_c.mean()  ) - ( pred_gen_z.mean()+pred_gen_b.mean()+pred_gen_p.mean()+pred_gen_c.mean() ) 
             
        EG_loss =  (p_code_loss+c_code_loss) + (bg_rf_loss+bg_class_loss) + child_rf_loss + fool_BD_loss + (5*z_pred_loss+5*b_pred_loss+10*p_pred_loss+10*c_pred_loss)
        EG_loss.backward()

        self.optimizerGE.step()




    def train(self):

        # prepare net, optimizer and loss
        self.netG, self.netsD, self.BD, self.encoder = load_network()   
        self.optimizersD, self.optimizerBD, self.optimizerGE = define_optimizers( self.netG, self.netsD, self.BD, self.encoder )
        self.RF_loss_un = nn.BCELoss(reduction='none')
        self.RF_loss = nn.BCELoss()   
        self.CE = CrossEntropy()        
        self.L1 = nn.L1Loss()
    


        # get init avg_G (the param in avg_G is what we want)
        avg_param_G = copy_G_params(self.netG) 

        for epoch in range(cfg.TRAIN.FIRST_MAX_EPOCH):              

            for data in self.dataloader:  
                     
                # prepare data              
                self.real_img126, self.real_img, self.real_z, self.real_b, self.real_p, self.real_c, self.warped_bbox = self.prepare_data(data)

                # forward for both E and G
                self.fake_z, self.fake_b, self.fake_p, self.fake_c = self.encoder( self.real_img, 'softmax' )              
                self.fake_imgs, self.fg_imgs, self.mk_imgs, self.fg_mk = self.netG( self.real_z, self.real_c, self.real_p, self.real_b, 'code'  )
                                
                # Update Discriminator networks in FineGAN      
                self.train_Dnet(0)
                self.train_Dnet(2)

                # Update Bi Discriminator
                self.train_BD()

                # Update Encoder and G network
                self.train_EG()
                for avg_p, p in zip( avg_param_G, self.netG.parameters() ):
                    avg_p.mul_(0.999).add_(0.001, p.data)

        
            # Save model&image for each epoch  
            backup_para = copy_G_params(self.netG)   
            load_params(self.netG, avg_param_G)

            self.encoder.eval()   
            self.netG.eval()    
            with torch.no_grad():   
                self.code_z, self.code_b, self.code_p, self.code_c = self.encoder( self.fixed_image,'softmax')   
                self.fake_imgs, self.fg_imgs, self.mk_imgs, self.fg_mk = self.netG(self.code_z, self.code_c, self.code_p, self.code_b, 'code')  
                save_img_results(None, (self.fake_imgs+self.fg_imgs+self.mk_imgs+self.fg_mk), epoch, self.image_dir)            
            self.encoder.train() 
            self.netG.train()            
        
        

            backup_para = copy_G_params(self.netG)   
            load_params(self.netG, avg_param_G)
            save_model( self.encoder, self.netG, self.netsD[0], self.netsD[1], self.netsD[2], self.BD, 0, self.model_dir )   
            save_opt(  self.optimizerGE,  self.optimizersD[0], self.optimizersD[2], self.optimizerBD,  0, self.opt_dir )   
            load_params(self.netG, backup_para)        

            print( str(epoch)+'th epoch finished' )










if __name__ == "__main__":

    
    
    manualSeed = random.randint(1, 10000)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)



    # prepare output folder for this running and save all files 
    output_dir = make_output_dir()
    shutil.copy2( sys.argv[0], output_dir)
    shutil.copy2( 'model_train.py', output_dir)
    shutil.copy2( 'config.py', output_dir)
    shutil.copy2( 'utils.py', output_dir)
    shutil.copy2( 'datasets.py', output_dir)

    
    trainer = Trainer(output_dir)   
    print('start training now')
    trainer.train()
      
        
