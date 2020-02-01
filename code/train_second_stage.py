from config import cfg
import os
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import torch.optim as optim
from model_train import G_NET, Encoder, Dis_Dis, FeatureExtractor 
from datasets import get_dataloader
import random
import torch.nn.functional as F
from utils import *
cudnn.benchmark = True
device = torch.device("cuda:" + cfg.GPU_ID)






# ################## Shared functions ###################




def define_optimizers(  extractor, dis_dis ):
    optimizerEX = optim.Adam( extractor.parameters(), lr=2e-4, betas=(0.5, 0.999) )
    optimizerDD = optim.Adam( dis_dis.parameters(), lr=2e-5, betas=(0.5, 0.999) )   
    return  optimizerEX, optimizerDD



def load_network():

    gpus = [int(ix) for ix in cfg.GPU_ID.split(',')]

 
    netG = G_NET()
    netG = torch.nn.DataParallel(netG, device_ids=gpus)
    netG.load_state_dict( torch.load( G_DIR )  )  

    encoder = Encoder()
    encoder = torch.nn.DataParallel(encoder, device_ids=gpus)
    encoder.load_state_dict( torch.load( E_DIR ) )

    extractor = FeatureExtractor(3,16) 
    extractor.apply(weights_init)
    extractor = torch.nn.DataParallel(extractor , device_ids=gpus)

    dis_dis = Dis_Dis(16)
    dis_dis.apply(weights_init)
    dis_dis = torch.nn.DataParallel(dis_dis , device_ids=gpus)

    netG.to(device)  
    encoder.to(device)  
    extractor.to(device)
    dis_dis.to(device)

    return netG,  encoder, extractor, dis_dis
  


def save_model(  dis_dis, extractor, epoch, model_dir):
    torch.save( extractor.state_dict(), '%s/EX_%d.pth' % (model_dir, epoch))



class Trainer(object):
    def __init__(self, output_dir):

        # make dir for all kinds of output 
        self.model_dir = os.path.join(output_dir , 'Model')
        os.makedirs(self.model_dir)
        self.image_dir = os.path.join(output_dir , 'Image')
        os.makedirs(self.image_dir)

        # make dataloader 
        self.dataloader = get_dataloader()
 
        # other variables
        self.batch_size = cfg.TRAIN.BATCH_SIZE 

        # get fixed images used for comparison for each epoch 
        self.fixed_image = self.prepare_data(  next(iter(self.dataloader)) )[0]
        save_img_results( self.fixed_image.cpu(), None, -1, self.image_dir )
    


    

    def prepare_data(self, data):

        real_img = data[1]       
        real_img = real_img.to(device)

        real_z = torch.FloatTensor( self.batch_size, cfg.GAN.Z_DIM ).normal_(0, 1).to(device)

        if random.uniform(0, 1)<0.2: 
            real_p = torch.softmax( torch.FloatTensor( self.batch_size, cfg.SUPER_CATEGORIES ).normal_(0, 1), dim =1).to(device)
        else:
            real_p = torch.zeros( self.batch_size, cfg.SUPER_CATEGORIES ).to(device)
            idxs = torch.LongTensor( self.batch_size ).random_(0, cfg.SUPER_CATEGORIES)
            for i, idx in enumerate(idxs):
                real_p[i,idx] = 1
        real_c = torch.zeros( self.batch_size, cfg.FINE_GRAINED_CATEGORIES ).to(device)
        idxs = torch.LongTensor( self.batch_size ).random_(0, cfg.FINE_GRAINED_CATEGORIES)
        for i, idx in enumerate(idxs):
            real_c[i,idx] = 1
        real_b = torch.zeros( self.batch_size, cfg.FINE_GRAINED_CATEGORIES ).to(device)
        idxs = torch.LongTensor( self.batch_size ).random_(0, cfg.FINE_GRAINED_CATEGORIES)
        for i, idx in enumerate(idxs):
            real_b[i,idx] = 1

        return  real_img, real_z, real_b, real_p, real_c 


    def train(self):

        # prepare net, optimizer and loss
        self.netG, self.encoder, self.extractor, self.dis_dis = load_network()   
        self.netG.eval()
        self.encoder.eval()

        self.optimizerEX, self.optimizerDD = define_optimizers(  self.extractor, self.dis_dis )    
        self.RF_loss = nn.BCELoss() 
        self.L1 = nn.L1Loss()
       

        for epoch in range(cfg.TRAIN.SECOND_MAX_EPOCH):
          

            for data in self.dataloader:          
                
                # prepare data              
                real_img, real_z, real_b, real_p, real_c   = self.prepare_data(data)


                # forward to get real distribution 
                with torch.no_grad():
                    real_distribution, real_fake_image = self.netG( real_z, real_c, real_p, real_b, 'code',  only=True )


                # forward feature extractor
                fake_distribution = self.extractor(real_img)                

                # update DD
                self.optimizerDD.zero_grad()
                fake_pred = self.dis_dis( fake_distribution.detach() )
                real_pred = self.dis_dis( real_distribution )
                DD_loss = self.RF_loss( fake_pred,torch.zeros_like(fake_pred) ) + self.RF_loss( real_pred,torch.ones_like(real_pred) )
                DD_loss.backward()
                self.optimizerDD.step()

                # update extractor
                self.optimizerEX.zero_grad()             
                fake_pred = self.dis_dis( fake_distribution )
                l1loss = self.L1( self.extractor(real_fake_image), real_distribution)
                EX_loss = self.RF_loss( fake_pred,torch.ones_like(fake_pred) )
                (EX_loss+l1loss).backward()
                self.optimizerEX.step()

           
            # Save model&image for each epoch 
            self.extractor.eval()
            with torch.no_grad():   
                code_z, code_b, _, code_c = self.encoder( self.fixed_image,'softmax')   
                feat_p = self.extractor(self.fixed_image)
                fake_imgs, fg_imgs, mk_imgs, fg_mk = self.netG( code_z, code_c, feat_p, code_b, 'feature')  
                save_img_results(None, ( fake_imgs+fg_imgs+mk_imgs+fg_mk), epoch, self.image_dir )
            self.extractor.train()      
            save_model(  self.dis_dis  ,self.extractor, 0, self.model_dir )   
            print( str(epoch)+'th epoch finished')









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

    G_DIR = sys.argv[2]
    E_DIR = sys.argv[3]

    
    trainer = Trainer(output_dir)   
    print('start training now')
    trainer.train()
      
        
