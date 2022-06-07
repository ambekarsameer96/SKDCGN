import argparse
from datetime import datetime
from pathlib import Path
from xmlrpc.client import Boolean
import numpy as np
from tqdm import tqdm
import os 
import repackage
repackage.up()
import cv2
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from mnists.config import get_cfg_defaults
from mnists.dataloader import get_dataloaders
from mnists.models import CGN, DiscLin, DiscConv
from utils import save_cfg, load_cfg, children, hook_outputs, Optimizers
from shared.losses import BinaryLoss, PerceptualLoss

def save(x, path, n_row, sz=64):
    x = F.interpolate(x, (sz, sz))
    save_image(x.data, path, nrow=n_row, normalize=True, padding=2)

def sample_image(cgn, sample_path, batches_done, device, n_row=3, n_classes=10):
    """Saves a grid of generated digits"""
    y_gen = np.arange(n_classes).repeat(n_row)
    y_gen = torch.LongTensor(y_gen).to(device)
    mask, foreground, background = cgn(y_gen)
    x_gen = mask * foreground + (1 - mask) * background

    save(x_gen.data, f"{sample_path}/0_{batches_done:d}_x_gen.png", n_row)
    save(mask.data, f"{sample_path}/1_{batches_done:d}_mask.png", n_row)
    save(foreground.data, f"{sample_path}/2_{batches_done:d}_foreground.png", n_row)
    save(background.data, f"{sample_path}/3_{batches_done:d}_background.png", n_row)
def inference(cfg, cgn, discriminator, dataloader, device):
    
    cgn.eval()
    discriminator.eval()
    fol1 = './Imgs'
    list_of_classes = [0,1,2,3,4,5,6,7,8,9]
    num_samples = 1000 
    
    x_gen_fol = fol1 + '/' + 'x_gen'
    mask_fol = fol1 + '/' + 'mask'
    foreground_fol = fol1 + '/' + 'foreground'
    background_fol = fol1 + '/' + 'background'
    if not os.path.exists(x_gen_fol):
        os.makedirs(x_gen_fol)
    if not os.path.exists(mask_fol):
        os.makedirs(mask_fol)
    if not os.path.exists(foreground_fol):
        os.makedirs(foreground_fol)
    if not os.path.exists(background_fol):
        os.makedirs(background_fol)
        
    with torch.no_grad():
        for class_num in list_of_classes:
            # 1 
            i=0
            print(class_num)
            while(i<num_samples):

                # x = data['ims'].to(device)
                # y = data['labels'].to(device)
                #y = torch.randint(cfg.MODEL.N_CLASSES, (len(y_gt),)).to(device)
                y = [class_num]
                y= torch.Tensor(y).int().to(device)
                mask, foreground, background = cgn(y)
                x_gen = mask * foreground + (1 - mask) * background

                
                x_gen_filename = x_gen_fol + '/' + 'x_gen_' + str(class_num) + '_' + str(i) + '.png'
                mask_filename = mask_fol + '/' + 'mask_' + str(class_num) + '_' + str(i) + '.png'
                foreground_filename = foreground_fol + '/' + 'foreground_' + str(class_num) + '_' + str(i) + '.png'
                background_filename = background_fol + '/' + 'background_' + str(class_num) + '_' + str(i) + '.png'
                save(x_gen.data, x_gen_filename, n_row=1)
                save(mask.data, mask_filename, n_row=1)
                save(foreground.data, foreground_filename, n_row=1)
                save(background.data, background_filename, n_row=1)
                i+=1

                # save(x_gen.data, f"{fol1}/"x_gen"{i}".png", n_row=1)
                # save(mask.data, f"{fol1}/{i:d}_mask.png", n_row=1)
                # save(foreground.data, f"{fol1}/{i:d}_foreground.png", n_row=1)
                # save(background.data, f"{fol1}/{i:d}_background.png", n_row=1)
                # i=i+1
def fit(cfg, cgn, discriminator, dataloader, opts, losses, device):

    # directories for experiments
    time_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    model_path = Path('.') / 'mnists' / 'experiments'
    model_path /= f'cgn_{cfg.TRAIN.DATASET}_{time_str}_{cfg.MODEL_NAME}'
    weights_path = model_path / 'weights'
    sample_path = model_path / 'samples'
    weights_path.mkdir(parents=True, exist_ok=True)
    sample_path.mkdir(parents=True, exist_ok=True)

    # dump config
    save_cfg(cfg, model_path / "cfg.yaml")

    # Training Loop
    L_perc, L_adv, L_binary = losses

    pbar = tqdm(range(cfg.TRAIN.EPOCHS))
    for epoch in pbar:
        for i, data in enumerate(dataloader):

            # Data and adversarial ground truths to device
            x_gt = data['ims'].to(device)
            y_gt = data['labels'].to(device)
            valid = torch.ones(len(y_gt),).to(device)
            fake = torch.zeros(len(y_gt),).to(device)

            #
            #  Train Generator
            #
            opts.zero_grad(['generator'])

            # Sample noise and labels as generator input
            y_gen = torch.randint(cfg.MODEL.N_CLASSES, (len(y_gt),)).to(device)

            # Generate a batch of images
            mask, foreground, background = cgn(y_gen)
            fol1 = './Imgs'
            if args.save_images:
                if not(os.path.exists('./Imgs')):
                    os.mkdir('Imgs')
                fol1 = './Imgs'
                mask_f1 = os.path.join(fol1, f"{fol1}/0_{i:d}_mask.png")
                foreground_f1 = os.path.join(fol1,f"{fol1}/1_{i:d}_foreground.png",)
                background_f1 = os.path.join(fol1,f"{fol1}/2_{i:d}_background.png")
                #save all images
                save_image(mask.data, f"{fol1}/0_{i:d}_mask.png",)
                save_image(foreground.data, f"{fol1}/1_{i:d}_foreground.png", )
                save_image(background.data, f"{fol1}/2_{i:d}_background.png", )
                #save the images as cv2 images 
                # cv2.imwrite(mask_f1, mask)
                # cv2.imwrite(foreground_f1, foreground)
                # cv2.imwrite( background_f1,background)

            x_gen = mask * foreground + (1 - mask) * background

            # Calc Losses
            validity = discriminator(x_gen, y_gen)

            losses_g = {}
            losses_g['adv'] = L_adv(validity, valid)
            losses_g['binary'] = L_binary(mask)
            losses_g['perc'] = L_perc(x_gen, x_gt)

            # Backprop and step
            loss_g = sum(losses_g.values())
            loss_g.backward()
            opts.step(['generator'], False)

            #
            # Train Discriminator
            #
            opts.zero_grad(['discriminator'])

            # Discriminate real and fake
            validity_real = discriminator(x_gt, y_gt)
            validity_fake = discriminator(x_gen.detach(), y_gen)

            # Losses
            losses_d = {}
            losses_d['real'] = L_adv(validity_real, valid)
            losses_d['fake'] = L_adv(validity_fake, fake)
            loss_d = sum(losses_d.values()) / 2

            # Backprop and step
            loss_d.backward()
            opts.step(['discriminator'], False)

            # Saving
            batches_done = epoch * len(dataloader) + i
            if batches_done % cfg.LOG.SAVE_ITER == 0:
                print("Saving samples and weights")
                sample_image(cgn, fol1, batches_done, device, n_row=1, n_classes=1)
                torch.save(cgn.state_dict(), f"{weights_path}/ckp_{batches_done:d}.pth")

            # Logging
            if cfg.LOG.LOSSES:
                msg = f"[Batch {i}/{len(dataloader)}]"
                msg += ''.join([f"[{k}: {v:.3f}]" for k, v in losses_d.items()])
                msg += ''.join([f"[{k}: {v:.3f}]" for k, v in losses_g.items()])
                pbar.set_description(msg)

def main(cfg):
    # model init
    cgn = CGN(n_classes=cfg.MODEL.N_CLASSES, latent_sz=cfg.MODEL.LATENT_SZ,
              ngf=cfg.MODEL.NGF, init_type=cfg.MODEL.INIT_TYPE,
              init_gain=cfg.MODEL.INIT_GAIN)
    Discriminator = DiscLin if cfg.MODEL.DISC == 'linear' else DiscConv
    discriminator = Discriminator(n_classes=cfg.MODEL.N_CLASSES, ndf=cfg.MODEL.NDF)

    # get data
    dataloader, _ = get_dataloaders(cfg.TRAIN.DATASET, cfg.TRAIN.BATCH_SIZE,
                                    cfg.TRAIN.WORKERS)

    # Loss functions
    L_adv = torch.nn.MSELoss()
    L_binary = BinaryLoss(cfg.LAMBDAS.MASK)
    L_perc = PerceptualLoss(style_wgts=cfg.LAMBDAS.PERC)
    losses = (L_perc, L_adv, L_binary)

    # Optimizers
    opts = Optimizers()
    opts.set('generator', cgn, lr=cfg.LR.LR, betas=cfg.LR.BETAS)
    opts.set('discriminator', discriminator, lr=cfg.LR.LR, betas=cfg.LR.BETAS)

    # push to device and train
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cgn = cgn.to(device)
    discriminator = discriminator.to(device)
    losses = (l.to(device) for l in losses)
    #load pretrained weights 
    # if cfg.TRAIN.PRETRAINED:
    #     ckp = torch.load(cfg.TRAIN.PRETRAINED)
    #     cgn.load_state_dict(ckp['generator'])
    #     #discriminator.load_state_dict(ckp['discriminator'])
    # load model
    model_path = os.path.join('experiments', args.dataset, 'weights', 'ckp.pth')
    print('Loading model from: ' + model_path)
    cgn.load_state_dict(torch.load(model_path))
    print('loaded model')

    #fit(cfg, cgn, discriminator, dataloader, opts, losses, device)
    inference(cfg, cgn, discriminator, dataloader, device)

def merge_args_and_cfg(args, cfg):
    cfg.MODEL_NAME = args.model_name
    cfg.LOG.SAVE_ITER = args.save_iter
    cfg.TRAIN.EPOCHS = args.epochs
    cfg.TRAIN.BATCH_SIZE = args.batch_size
    return cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='experiments\cgn_double_colored_MNIST\cfg.yaml',
                        help="path to a cfg file")
    parser.add_argument('--model_name', default='tmp',
                        help='Weights and samples will be saved under experiments/model_name')
    parser.add_argument("--save_iter", type=int, default=1000,
                        help="interval between image sampling")
    parser.add_argument("--epochs", type=int, default=3,
                        help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="size of the batches")
    parser.add_argument('--save_images', type=Boolean, default=False, help='Saving all the images in the folder ')
    parser.add_argument('--dataset', type=str, default='cgn_double_colored_MNIST', help='cgn_double_colored_MNIST, cgn_colored_MNIST, cgn_wildlife_MNIST')
    args = parser.parse_args()

    # get cfg
    cfg = load_cfg(args.cfg) if args.cfg else get_cfg_defaults()
    # add additional arguments in the argparser and in the function below
    cfg = merge_args_and_cfg(args, cfg)

    print(cfg)
    main(cfg)
