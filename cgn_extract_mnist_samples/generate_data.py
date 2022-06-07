import argparse
import warnings
from tqdm import trange
from tqdm import tqdm
import torch
import repackage
import os 
import pandas as pd
from os.path import join
import torch.nn.functional as F
from torchvision.utils import save_image
import numpy as np
repackage.up()

from mnists.train_cgn import CGN
from mnists.dataloader import get_dataloaders
from utils import load_cfg

def save(x, path, n_row, sz=64):
    x = F.interpolate(x, (sz, sz))
    save_image(x.data, path, nrow=n_row, normalize=True, padding=2)

def generate_cf_dataset(cgn, path, dataset_size, no_cfs, device):
    
    cgn.batch_size = 100
    n_classes = 10
    
    fol1 = './Imgs_CF_new'
    if not os.path.exists(fol1):
        os.mkdir(fol1)
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
    
    
    df = pd.DataFrame(columns=['im_name', 'shape_cls', 'texture_cls', 'bg_cls'])
    csv_path = join(fol1, 'labels.csv')
    df.to_csv(csv_path)

    df_noise = pd.DataFrame(columns=['im_name', 'noise'])
    noise_path = join(fol1, 'noise.csv')
    with torch.no_grad():
        for class_num in tqdm(list_of_classes, desc='Generating CFs'):
        
            # generate initial mask
            # y_gen = torch.randint(n_classes, (cgn.batch_size,)).to(device)
            # mask, _, _ = cgn(y_gen)
            print(f"Generating mask for class {class_num}")
            y_gen = [class_num]
            print(y_gen)
            y_gen= torch.Tensor(y_gen).int().to(device)
            print(y_gen)
            #noise_output, mask, foreground, background = cgn(y_gen)
            #write to csv file all 3 images have same noise output
            # image name 
            #df_noise = pd.DataFrame(columns=[im_name + f'{i}_{img_num}'] + list(noise_output))
            #df_noise.to_csv(noise_path, mode='a')
            im_name = str(class_num)
            ys_pd = [class_num, class_num, class_num]
            
            # generate counterfactuals, i.e., same masks, foreground/background vary
            for i in tqdm(range(no_cfs,), desc='Inner Loop'):
                noise_data, mask, foreground, background = cgn(ys=y_gen, counterfactual=True)
                x_gen = mask * foreground + (1 - mask) * background
                #save images 
                im_name = str(class_num) 
                im_name_num = im_name + '_' + str(i) 
                #print(noise_data.shape, '---------noise shape')
                array_has_nan = np.isnan(noise_data)
                #print(array_has_nan, '---------nan')
                #print('Range of noise data', np.min(noise_data), np.max(noise_data))
                if array_has_nan.any()==True:
                    print(array_has_nan, '---------nan values Found')
                array_has_nan[np.isnan(array_has_nan)] = 0
                
                
                x_gen_filename = join(x_gen_fol, im_name_num +  '_x_gen.jpg')
                mask_filename = join(mask_fol,   im_name_num + '_mask.jpg')
                foreground_filename = join(foreground_fol,  im_name_num + '_foreground.jpg')
                background_filename = join(background_fol,   im_name_num + '_background.jpg')
                

                save(x_gen.data, x_gen_filename, n_row=1)
                
                save(mask.data, mask_filename, n_row=1)
                save(foreground.data, foreground_filename, n_row=1)
                save(background.data, background_filename, n_row=1)
                df_noise = pd.DataFrame(columns=[im_name_num] + list(noise_data))
                df_noise.to_csv(noise_path, mode='a')
                
                

                #x.append(x_gen.detach().cpu())
                #y.append(y_gen.detach().cpu())
            #3 set of images and labels
            df = pd.DataFrame(columns=[im_name_num] + ys_pd)
            df.to_csv(csv_path, mode='a')
            


    #dataset = [torch.cat(x), torch.cat(y)]
    #print(f"x shape {dataset[0].shape}, y shape {dataset[1].shape}")
    print('Generation done')
    #torch.save(dataset, 'mnists/data/' + path)


def generate_dataset(dl, path):
    x, y = [], []
    for data in dl:
        x.append(data['ims'].cpu())
        y.append(data['labels'].cpu())

    dataset = [torch.cat(x), torch.cat(y)]

    print(f"Saving to {path}")
    print(f"x shape: {dataset[0].shape}, y shape: {dataset[1].shape}")
    torch.save(dataset, 'mnists/data/' + path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        choices=['cgn_double_colored_MNIST', 'cgn_colored_MNIST', 'cgn_wildlife_MNIST'], default= 'cgn_double_colored_MNIST',
                        help='Name of the dataset. Make sure the name and the weight_path match')
    parser.add_argument('--weight_path', default='',
                        help='Provide path to .pth of the model')
    parser.add_argument('--dataset_size', type=float, default=5e4,
                        help='Size of the dataset. For counterfactual data: the more the better.')
    parser.add_argument('--no_cfs', type=int, default=1000,
                        help='How many counterfactuals to sample per datapoint')
    parser.add_argument('--ablation', type=bool, default=False, metavar='A',
                        help="Whether to ablate how many cf images used")
    args = parser.parse_args()
    print(args)

    assert args.weight_path or args.dataset, "Supply dataset name or weight path."
    if args.weight_path: assert args.dataset, "Also supply the dataset type."

    # Generate the dataset
    if 1==2:
        # get dataloader
        dl_train, dl_test = get_dataloaders(args.dataset, batch_size=1000, workers=1)

        # generate
        generate_dataset(dl=dl_train, path=args.dataset + '_train.pth')
        generate_dataset(dl=dl_test, path=args.dataset + '_test.pth')

    # Generate counterfactual dataset
    else:
        # load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cgn = CGN()
        #cgn.load_state_dict(torch.load(args.weight_path, 'cpu'))
        model_path = os.path.join('experiments', args.dataset, 'weights', 'ckp.pth')
        print('Loading model from: ' + model_path)
        cgn.load_state_dict(torch.load(model_path))
        print('loaded model')
        

        cgn.to(device).eval()
        if args.ablation:
            for i in [1, 5, 10] if args.dataset == 'colored_MNIST' else [1, 5, 10, 20]:
                print(f"Generating the counterfactual {args.dataset} of size {args.dataset_size}")
                generate_cf_dataset(cgn=cgn, path=args.dataset + f'_counterfactual_{i}.pth',
                                    dataset_size=args.dataset_size, no_cfs=i,
                                    device=device)
        else:
            generate_cf_dataset(cgn=cgn, path=args.dataset + f'_counterfactual.pth',
                                dataset_size=args.dataset_size, no_cfs=args.no_cfs,
                                device=device)
