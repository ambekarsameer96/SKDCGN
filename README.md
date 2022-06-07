# Counterfactual Generative Networks #

[![CGN](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1O0wScyr_Xzcvq-Ypm1rMal6A2Zoh8EYP?authuser=1#scrollTo=-wk70pN-sJtU) <br>
```

## Setup ##
Install anaconda (if you don't have it yet)
```Shell
wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
bash Anaconda3-2020.11-Linux-x86_64.sh
source ~/.profile
```

Clone the repo and build the environment

```Shell
git clone https://github.com/ambekarsameer96/FACT_AI.git
cd code/counterfactual_generative_networks-main
conda env create -f environment.yml
conda activate cgn
```

Make all scripts executable: ```chmod +x  scripts/*```. Then, download the datasets (colored MNIST, Cue-Conflict, IN-9) and the pre-trained weights (CGN, U2-Net). Comment out the ones you don't need.

```Shell
./scripts/download_data.sh
./scripts/download_weights.sh
```

## MNIST Experiments ##
<img src="/code/counterfactual_generative_networks-main/media/0_46000_x_gen_double_colored.png" width="15%">

The main functions of this sub-repo are:
- Generating the MNIST variants
- Training a CGN
- Generating counterfactual datasets
- Training a shape classifier

### Train the CGN ###

#MNIST dataset
"""

Ablation study code running
```Shell
python mnists/train_classifier.py --dataset double_colored_MNIST_counterfactual --ablation True
```

"""To run with SSIM loss (Use SSIM Flag to activate it and mention the corresponding the dataset name and other fields.)

```Shell
usage: python mnists/train_cgn.py --cfg Dataset_cfg_file ---ssim_flag true
```


"""
```Shell


python mnists/train_cgn.py --cfg mnists/experiments/cgn_wildlife_MNIST/cfg.yaml ---ssim_flag true

python mnists/generate_data.py \
--dataset wildlife_MNIST --no_cfs 10 --dataset_size 100000

python mnists/train_classifier.py --dataset wildlife_MNIST_counterfactual
```



"""To run with color jitter augmentation. Seperate files with the extension/suffix augment hve been created which perform the same task as the original files but the only difference is these files perform 'Color Jitter' augmentation. 

```
usage: python mnists/train_cgn.py --cfg Dataset_cfg_file ---ssim_flag true
```



"""
```Shell

python mnists/train_cgn_augment.py --cfg mnists/experiments/cgn_wildlife_MNIST/cfg.yaml

python mnists/generate_data_augment.py \
--dataset wildlife_MNIST --no_cfs 10 --dataset_size 100000

python mnists/train_classifier_augment.py --dataset wildlife_MNIST_counterfactual
__Distributed Training__. To switch to multi-GPU training, run ```echo $CUDA_VISIBLE_DEVICES``` to see if the GPUs are visible. In the case of a
single node with several GPUs, you can run, e.g.,
```

### Imagenet dataset
<img src="/code/counterfactual_generative_networks-main/media/good_image_1.png" width="100%">


## Test Inception score
"""
```Shell
python imagenet/generate_data.py --n_data 32 --weights_path imagenet/weights/cgn.pth --mode random --run_name val --truncation 0.5 --batch_sz 1
```
"""
## Training the CGN

NOTE: Training the CGN for Imagenet utilises biggan-256 and U2-net weights. It runs for 1.2m iterations(approx 0.5/s). Prefer to skip this part if adequate resource not available.
"""
```Shell

python imagenet/train_cgn.py --model_name MODEL_NAME
```

"""## Generate Counterfactual Images

2 Folders of counterfactual images are needed(Train, Val). Train has 35,000 counterfactuals, Val has 5000 counterfactual images. We split it to provided 1:1 ratio that is recommended in the paper(Imagenet-1k is replaced with Imagenet-1k(mini))
"""
```Shell

python imagenet/generate_data.py --n_data 35000 --weights_path imagenet/weights/cgn.pth --mode random --run_name train --truncation 0.5 --batch_sz 1

python imagenet/generate_data.py --n_data 5000 --weights_path imagenet/weights/cgn.pth --mode random --run_name val --truncation 0.5 --batch_sz 1
```
"""Move the val, train into a cf_data folder"""
```Shell

%cp -r /content/counterfactual_generative_networks/imagenet/data/train /content/counterfactual_generative_networks/imagenet/data/cf_data
```
```Shell

Commented out IPython magic to ensure Python compatibility.
%cp -r /content/counterfactual_generative_networks/imagenet/data/val /content/counterfactual_generative_networks/imagenet/data/cf_data
```
"""## Training the classifier

We replaced Imagenet-1k with Imagenet-1k(mini).
"""
```Shell

python imagenet/train_classifier.py -a resnet50 -b 32 --lr 0.001 -j 6 \
--epochs 45 --pretrained --cf_data imagenet/cf_data --name RUN_NAME
```

"""
## Plotting Explainability heatmaps
<img src="/code/counterfactual_generative_networks-main/media/ipod_lime_plot_1.png" width="100%">


Replace the --image_loc path with the image of your choice.

```Shell

python lime_plot.py --image_loc '/content/counterfactual_generative_networks/imagenet/data/mini-imagenet/train/fg_n02002556_34176_bg_n03272562_13865.JPEG'
```

__Visualization__. To visualize the Tensorboard outputs, run ```tensorboard --logdir=imagenet/runs``` and open the local address in your browser.

## Acknowledgments ##
We like to acknowledge several repos of which we use parts of code, data, or models in our implementation:

- colored MNIST by [feidfoe](https://github.com/feidfoe/learning-not-to-learn)
- pre-trained BigGAN by [huggingface](https://github.com/huggingface/pytorch-pretrained-BigGAN)
- U2-Net by [NathanUA](https://github.com/NathanUA/U-2-Net/)
- Imagenet training by and with [pytorch](https://github.com/pytorch/examples/tree/master/imagenet)
- Style-vs-Shape evaluation by [rgeirhos](https://github.com/rgeirhos/texture-vs-shape)
- BG-Gap evaluation by [MadryLab](https://github.com/MadryLab/backgrounds_challenge)
