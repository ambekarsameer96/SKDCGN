# SKDCGN- Source-free Knowledge Distillation of Counterfactual Generative Networks using cGANs

Note: To run the experiments, saved models are required. As the size of the models are quite large, we have not provided them over here. The links for the models to download can be provided on demand.

## Setup ##
Install anaconda (if you don't have it yet)
```Shell
wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
bash Anaconda3-2020.11-Linux-x86_64.sh
source ~/.profile
```

Clone the repo and build the environment

```Shell
git clone https://github.com/ambekarsameer96/DL2.git
conda env create -f env.yml
conda activate cgn
```

## SKDCGN ##


### Training the SKDCGN



## MNIST Experiments ###
For MNIST, first use pretrained weights from the authors repo that has been forked from the 'CGN' paper. 

```bash
python cgn_extract_mnist_samples/generate_data.py

```
The images for every IM, noise, labels will be saved in a directory Imgs_CF, which can be converted into a numpy array using converter.ipynb 

Run following commands to run for every IM, it will use existing losses + KL_Divergence as loss function integrated in models.py
```bash
python SKDCGN_MNIST/main_bg_exp.py

python SKDCGN_MNIST/main_fg_exp.py

python SKDCGN_MNIST/main_mask_exp.py

```

## Baseline ###

### Training TinyGAN for 1000 classes
```bash
$ bash Baseline_code/TinyGAN_updated_1000_classes/train.sh
```

### Training the Baseline

NOTE: Training the Baseline for ImageNet utilises TinyGAN and U2-Net weights. It runs for 1.2m iterations(approx 0.5/s). Prefer to skip this part if adequate resource not available.

```bash

python Baseline_code/Baseline/code/counterfactual_generative_networks-main/imagenet/train_cgn.py --model_name MODEL_NAME
```

### Generate Counterfactual Images

2 Folders of counterfactual images are required (Val, Test). Val has 5,000 counterfactuals, Test has 2000 counterfactual images. 

```bash

python Baseline_code/Baseline/code/counterfactual_generative_networks-main/imagenet/generate_data.py --n_data 5000 --weights_path imagenet/weights/cgn.pth --mode random --run_name val --truncation 0.5 --batch_sz 1

python Baseline_code/Baseline/code/counterfactual_generative_networks-main/imagenet/generate_data.py --n_data 2000 --weights_path imagenet/weights/cgn.pth --mode random --run_name test --truncation 0.5 --batch_sz 1
```

## Shape IM experiments ##
### Run following command with desired modifications in the training loop
```bash
python cgn_extract_mnist_samples/train_cgn_Shape_IM.py
```

### Generate datasets and train classifier. Final classifier results will be in resulting SLURM file.
```bash
python cgn_extract_mnist_samples/generate_data.py --dataset 'double_colored_MNIST'

python cgn_extract_mnist_samples/generate_data.py --file_name 'train_noise' --dataset 'double_colored_MNIST' --weight_path 'mnists/experiments/cgn_double_colored_MNIST_2022_06_02_14_31_20_tmp_noise01/weights/ckp_46000.pth'

python cgn_extract_mnist_samples/train_classifier.py --dataset "double_colored_MNIST_counterfactual"
```

### Repeat process with remaining modifications (don't forget to retrain network with the respective modifications!)
```bash
python cgn_extract_mnist_samples/generate_data.py --file_name 'train_transparent' --dataset 'double_colored_MNIST' --weight_path 'mnists/experiments/cgn_double_colored_MNIST_2022_06_01_10_24_42_tmp_transparent75/weights/ckp_46000.pth'

python cgn_extract_mnist_samples/generate_data.py --file_name 'train_rotation' --dataset 'double_colored_MNIST' --weight_path 'mnists/experiments/cgn_double_colored_MNIST_2022_06_03_09_56_27_tmp_rotation180deg/weights/ckp_46000.pth'
```

## Acknowledgments ##
We like to acknowledge several repos of which we use parts of code, data, or models in our implementation:

- colored MNIST by [feidfoe](https://github.com/feidfoe/learning-not-to-learn)
- pre-trained BigGAN by [huggingface](https://github.com/huggingface/pytorch-pretrained-BigGAN)
- U2-Net by [NathanUA](https://github.com/NathanUA/U-2-Net/)
- Imagenet training by and with [pytorch](https://github.com/pytorch/examples/tree/master/imagenet)
- Style-vs-Shape evaluation by [rgeirhos](https://github.com/rgeirhos/texture-vs-shape)
- BG-Gap evaluation by [MadryLab](https://github.com/MadryLab/backgrounds_challenge)
