# SKDCGN- Source-free Knowledge Distillation of Counterfactual Generative Networks using cGANs#



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
cd code/counterfactual_generative_networks-main
conda env create -f environment.yml
conda activate cgn
```

### Baseline ###

## Training TinyGAN for 1000 classes
```bash
$ bash Baseline_code/TinyGAN_updated_1000_classes/train.sh
```

## Training the Baseline

NOTE: Training the Baseline for ImageNet utilises TinyGAN and U2-Net weights. It runs for 1.2m iterations(approx 0.5/s). Prefer to skip this part if adequate resource not available.
"""
```bash

python Baseline_code/Baseline/code/counterfactual_generative_networks-main/imagenet/train_cgn.py --model_name MODEL_NAME
```

"""## Generate Counterfactual Images

2 Folders of counterfactual images are needed (Val, Test). Val has 5,000 counterfactuals, Test has 2000 counterfactual images. 
"""
```bash

python Baseline_code/Baseline/code/counterfactual_generative_networks-main/imagenet/generate_data.py --n_data 5000 --weights_path imagenet/weights/cgn.pth --mode random --run_name val --truncation 0.5 --batch_sz 1

python Baseline_code/Baseline/code/counterfactual_generative_networks-main/imagenet/generate_data.py --n_data 2000 --weights_path imagenet/weights/cgn.pth --mode random --run_name test --truncation 0.5 --batch_sz 1
```


## Acknowledgments ##
We like to acknowledge several repos of which we use parts of code, data, or models in our implementation:

- colored MNIST by [feidfoe](https://github.com/feidfoe/learning-not-to-learn)
- pre-trained BigGAN by [huggingface](https://github.com/huggingface/pytorch-pretrained-BigGAN)
- U2-Net by [NathanUA](https://github.com/NathanUA/U-2-Net/)
- Imagenet training by and with [pytorch](https://github.com/pytorch/examples/tree/master/imagenet)
- Style-vs-Shape evaluation by [rgeirhos](https://github.com/rgeirhos/texture-vs-shape)
- BG-Gap evaluation by [MadryLab](https://github.com/MadryLab/backgrounds_challenge)
