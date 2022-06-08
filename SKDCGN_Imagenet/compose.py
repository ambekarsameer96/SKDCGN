import torch
import numpy
from model import Generator
from utils import *
import utils
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from scipy.stats import truncnorm
import random

shape = Generator(image_size=256, conv_dim=32, z_dim=128, c_dim=128, repeat_num=5)
device = 'cpu'
shape.load_state_dict(torch.load('gan/models/shape/generator_15.pt',map_location=device))
shape.eval()

with torch.no_grad():
    for i in range(5):
        label = torch.tensor([10])
        noise = torch.FloatTensor(utils.truncated_normal(128))
        img = shape(noise, label).detach().cpu()
        save_image(utils.denorm(img.cpu()), f'out{i}.png')