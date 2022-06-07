import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import numpy as np
import os, json
from lime import lime_image
import torch
from torchvision import models, transforms
from torch.autograd import Variable
import torch.nn.functional as F
import argparse
from imagenet.models import InvariantEnsemble
from skimage.segmentation import mark_boundaries
from collections import OrderedDict


def get_image(path):
    with open(os.path.abspath(path), 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


# resize and take the center part of image to what our model expects
def get_input_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    return transf


def get_input_tensors(img):
    transf = get_input_transform()
    # unsqeeze converts single image to batch of 1
    return transf(img).unsqueeze(0)


# model = models.inception_v3(pretrained=True)
def get_pil_transform():
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224)
    ])

    return transf


def get_preprocess_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    transf = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    return transf

pill_transf = get_pil_transform()
preprocess_transform = get_preprocess_transform()


def batch_predict(images):
    model.eval()
    batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch = batch.to(device)

    logits = model(batch)
    if isinstance(logits, dict):
      probs = F.softmax(logits['avg_preds'], dim=1)
    else:
      probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_loc', default='',
                        help='image that has to be checked for classifier')
    parser.add_argument('--model_loc', default='',
                        help='provide path to continue training')
    args = parser.parse_args()
    # Choose the type of model
    model = InvariantEnsemble('resnet50', True)
    if args.model_loc:
      print("Inside model loc")
      checkpoint = torch.load(args.model_loc)
      # create new OrderedDict that does not contain `module.`
      state_dict = checkpoint['state_dict']
      new_state_dict = OrderedDict()
      for k, v in state_dict.items():
          name = k[7:] # remove `module.`
          new_state_dict[name] = v
      # load params
      model.load_state_dict(new_state_dict)
    else:
      print("Not inside model loc")
      model = models.inception_v3(pretrained=True)
    img = get_image(args.image_loc)
    idx2label, cls2label, cls2idx = [], {}, {}
    with open('imagenet/data/imagenet_class_index.json', 'r') as read_file:
        class_idx = json.load(read_file)
        idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
        cls2label = {class_idx[str(k)][0]: class_idx[str(k)][1] for k in range(len(class_idx))}
        cls2idx = {class_idx[str(k)][0]: k for k in range(len(class_idx))}

    img_t = get_input_tensors(img)
    model.eval()
    if args.model_loc:
      logits = model(img_t)['avg_preds']
    else:
      logits = model(img_t)
    probs = F.softmax(logits, dim=1)
    probs5 = probs.topk(5)
    print(tuple((p,c, idx2label[c]) for p, c in zip(probs5[0][0].detach().numpy(), probs5[1][0].detach().numpy())))
    test_pred = batch_predict([pill_transf(img)])
    test_pred.squeeze().argmax()
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(np.array(pill_transf(img)),
                                             batch_predict, # classification function
                                             top_labels=5,
                                             hide_color=0,
                                             num_samples=1000) # number of images that will be sent to classification function
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
    img_boundry2 = mark_boundaries(temp/255.0, mask)
    plt.imshow(img_boundry2)
    plt.savefig("img_boundry2.png")
