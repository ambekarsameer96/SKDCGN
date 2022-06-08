#!/usr/bin/env python
# coding: utf-8

# In[8]:


import os
from glob import glob
import numpy as np
import pandas as pd
import torch


# In[4]:


df = pd.read_csv('../../../../Downloads/noise.csv', header=None)
df.columns = ['blank', 'img name', 'noise']
df = df.drop(['blank'], axis=1)
df['img name'] = df['img name'].apply(lambda x: '_'.join(x.split('_')[-2:]))
df['noise'] = df['noise'].apply(lambda x: np.fromstring(x[1:-1], sep=' '))
np.save('all_noises.npy', df['noise'].values)


# In[63]:


df = pd.read_csv('labels.csv')
df.columns = ['blank', 'img name', 'label', '_1', '_2']
df = df.drop(['blank','_1', '_2', 'img name'], axis=1)
np.save('all_labels.npy', df['label'].values)

