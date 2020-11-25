#!/usr/bin/env python
# coding: utf-8

# In[1]:


import imageio
import requests
import bz2
from PIL import Image
import torch
import torchvision.transforms as transforms
import dlib
from pix2pixHD.data.base_dataset import __scale_width
from pix2pixHD.models.networks import define_G
import pix2pixHD.util.util as util
from aligner import align_face

import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')


# ## Load image

# In[ ]:


img_url = 'http://www.reviewstar.net/newsphoto/reviewstar/news/2015/09/11/2015091116374234-1-rstarnyj.jpg'
img_filename = 'image.jpg'
imageio.imwrite(img_filename, imageio.imread(img_url))


# ## Face aligment

# In[ ]:


def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path

def download(url, file_name):
    with open(file_name, "wb") as file:
        response = requests.get(url)
        file.write(response.content)


# In[ ]:


shape_model_url = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
shape_model_path = 'landmarks.dat'
download(shape_model_url, shape_model_path)
shape_predictor = dlib.shape_predictor(unpack_bz2(shape_model_path))


# In[ ]:


aligned_img = align_face(img_filename, shape_predictor)[0]


# ## Prepare pix2pixHD model

# In[ ]:


def get_eval_transform(loadSize=512):
    transform_list = []
    transform_list.append(transforms.Lambda(lambda img: __scale_width(img,
                                                                      loadSize,
                                                                      Image.BICUBIC)))
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

transform = get_eval_transform()


# In[ ]:


config_G = {
    'input_nc': 3,
    'output_nc': 3,
    'ngf': 64,
    'netG': 'global',
    'n_downsample_global': 4,
    'n_blocks_global': 9,
    'n_local_enhancers': 1,
    'norm': 'instance',
}

# TBAL
weights_path = './pix2pixHD/checkpoints/r512_aging_pos/latest_net_G.pth'

model = define_G(**config_G)
pretrained_dict = torch.load(weights_path)
model.load_state_dict(pretrained_dict)
model.cuda();


# ## Use model

# In[ ]:


img = transform(aligned_img).unsqueeze(0)

with torch.no_grad():
    out = model(img.cuda())

out = util.tensor2im(out.data[0])


# ## Result

# In[ ]:


plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(aligned_img)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(out)
plt.axis('off')

plt.tight_layout();


# In[ ]:


imageio.imsave('result.jpg', out)


# In[ ]:




