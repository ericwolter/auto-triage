import numpy as np
import glob
import os
import tqdm
from PIL import Image

def load_image(path):
    im = Image.open(path)
    im = im.resize((224,224), resample=Image.LANCZOS)


    #img_arr = np.ndarray((224,224,3),np.float32)
    #constants=(123.68,116.779,103.939)
    #for i in range(image.shape[2]):
    #    x = image[:,:,i]
    #    x_p = np.pad(x, ((0, 224 - width), (0, 224 - height)), mode = "constant", constant_values = constants[i])
    #    img_arr[:,:,i] = x_p

    return im

for filepath in tqdm.tqdm(glob.glob('/media/home/Data/code/auto-triage/data/images/*.JPG')):
    basename = os.path.basename(filepath)
    postpath = os.path.join('/media/home/Data/code/auto-triage/data/images_post_distort',basename)
    #print(basename, postpath)
    image = load_image(filepath)
    image.save(postpath, "JPEG")
