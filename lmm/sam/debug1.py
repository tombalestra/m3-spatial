from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

sam = sam_model_registry["vit_h"](checkpoint="/disk1/checkpoint/vlcore/sam/sam_vit_h_4b8939.pth").cuda()
mask_generator = SamAutomaticMaskGenerator(sam)

image = Image.open('/disk1/data/m3/data_v2/tabletop_v2/images/frame_000000.jpg')
image = np.array(image.convert("RGB"))
masks = mask_generator.generate(image)

plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.savefig("image.png")