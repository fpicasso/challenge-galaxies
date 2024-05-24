import numpy as np
import matplotlib.pyplot as plt


from astroNN.datasets import load_galaxy10
from astroNN.datasets.galaxy10 import galaxy10cls_lookup, galaxy10_confusion

from pathlib import Path

images, labels = load_galaxy10()
imagesBarredSpiral = np.array(images[labels == 5]) 

images_after_resize = []
for img in imagesBarredSpiral:
    imageBS = torch.from_numpy(img).permute(2, 0, 1)

    img64 = v2.Resize(size=64)(imageBS)
    
    images_after_resize.append(img64.permute(1,2,0))
    print(img64.permute(1,2,0).shape)

torch.save(image_list_final, "galaxy5")
