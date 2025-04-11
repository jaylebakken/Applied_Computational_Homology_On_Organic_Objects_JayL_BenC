import math
from PIL import Image
import numpy as np


img = Image.open("Dataset/POLLEN73S/acrocomia_aculeta/Figura1.TIF")
img = img.convert('L')
imat = np.asarray(img)
imat = np.where(imat<120, imat,255)
print(imat)
img = Image.fromarray(imat,mode="L")
img.save("ooohahhh.png")
