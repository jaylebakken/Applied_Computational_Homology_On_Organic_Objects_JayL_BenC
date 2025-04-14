
from PIL import Image
import numpy as np
import gudhi.cover_complex as gud
import os

for i in range(150):
    img = Image.open("Dataset/POLLEN73S/acrocomia_aculeta/Figura1.TIF")
    img = img.convert('L')
    imat = np.asarray(img)
    imat = np.where(imat<(50+i), imat,255)
    img = Image.fromarray(imat,mode="L")
    #img.save("sampleImages/sampleimg_{}.png".format(int(i)))
print(imat)

#mp = gud.MapperComplex()
#mp.fit(X=imat)

