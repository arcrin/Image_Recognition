from PIL import Image
from numpy import *
from pylab import *


im = array(Image.open('../data/empire.jpg').convert('l'))

im2 = 255 - im

im3 = (100.0/255) * im + 100

im4 = 255.0 * (im/255.0) ** 2
#
# imshow(im)
# imshow(im2)
# imshow(im3)
# imshow(im4)
print(int(im.min()), int(im.max()))
print(int(im2.min()), int(im2.max()))
print(int(im3.min()), int(im3.max()))
print(int(im4.min()), int(im4.max()))

show()
