from PIL import Image
from pylab import *


pil_im = Image.open('../data/empire.jpg').convert('L')
im = array(pil_im)

pil_im2 = Image.fromarray(im)
imshow(pil_im2)

show()