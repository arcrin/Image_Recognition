from PIL import Image
from pylab import *


# read image to array
im = array(Image.open('../data/empire.jpg'))
imshow(im)
figure()
hist(im.flatten(), 128)
show()
