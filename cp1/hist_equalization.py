from imtools import histeq
from PIL import Image
from numpy import *
import os
import sys

sys.path.append('C:\\Project\\Python\\Image Recognition\\cp1')

im = array(Image.open('../data/AquaTermi_lowcontrast.jpg').convert('L'))
im2, cdf = histeq(im)
