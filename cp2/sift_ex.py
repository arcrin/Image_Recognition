from sift import *
from numpy import *
from PIL import *
from pylab import *


imname = '../data/cpu_01.jpg'
im1 = array(Image.open(imname).convert('L'))
process_image(imname, 'cpu_01.sift')
l1, d1 = read_features_from_file('cpu_01.sift')

figure()
gray()
plot_features(im1, l1[:1000], circle=True)
show()