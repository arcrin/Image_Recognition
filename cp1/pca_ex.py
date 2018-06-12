from PIL import Image
from numpy import *
from pylab import *
from imtools import pca, get_imlist


imlist = get_imlist('../data/a_thumbs')
print(imlist)

im = array(Image.open(imlist[0]))
m, n = im.shape[0:2]
imnbr = len(imlist)

immatrix = array([array(Image.open(im)).flatten() for im in imlist], 'f')

V, S, immean = pca(immatrix)
figure()
gray()
subplot(2, 4, 1)
imshow(immean.reshape(m, n))
for i in range(7):
    subplot(2, 4, i+2)
    imshow(V[i].reshape(m, n))

show()
