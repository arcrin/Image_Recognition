from numpy import array
from PIL import Image
from harris import *

wid = 5

im1 = array(Image.open('../data/cpu_01.jpg').convert('L'))
im2 = array(Image.open('../data/cpu_02.jpg').convert('L'))

harrisim = compute_harris_response(im1, 5)
filtered_coord1 = get_harris_points(harrisim, wid+1)
d1 = get_descriptors(im1, filtered_coord1, wid)

harrisim = compute_harris_response(im2, 5)
filtered_coord2 = get_harris_points(harrisim, wid+1)
d2 = get_descriptors(im2, filtered_coord2, wid)


print("start matching")
print(match(d1, d2))

# figure()
# gray()
# plot_matches(im1, im2, filtered_coord1, filtered_coord2, matches[100:])
# show()