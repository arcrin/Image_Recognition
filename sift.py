from PIL import Image
from numpy import loadtxt, savetxt, hstack, arange, pi, cos, sin
from pylab import imshow, plot, axis
import os


def process_image(image_name, result_name, params="--edge-thresh 10 --peak-thresh 5"):
    """
    Process an image and save the results in a file
    :param image_name:
    :param result_name:
    :param params:
    :return:
    """

    if image_name[-3:] != 'pgm':
        # create a pgm file
        im = Image.open(image_name).convert('L')
        im.save('tmp.pgm')
        image_name = 'tmp.pgm'

    cmmd = str("sift " + image_name + " --output=" + result_name +
               " " + params)
    print(cmmd)
    os.system(cmmd)
    print('processed', image_name, 'to', result_name)


def read_features_from_file(filename):
    """
    Read feature properties and return in matrix form
    :param filename:
    :return:
    """
    f = loadtxt(filename)
    return f[:, :4], f[:, :4]


def write_features_to_file(filename, locs, desc):
    """
    Save feature location and descriptor to file
    :param filename:
    :param locs:
    :param desc:
    :return:
    """

    savetxt(filename, hstack((locs, desc)))



def plot_features(im,locs,circle=False):
  """ Show image with features. input: im (image as array),
    locs (row, col, scale, orientation of each feature). """

  def draw_circle(c,r):
    t = arange(0,1.01,.01)*2*pi
    x = r*cos(t) + c[0]
    y = r*sin(t) + c[1]
    plot(x,y,'b',linewidth=2)

  imshow(im)
  if circle:
    for p in locs:
      draw_circle(p[:2],p[2])
  else:
    plot(locs[:,0],locs[:,1],'ob')
  axis('off')