from PIL import Image
from pylab import *
import os


def get_imlist(path):
    """
    Returns a list of file names for all jpg images in a directory.
    :param path:
    :return:
    """
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]


def imresize(im, sz):
    """
    Resize an image array using PIL
    :param im:
    :param sz:
    :return:
    """
    pil_im = Image.fromarray(uint8(im))
    return array(pil_im.resize(sz))


def histeq(im, nbr_bins=256):
    """
    Histogram equalization of a grayscale image
    :param im:
    :param nbr_bins:
    :return:
    """

    # get image histogram
    imhist, bins = histogram(im.flatten(), nbr_bins, normed=True)
    cdf = imhist.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1]

    # use linear interpolation of cdf to find new pixel values
    im2 = interp(im.flatten(), bins[:-1], cdf)

    return im2.reshape(im.shape), cdf


def compute_acerage(imlist):
    """
    Compute the acerage of a list of images.
    :param imlist:
    :return:
    """
    # open first image and make into array of tupe float
    acerageim = array(Image.open(imlist[0]), 'f')

    for imname in imlist[1:]:
        try:
            acerageim += array(Image.open(imname))
        except:
            print(imname + '...skipped')
    acerageim /= len(imlist)

    # return average as uint8
    return array(acerageim, 'uint8')
