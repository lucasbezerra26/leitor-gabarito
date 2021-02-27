from logging import warn

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage import img_as_float
from skimage import util
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.io import imread
from skimage.morphology import remove_small_objects

LYRICS = {
    1: 'a',
    2: 'b',
    3: 'c',
    4: 'd',
    5: 'e',
}


class CustomError(Exception):
    def __init__(self, m):
        self.message = m

    def __str__(self):
        return self.message


def plot_edge(img):
    plt.figure(figsize=(10, 20))
    plt.imshow(img, cmap="gray")


def plot_function(images, title=None):
    fig, ax = plt.subplots(1, len(images), figsize=(30, 30))
    for i, edge in enumerate(images):
        ax[i].imshow(edge, cmap="gray")
        if title:
            ax[i].set_title(title[i])
        else:
            ax[i].set_title(str(i))


def remove_small_objects_my(ar, min_size=64, connectivity=1, in_place=False):
    if in_place:
        out = ar
    else:
        out = ar.copy()
    if min_size == 0:
        return out
    if out.dtype == bool:
        selem = ndi.generate_binary_structure(ar.ndim, connectivity)
        ccs = np.zeros_like(ar, dtype=np.int32)
        ndi.label(ar, selem, output=ccs)
    else:
        ccs = out
    try:
        component_sizes = np.bincount(ccs.ravel())
    except ValueError:
        raise ValueError("Negative value labels are not supported. Try "
                         "relabeling the input with `scipy.ndimage.label` or "
                         "`skimage.morphology.label`.")

    if len(component_sizes) == 2 and out.dtype != bool:
        warn("Only one label was provided to `remove_small_objects`. "
             "Did you mean to use a boolean array?")

    mat = component_sizes
    mat[0] = 1
    min_size = max(mat) - 1

    too_small = component_sizes < min_size
    too_small_mask = too_small[ccs]
    out[too_small_mask] = 0

    return out


def remove_small_objects_my2(ar):
    selem = ndi.generate_binary_structure(ar.ndim, 1)
    ccs = np.zeros_like(ar, dtype=np.int32)
    ndi.label(ar, selem, output=ccs)

    component_sizes = np.bincount(ccs.ravel())

    out = ar.copy()
    index = sorted(component_sizes, reverse=True)
    size = len(index)
    index = index[size // 2]
    too_small = component_sizes < index
    too_small_mask = too_small[ccs]
    out[too_small_mask] = 0
    return out


def return_rectangle(ar):
    selem = ndi.generate_binary_structure(ar.ndim, 1)
    ccs = np.zeros_like(ar, dtype=np.int32)
    ndi.label(ar, selem, output=ccs)
    component_sizes = np.bincount(ccs.ravel())

    too_small = component_sizes < 0
    out_final = list()
    for i in range(len(too_small) - 1):
        index = i + 1
        small = too_small.copy()
        small[:index] = True
        small[(index + 1):] = True
        small[0] = False
        out = ar.copy()
        too_small_mask = small[ccs]
        out[too_small_mask] = 0
        out_final.append(out)
    if len(out_final) == 0:
        raise CustomError("No rectangle found")
    return out_final


def return_lyrics(ar, max_label_int=5):
    selem = ndi.generate_binary_structure(ar.ndim, 1)
    ccs = np.zeros_like(ar, dtype=np.int32)
    ndi.label(ar, selem, output=ccs)
    if max_label_int == 0:
        return ndi.find_objects(ccs)
    return ndi.find_objects(ccs, max_label_int)


def percentage(img_lyrics, verbose=False, percent=0.40):
    num_dark_px = len(img_lyrics[img_lyrics < 0.3])
    num_total_px = img_lyrics.shape[0] * img_lyrics.shape[1]
    dark_area_pcnt = num_dark_px / num_total_px

    if verbose:
        print(dark_area_pcnt)

    return dark_area_pcnt > percent


def marked(rectangles_slices, img, img_original, img_otsu):
    marked_dict = dict()

    for index, rectangle in enumerate(rectangles_slices):
        alternatives = img_original.copy()
        # TODO rectangle is a list
        alternatives[rectangle == False] = False

        alternatives = ndi.binary_fill_holes(alternatives)
        lyrics_slices = return_lyrics(alternatives, 0)

        if None in lyrics_slices:
            raise CustomError("Did not identify all letters")

        lyrics_slices = sorted(lyrics_slices, key=lambda x: area(list(x)), reverse=True)
        lyrics_slices = lyrics_slices[:5]
        lyrics_slices = sorted(lyrics_slices, key=lambda x: x[1].start)

        if len(lyrics_slices) != 5:
            raise CustomError("Did not identify all letters")

        marked_dict[index + 1] = {}
        for index_intern, lyric in enumerate(lyrics_slices):
            lyric_final = img[lyric]
            aux = False
            if index == 24:
                plot_edge(lyric_final)
                print(marked_dict[index + 1], index + 1)
                print(percentage(lyric_final, verbose=True))
            marked_dict[index + 1][LYRICS[index_intern + 1]] = percentage(lyric_final, verbose=False)

    return marked_dict


def return_lyrics_of_dict(data):
    result = None
    for key, value in zip(data.keys(), data.values()):
        if value and result:
            return None
        elif value:
            result = key
    return result


def convert_in_lyrics(marked_alternatives):
    marked_copy = marked_alternatives.copy()
    for key, lyrics in zip(marked_copy.keys(), marked_copy.values()):
        marked_copy[key] = return_lyrics_of_dict(lyrics)
    return marked_copy


def area(slice_):
    return (slice_[0].start - slice_[0].stop) * (slice_[1].start - slice_[1].stop)


def remove_shadow(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)

    rgb_planes = cv2.split(img)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    result_norm = cv2.merge(result_norm_planes)
    return result_norm


def processing(img_original, img_removed_shadow=None, verbose=False, flag=False):
    img = img_removed_shadow
    thresh = threshold_otsu(img)
    img_otsu = img > thresh

    inverted_img = util.invert(img_otsu)

    biggest_object = remove_small_objects_my(inverted_img)
    biggest_object_fill = ndi.binary_fill_holes(biggest_object)

    new_segmentation = biggest_object_fill.copy()
    new_segmentation[biggest_object == True] = False
    marked_dict, marked_dict_resume = None, None
    try:
        rectangles = remove_small_objects_my2(remove_small_objects(new_segmentation, 500))
        rectangles_list = return_rectangle(rectangles)

        marked_dict = marked(rectangles_list, img_original, inverted_img.copy(), img_otsu)
        title = "Sem erro:"
        marked_dict_resume = convert_in_lyrics(marked_dict)

    except Exception as e:
        print(e)
        title = "Com erro:"

    if verbose:
        header = [img_original, img, img_otsu, inverted_img, biggest_object, biggest_object_fill, new_segmentation]
        title = [f'{title} original', 'Sem sombra', 'otsu', 'cores invertida', 'Maior objeto na img',
                 'Regiao do maior obj', 'Segmentacao']
        plot_function(header, title=title)

    return marked_dict, marked_dict_resume


def read_img(path, verbose=False, flag=False):
    img_original = rgb2gray(img_as_float(imread(path)))
    removed = rgb2gray(img_as_float(remove_shadow(path)))
    return processing(img_original=img_original, img_removed_shadow=removed, verbose=verbose, flag=flag)
