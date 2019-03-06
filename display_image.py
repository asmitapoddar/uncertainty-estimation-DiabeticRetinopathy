import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')   # Force matplotlib to not use any Xwindows backend.
from pathlib import Path
import matplotlib.pyplot as plt
import os

def imshow_grid(data, height=None, width=None, normalize=False, padsize=1, padval=0):
    '''
    Take an array of shape (N, H, W) or (N, H, W, C)
    and visualize each (H, W) image in a grid style (height x width).
    '''
    if normalize:
        data -= data.min()
        data /= data.max()

    N = data.shape[0]
    if height is None:
        if width is None:
            # height = int(np.ceil(np.sqrt(N)))
            height = 2 * N
        else:
            height = 2 * N
            height = int(np.ceil(N / float(width)))

    if width is None:
        width = 2 * N
        width = int(np.ceil(N / float(height)))

    assert height * width >= N

    # append padding
    padding = ((0, (width * height) - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((height, width) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((height * data.shape[1], width * data.shape[3]) + data.shape[4:])

    plt.imshow(data)
    plt.savefig(os.path.join(os.getcwd(), 'Result', 'output', 'image.png'))

    return data

def normalize(attrs, ptile=99):
    '''Normalize the provided attributions so that they fall between
    -1.0 and 1.0.
    '''
    h = np.percentile(attrs, ptile)
    l = np.percentile(attrs, 100 - ptile)
    return np.clip(attrs / max(abs(h), abs(l)), -1.0, 1.0)


def normalize_one_side(attrs, ptile=99):
    '''Normalize the provided attributions so that they fall between
    -1.0 and 1.0.
    '''
    h = np.percentile(attrs, ptile)
    return np.clip(attrs / h, 0.0, 1.0)


def gray_scale(img):
    '''Converts the provided RGB image to gray scale.
    '''
    img = np.average(img, axis=2)
    return np.transpose([img, img, img], axes=[1, 2, 0])


def pil_img(a):
    '''Returns a PIL image created from the provided RGB array.
    '''
    a = np.uint8(a)
    return Image.fromarray(a)

def show_img():
    R = np.array([255, 0, 0])
    G = np.array([0, 255, 0])
    B = np.array([0, 0, 255])

    stoch_output = "Result/pred"
    no_stoch_output = "Result/pred"
    img_dir = "Result/data"
    output_dir = "Result/output"

    def visualize_attrs(img, attrs, pos_ch=B, neg_ch=R):
        '''Visaualizes the provided attributions by first aggregating them along the color channel and then overlaying the positive attributions
         along pos_ch, and negative attributions along neg_ch.

         The provided image and attributions must of shape (224, 224, 3).
        '''
        '''
        pos_attrs = normalize_one_side(pos_attrs, ptile=99.9)
        if np.max(neg_attrs) != 0:
            neg_attrs = normalize_one_side(neg_attrs, ptile=99.9)
        attrs_mask = pos_attrs * pos_ch + neg_attrs * neg_ch
        vis = 0.3 * gray_scale(img) + 0.7 * attrs_mask
        '''
        pos_attrs = attrs * (attrs >= 0.0)
        neg_attrs = -1.0 * attrs * (attrs < 0.0)
        pos_attrs = normalize_one_side(pos_attrs, ptile=100)
        if np.max(neg_attrs) != 0:
            neg_attrs = normalize_one_side(neg_attrs, ptile=100)

        attrs_mask = pos_attrs * pos_ch + neg_attrs * neg_ch
        vis =  0.3 * gray_scale(img) + 0.7 * attrs_mask
        return np.uint8(vis)

    pics = []

    output_dir = Path(stoch_output.replace(stoch_output, output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    for dirpath, dirnames, filenames in os.walk(stoch_output):
            print(dirpath)
            for file in filenames:
                if file.endswith('.npy'):
                    filepath = os.path.join(dirpath, file)
                    attrs = np.load(filepath)
                    attrs = np.asarray([gray_scale(attr) for attr in attrs])
                    basename = os.path.basename(filepath).replace('.png.npy', '')

                    attrs_std = np.std(attrs, axis=0)  #     attr_mean = np.expand_dims(attrs.mean(axis=0),0)
                    attr_mean = attrs.mean(axis=0)
                    # substract std/add std depending on the direction of attributions
                    pos_attrs = attr_mean * (attr_mean >= 0.0)

                    neg_attrs = -1.0 * attr_mean * (attr_mean < 0.0)
                    attr_mean_sq = pos_attrs ** 2 - neg_attrs ** 2
                    attr_weighted = attr_mean * attrs_std
                    print(filepath.replace(stoch_output, no_stoch_output))
                    attr_fixed = np.load(filepath.replace(stoch_output, no_stoch_output))

                    #     attrs = np.concatenate([attrs, attr_mean], axis=0)
                    attr_size = attrs.shape[1:3]
                    #img_path = filepath.replace('.npy', '').replace(stoch_output, img_dir)
                    img_path = os.path.join(img_dir,'image.png')

                    img = Image.open(img_path).resize(attr_size)

                    #output_dir = Path(dirpath.replace(stoch_output, output_dir))
                    #output_dir.mkdir(parents=True, exist_ok=True)

                    # FOR GRID
                    vis = np.asarray([visualize_attrs(img, attr) for attr in [attr_mean, attrs_std, attr_weighted,
                                                                              attr_mean_sq, attr_fixed[0]]] + [np.asarray(img)])
                    vis = np.asarray([visualize_attrs(img, attr) for attr in [attr_mean, attrs_std, ]] + [np.asarray(img)])
                    d = imshow_grid(vis)

                    # FOR INDIVIDUAL IMAGES
                    vis1 = np.asarray([visualize_attrs(img, attr_fixed[0]) + [np.asarray(img)]])

                    # out.save(os.path.join(output_dir.as_posix(), file))
                    print('Shape', vis1[0][0].shape)
                    plt.imshow(vis1[0][0])
                    plt.savefig(os.path.join(output_dir.as_posix(), basename))
                    print('Doneee')

                    #return (vis1[0][0])
                    #return d