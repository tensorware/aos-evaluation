import os
import json
import tempfile

import glob as gb
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image
from zipfile import ZipFile


def load_data(path):
    data = {}
    data_folder = tempfile.mkdtemp()

    # extract files
    for zip_path in sorted(gb.glob(path, recursive=True)):
        with ZipFile(zip_path, 'r') as z:
            zip_name = os.path.basename(zip_path).rsplit('.', 1)[0]
            folder_path = os.path.join(data_folder, zip_name)

            print(f'extract {folder_path}')
            z.extractall(folder_path)

    # load config
    configs = {}
    for config_path in sorted(gb.glob(os.path.join(data_folder, '**', '*.json'), recursive=True)):
        config_file = os.path.basename(config_path)
        config_name = os.path.basename(config_file).rsplit('.', 1)[0]
        config_dir = os.path.dirname(os.path.relpath(config_path, data_folder))

        keys = config_dir.split(os.path.sep) + [config_name]
        set_value(configs, keys, json.load(open(config_path)))

    # load images
    images = {}
    for image_path in sorted(gb.glob(os.path.join(data_folder, '**', '*.png'), recursive=True)):
        image_file = os.path.basename(image_path)
        image_name = os.path.basename(image_file).rsplit('.', 1)[0]
        image_dir = os.path.dirname(os.path.relpath(image_path, data_folder))

        keys = image_dir.split(os.path.sep) + [image_name]
        set_value(images, keys, np.array(Image.open(image_path)))

    # generate dataframe
    for simulation in images:
        df = pd.DataFrame()

        # image parameter
        parameters = dict()
        parameters.update(get_value(configs, [simulation, 'drone', 'drone']))
        parameters.update(get_value(configs, [simulation, 'config'])['drone']['camera'])
        parameters.update({'preset': get_value(configs, [simulation, 'config'])['preset']})
        parameters.update({'size': get_value(configs, [simulation, 'config'])['forest']['size']})
        parameters.update({'ground': get_value(configs, [simulation, 'config'])['forest']['ground']})
        parameters.update({'color': get_value(configs, [simulation, 'config'])['material']['color']['plane']})

        # image center
        img_center = {}
        for capture in get_value(configs, [simulation, 'drone', 'camera', 'camera', 'captures']):
            img_center[capture['image']] = capture['center']

        # image data
        for img_name, img_data in get_value(images, [simulation, 'drone', 'camera']).items():
            img_number = int(img_name.split('-')[1])
            img_type = img_name.split('-')[-1] if img_name.count('-') == 2 else 'raw'

            df = df.append(pd.json_normalize({
                'number': img_number,
                'name': img_name,
                'type': img_type,
                'data': img_data,
                **img_center[img_number]
            }), ignore_index=True)

        df = df.astype({'number': np.int})
        df = df.sort_values(by=['number', 'type'], ignore_index=True)

        # stage, parameters and images
        data[simulation] = {
            'stage': get_value(images, [simulation, 'stage', 'image']),
            'parameters': parameters,
            'images': df
        }

    return data


def integrate_image(images, parameters, N=30):
    integrated = []

    # mask ground color
    color = rgba(parameters['color'])

    # current image
    for i, row in images.iterrows():
        img, img_x, img_z = row[['data', 'processed.x', 'processed.z']]

        # mask current image
        ground_mask = np.all(img == color, axis=2)

        # previous N images
        for j, prev_row in images[max(0, i - N):i][::-1].iterrows():
            prev_img, prev_img_x, prev_img_z = prev_row[['data', 'processed.x', 'processed.z']]
            delta = np.subtract([prev_img_x, prev_img_z], [img_x, img_z]).astype(np.int16)

            # shift and mask previous image
            prev_img_shifted = shift_image(prev_img, dx=delta[0], dy=delta[1])
            ground_mask = ground_mask | np.all(prev_img_shifted == color, axis=2)

        # append integrated image
        integrated.append(ground_mask.copy())

    return np.array(integrated)


def integrate_ground(images, parameters):
    ratio = parameters['resolution'] / parameters['coverage']

    # ground size
    size = np.floor(parameters['ground'] * ratio).astype(np.uint16)
    
    # ground tensor with scanned/visible counts
    ground = np.zeros((size, size, 2)).astype(np.uint16)
    
    # alpha tensor with scanned/visible counts per alpha (binned)
    alphas = np.zeros((size, size, 2, parameters['view'] + 2)).astype(np.uint16)

    # mask ground color
    color = rgba(parameters['color'])

    # current image
    for i, row in images.iterrows():
        img, img_x, img_z = row[['data', 'processed.x', 'processed.z']]

        # image center, radius and border position on target area
        center = np.floor(np.add([size / 2, size / 2], [img_x, img_z])).astype(np.int16)
        radius = np.floor([img.shape[0] / 2, img.shape[1] / 2]).astype(np.int16)
        border = np.array([center - radius, center + radius]).T

        # remove image pixels outside target area
        clipped = np.clip(border, 0, size - 1)
        offset = np.where(clipped - border == 0, None, clipped - border)

        # image slice indices inside target area
        slice_x, slice_y = slice(*clipped[0]), slice(*clipped[1])
        slice_x_offset, slice_y_offset = slice(*offset[0]), slice(*offset[1])

        # visible ground mask
        visible_mask = np.all(img[slice_y_offset, slice_x_offset] == color, axis=2)
        scanned_mask = np.full(visible_mask.shape, True)
        shift_mask = np.array([slice_x.start - center[0], slice_y.start - center[1]])
        
        # calculate alphas
        alphas_scanned_value = calculate_alphas(scanned_mask, shift_mask, parameters)
        alphas_visible_value = calculate_alphas(visible_mask, shift_mask, parameters)

        # count alphas scanned
        alphas_scanned = alphas[slice_y, slice_x][scanned_mask, 0, alphas_scanned_value]
        alphas[slice_y, slice_x][scanned_mask, 0, alphas_scanned_value] = alphas_scanned + 1

        # count alphas visible
        alphas_visible = alphas[slice_y, slice_x][visible_mask, 1, alphas_visible_value]
        alphas[slice_y, slice_x][visible_mask, 1, alphas_visible_value] = alphas_visible + 1

        # count ground scanned
        ground_scanned = ground[slice_y, slice_x][:, :, 0]
        ground[slice_y, slice_x][:, :, 0] = ground_scanned + 1

        # count ground visible
        ground_visible = ground[slice_y, slice_x][visible_mask, 1]
        ground[slice_y, slice_x][visible_mask, 1] = ground_visible + 1

    return ground, alphas[:, :, :, :-1] # drop last dimension


def calculate_alphas(mask, shift, parameters):
    ratio = parameters['resolution'] / parameters['coverage']

    # ground indices
    mask_x, mask_y = np.nonzero(mask)[::-1]
    distance_x, distance_y = mask_x + shift[0], mask_y + shift[1]

    # field of view triangle
    a = parameters['height'] * ratio
    b = np.linalg.norm([distance_x, distance_y], axis=0, keepdims=True)[0]
    c = np.sqrt(a**2 + b**2)

    # alpha values in degree rounded to nearest integer
    alpha = np.arccos((a**2 - b**2 + c**2) / (2 * a * c))
    alpha = np.floor(np.rad2deg(alpha)).astype(np.int16)

    # move alphas to last dimension
    # alpha[distance_x != 0] = -1 # 1D scan along x
    # alpha[distance_x != 0] = -1 # 1D scan along y

    return alpha


def aggregate_alphas(alphas, sample=None):
    
    # scanned alpha indices
    alphas_idx = np.nonzero(alphas[:, :, 0])
    sample_idx = np.random.choice(np.arange(alphas_idx[0].shape[0]), sample) if sample else slice(None)
    alphas_idx_x, alphas_idx_y, alphas_idx_a = [alpha_idx[sample_idx] for alpha_idx in alphas_idx]

    # alpha values
    scanned_alphas = alphas[alphas_idx_x, alphas_idx_y, 0, alphas_idx_a]
    visible_alphas = alphas[alphas_idx_x, alphas_idx_y, 1, alphas_idx_a]
    
    # alphas data
    data_alphas = np.array([alphas_idx_a, scanned_alphas, visible_alphas, visible_alphas / scanned_alphas]).T
    
    # alphas dataframe
    df_alphas = pd.DataFrame(data_alphas, columns=['alpha', 'scanned', 'visible', 'ratio'])
    df_alphas = df_alphas.apply(pd.to_numeric, downcast='integer')
    
    # aggregate alphas
    df_alphas_agg = df_alphas.groupby('alpha').mean()
    # df_alphas_agg['result'] = df_alphas_agg['visible'] / df_alphas_agg['scanned'] # same as ratio mean
    
    return df_alphas_agg.reset_index()


def grayscale_image(image):
    return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])


def normalize_image(image, cap=None):
    image = image.astype(np.float64)
    for i in range(3):
        minimum, maximum = image[..., i].min(), image[..., i].max()
        if minimum != maximum:
            image[..., i] -= minimum
            image[..., i] *= ((cap if cap else maximum) / (maximum - minimum))
    return image.astype(np.uint8)


def shift_image(image, dx, dy):
    X = image.copy()
    X = np.roll(X, dy, axis=0)
    X = np.roll(X, dx, axis=1)
    if dy > 0:
        X[:dy, :] = 0
    elif dy < 0:
        X[dy:, :] = 0
    if dx > 0:
        X[:, :dx] = 0
    elif dx < 0:
        X[:, dx:] = 0
    return X


def plot_heatmap(ax, image, label):
    ax.set_title(label)
    sns.heatmap(image, xticklabels=False, yticklabels=False, ax=ax)


def plot_histogram(ax, df, x, label):
    ax.set_title(label)
    sns.histplot(df, x=x, bins=40, alpha=0.8, kde=True, ax=ax)


def plot_image(ax, image, label):
    ax.set_title(label)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(image)


def plot_images(images, labels, rows=5, cols=5):
    fig, axes = plt.subplots(rows, cols, figsize=(16, 16))
    axes = [a for axs in axes for a in axs]
    for i, image in enumerate(images[:rows * cols]):
        plot_image(axes[i], image, labels[i])
    plt.show()


def export_plot(fig, path):
    fig.savefig(path, transparent=True)
    fig.clf()
    plt.close()


def rgba(color):
    r = (color & 0xff0000) >> 16
    g = (color & 0x00ff00) >> 8
    b = (color & 0x0000ff)
    a = 255
    return np.array([r, g, b, a])


def get_value(dic, keys):
    for key in keys:
        dic = dic[key]
    return dic


def set_value(dic, keys, value):
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = value
    return dic
