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


def get_value(dic, keys):
    for key in keys:
        dic = dic[key]
    return dic


def set_value(dic, keys, value):
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = value
    return dic


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


def rgba_color(color):
    return np.array([(color & 0xff0000) >> 16, (color & 0x00ff00) >> 8, (color & 0x0000ff), 255])


def grayscale_image(image):
    return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])


def normalize_image(image):
    image = image.astype(np.float64)
    for i in range(3):
        minimum, maximum = image[..., i].min(), image[..., i].max()
        if minimum != maximum:
            image[..., i] -= minimum
            image[..., i] *= (maximum / (maximum - minimum))
    return image.astype(np.uint8)


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


def integrate_image(images, parameters, N=30):
    integrated = []

    # mask ground color
    color = rgba_color(parameters['color'])

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
    size = np.ceil((parameters['ground'] + 1) * ratio).astype(np.int16)
    ground = np.zeros((size, size, 3))

    # mask ground color
    color = rgba_color(parameters['color'])

    # current image
    alphas = []
    for i, row in images.iterrows():
        img, img_x, img_z = row[['data', 'processed.x', 'processed.z']]
        center = np.ceil(np.add([size / 2, size / 2], [img_x, img_z])).astype(np.int16)

        slice_outer = np.array([img.shape[0], img.shape[1]]) // 2
        slice_border = np.array([center - slice_outer, center + slice_outer]).T

        slice_clipped = np.clip(slice_border, 0, size - 1)
        slice_offset = np.where(slice_clipped - slice_border == 0, None, slice_clipped - slice_border)

        # slice indices
        slice_x, slice_y = slice(*slice_clipped[0]), slice(*slice_clipped[1])
        slice_x_offset, slice_y_offset = slice(*slice_offset[0]), slice(*slice_offset[1])

        # visible ground indices
        visible_mask = np.all(img == color, axis=2)[slice_y_offset, slice_x_offset]
        visible_mask_x, visible_mask_y = np.nonzero(visible_mask)
        visible_ground_x, visible_ground_y = visible_mask_x + slice_x.start, visible_mask_y + slice_y.start

        # visible ground alpha
        distance = np.linalg.norm([center[0] - visible_ground_x, center[1] - visible_ground_y], axis=0, keepdims=True)
        alpha = np.arcsin(distance / np.sqrt(distance**2 + (parameters['height'] * ratio)**2))

        # count captures (0: red)
        ground_red = ground[slice_y, slice_x][:, :, 0]

        # count visibility (1: green)
        ground_green = ground[slice_y, slice_x][visible_mask, 1]

        # count ? (2: blue)
        ground_blue = ground[slice_y, slice_x][visible_mask, 2]

        # update image
        ground[slice_y, slice_x][:, :, 0] = ground_red + 1
        ground[slice_y, slice_x][visible_mask, 1] = ground_green + 1
        ground[slice_y, slice_x][visible_mask, 2] = ground_blue + 0

        # interesting (last alpha captured)
        # np.putmask(tmp[slice_y, slice_x][:, :, 2], visible_mask, alpha)

        # append alpha angles per visible points
        alphas.append([visible_ground_x, visible_ground_y, alpha])

    # flatten alpha
    alphas = np.array(alphas)
    alphas = np.array([
        np.hstack(alphas[:, 0]).flatten(),
        np.hstack(alphas[:, 1]).flatten(),
        np.rad2deg(np.hstack(alphas[:, 2]).flatten())
    ]).astype(np.float64)

    return ground, alphas
