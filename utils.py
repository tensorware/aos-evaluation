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


def load_data(path, limit=None):
    data = {}

    # data zip paths
    zip_paths = sorted(gb.glob(path, recursive=True))
    zip_paths = zip_paths[:limit] if limit and limit < len(zip_paths) else zip_paths

    # extract zip files
    data_folder = tempfile.mkdtemp()
    for zip_path in zip_paths:
        with ZipFile(zip_path, 'r') as z:
            zip_name = os.path.basename(zip_path).rsplit('.', 1)[0]
            folder_path = os.path.join(data_folder, zip_name)

            print(f'extract {folder_path}')
            z.extractall(folder_path)

    # load json files
    configs = {}
    for config_path in sorted(gb.glob(os.path.join(data_folder, '**', '*.json'), recursive=True)):
        config_file = os.path.basename(config_path)
        config_dir = os.path.dirname(os.path.relpath(config_path, data_folder))

        keys = config_dir.split(os.path.sep) + [config_file]
        set_value(configs, keys, json.load(open(config_path)))

    # load png images
    images = {}
    for image_path in sorted(gb.glob(os.path.join(data_folder, '**', '*.png'), recursive=True)):
        image_file = os.path.basename(image_path)
        image_dir = os.path.dirname(os.path.relpath(image_path, data_folder))

        keys = image_dir.split(os.path.sep) + [image_file]
        set_value(images, keys, np.array(Image.open(image_path)))

    # generate data
    for simulation in images:

        # parameters
        parameters = load_parameters(simulation, configs)

        # dataframe images
        df_images = load_images(simulation, configs, images, parameters)
        df_images = df_images.sort_values('number', ignore_index=True)
        df_images = df_images.reset_index(drop=True)

        # dataframe persons
        df_persons = load_persons(simulation, configs)

        # dataframe trees
        df_trees = load_trees(simulation, configs)

        # stage image
        stage = get_value(images, [simulation, 'stage'])['image.png']

        # set simulation data
        data[simulation] = {
            'parameters': parameters,
            'images': df_images,
            'persons': df_persons,
            'trees': df_trees,
            'stage': stage
        }

    return data


def load_parameters(simulation, configs):
    
    # load json
    settings_json = json.load(open('settings.json'))
    config_json = get_value(configs, [simulation, 'config.json'])
    drone_json = get_value(configs, [simulation, 'drone', 'drone.json'])

    # parameters
    parameters = config_json.copy()

    # delete parameters
    del_keys(parameters, 'drone.cpu')
    del_keys(parameters, 'drone.eastWest')
    del_keys(parameters, 'drone.northSouth')
    del_keys(parameters, 'drone.camera.images')
    del_keys(parameters, 'forest.trees')

    # add parameters
    set_value(parameters, 'drone.coverage', get_value(drone_json, 'coverage'))
    set_value(parameters, 'url', f'{get_value(settings_json, "simulation.url")}#preset={get_value(parameters, "preset")}')

    return parameters


def load_images(simulation, configs, images, parameters):
    
    # load json
    camera_json = get_value(configs, [simulation, 'drone', 'camera', 'camera.json'])

    # image center
    img_center = {}
    for capture in get_value(camera_json, 'captures'):
        img_number = get_value(capture, 'image')
        img_center[img_number] = get_value(capture, 'center')

    # image dataframes
    df_images = []
    for img_name, img_data in get_value(images, [simulation, 'drone', 'camera']).items():

        # image number and type
        img_parts = img_name.split('.')[0].split('-')
        img_number = int(img_parts[1])
        img_type = img_parts[-1]

        # normalize nested objects
        df_image = pd.json_normalize({
            'number': img_number,
            'name': img_name,
            'type': img_type,
            'data': img_data,
            **img_center[img_number]
        })

        # cast numeric datatypes
        cast = {'float32': ['x', 'y', 'z']}
        for dtype, columns in cast.items():
            df_image[columns] = df_image[columns].astype(dtype)

        # create preview url
        df_image['url'] = '&'.join([
            get_value(parameters, 'url'),
            f'drone.height={get_value(parameters, "drone.height", 0)}',
            f'drone.rotation={get_value(parameters, "drone.rotation", 0)}',
            f'drone.eastWest={df_image["x"].item():0.2f}',
            f'drone.northSouth={df_image["z"].item():0.2f}',
            f'drone.camera.view={get_value(parameters, "drone.camera.view", 0)}'
        ])

        # append to dataframes
        df_images.append(df_image)

    return pd.concat(df_images, ignore_index=True)


def load_persons(simulation, configs):
    
    # load json
    persons_json = get_value(configs, [simulation, 'forest', 'persons.json'])

    # normalize nested objects
    df_persons = pd.json_normalize(persons_json['tracks'])

    # cast numeric datatypes
    cast = {'float32': [x for x in df_persons.columns if x not in ['person', 'activity']]}
    for dtype, columns in cast.items():
        df_persons[columns] = df_persons[columns].astype(dtype)

    return df_persons


def load_trees(simulation, configs):
    
    # load json
    trees_json = get_value(configs, [simulation, 'forest', 'trees.json'])

    # normalize nested objects
    df_trees = pd.json_normalize(trees_json['locations'])

    # cast numeric datatypes
    cast = {'float32': [x for x in df_trees.columns if x not in ['tree']]}
    for dtype, columns in cast.items():
        df_trees[columns] = df_trees[columns].astype(dtype)

    return df_trees


def sample_data(parameters):

    # images step size and coverage
    step = get_value(parameters, 'drone.camera.sampling')
    coverage = get_value(parameters, 'drone.coverage')

    # possible increased samplings steps
    sup_steps = np.arange(step, coverage + step, step=step)

    # possible decreased number of captures
    sub_samples = np.arange(1, np.ceil(coverage / step).astype(np.int16) + 1, step=1)

    # possible partitions
    partitions = coverage / sup_steps

    # create data array
    data = []
    for num in sub_samples:

        # index to nearest partition
        idx = np.abs(num - partitions).argmin()

        # difference to nearest partition
        diff = np.abs(num - partitions[idx])

        # append to data       
        data.append({'num': num, 'idx': idx, 'diff': diff, 'dist': sup_steps[idx]})
    
    # group by index and pick nearest approximation of partition
    df_partitions = pd.DataFrame(data)
    df_partitions = df_partitions.loc[df_partitions.groupby('idx')['diff'].idxmin()]
    df_partitions = df_partitions.convert_dtypes()

    # dictionary with { num_captures_per_point: index_to_pick_every_i_th_image }
    N = dict(zip(df_partitions['num'].astype(int), df_partitions['idx'].astype(int) + 1))

    # dictionary with { num_captures_per_point: distance_in_m_between_every_image }
    M = dict(zip(df_partitions['num'].astype(int), df_partitions['dist'].astype(float)))

    return N, M


def integrate_image(df_images, parameters, N=30):
    
    # mask ground color
    color = to_rgba(get_value(parameters, 'material.color.plane'))

    # current image
    integrated = []
    for i, row in df_images.iterrows():

        # image center in pixel
        img = row['data']
        img_x = to_pixel(row['x'], parameters)
        img_y = to_pixel(row['z'], parameters)

        # mask current image
        ground_mask = np.all(img == color, axis=2)

        # previous N images
        for j, prev_row in df_images[max(0, i - N):i][::-1].iterrows():
            prev_img = prev_row['data']
            prev_img_x = to_pixel(prev_row['x'], parameters)
            prev_img_y = to_pixel(prev_row['z'], parameters)

            # distance between images
            delta = np.subtract([prev_img_x, prev_img_y], [img_x, img_y]).astype(np.int16)

            # shift and mask previous image
            prev_img_shifted = shift_image(prev_img, dx=delta[0], dy=delta[1])
            ground_mask = ground_mask | np.all(prev_img_shifted == color, axis=2)

        # append integrated image
        integrated.append(ground_mask.copy())

    return np.array(integrated)


def integrate_ground(df_images, parameters):
    
    # ground size
    size = to_pixel(get_value(parameters, 'forest.ground'), parameters)

    # ground tensor with scanned/visible counts
    ground = np.zeros((size, size, 2)).astype(np.uint16)

    # alpha tensor with scanned/visible counts per alpha (binned)
    alphas = np.zeros((size, size, 2, get_value(parameters, 'drone.camera.view') + 1)).astype(np.uint16)

    # mask ground color
    color = to_rgba(get_value(parameters, 'material.color.plane'))

    # current image
    for i, row in df_images.iterrows():

        # image center in pixel
        img = row['data']
        img_x = to_pixel(row['x'], parameters)
        img_y = to_pixel(row['z'], parameters)

        # image center, radius and border position on target area
        center = np.floor(np.add([size / 2, size / 2], [img_x, img_y])).astype(np.int16)
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

    # drop last dimension *
    return ground, alphas[:, :, :, :-1]


def calculate_alphas(mask, shift, parameters):
    
    # ground indices
    mask_x, mask_y = np.nonzero(mask)[::-1]
    distance_x, distance_y = mask_x + shift[0], mask_y + shift[1]

    # field of view triangle
    a = to_pixel(get_value(parameters, 'drone.height'), parameters)
    b = np.linalg.norm([distance_x, distance_y], axis=0, keepdims=True)[0]
    c = np.sqrt(a**2 + b**2)

    # alpha values floored to nearest integer
    alpha = np.arccos((a**2 - b**2 + c**2) / (2 * a * c))
    alpha = np.floor(np.rad2deg(alpha)).astype(np.int16)

    # move alphas to last dimension *
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

    return df_alphas_agg.reset_index()


def ground_positions(df_positions, rectangle, parameters):
    
    # ground size
    size = to_pixel(get_value(parameters, 'forest.ground'), parameters)
    
    # convert positions to integer indices
    positions = df_positions.apply(lambda x: to_pixel(x, parameters)).to_numpy()
    idxs = positions + np.floor(size / 2).astype(np.int16) - 1
    
    # remove positions outside ground area
    xy_min, xy_max = np.array(rectangle).T.astype(np.int16)
    idxs = np.delete(idxs, np.where((idxs > xy_max - 1) | (xy_min > idxs))[0], axis=0)
    
    return idxs


def calculate_statistics(df_images, df_trees, ground, parameters):
    
    # ground visibility
    ground_scanned = np.count_nonzero(ground[:, :, 0])
    ground_visible = np.count_nonzero(ground[:, :, 1])
    ground_visibility = ground_visible / ground_scanned
    
    # image width and height
    coverage = to_pixel(get_value(parameters, 'drone.coverage'), parameters)
    w, h = np.ceil(coverage / 2).astype(np.int16), np.ceil(coverage / 2).astype(np.int16)
    
    # image positions
    ground_rect = [[0, ground.shape[0]], [0, ground.shape[1]]]
    images_pos = ground_positions(df_images[['x', 'z']], ground_rect, parameters)
    
    # tree positions inside image
    trees_count = []
    for x, y in images_pos:
        image_rect = np.array([[x - w, x + w], [y - h, y + h]])
        trees_pos = ground_positions(df_trees[['position.x', 'position.z']], image_rect, parameters)
        trees_count.append(trees_pos.shape[0])
    
    return {
        'ground_visibility': ground_visibility,
        'trees_per_image': np.mean(trees_count)
    }


def to_pixel(value, parameters):
    coverage = get_value(parameters, 'drone.coverage')
    resolution = get_value(parameters, 'drone.camera.resolution')
    return np.floor(value * resolution / coverage).astype(np.int16)


def to_rgba(color):
    r = (color & 0xff0000) >> 16
    g = (color & 0x00ff00) >> 8
    b = (color & 0x0000ff)
    a = 255
    return np.array([r, g, b, a])


def grayscale_image(image):
    return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])


def normalize_image(image, cap=None):
    image = image.astype(np.float64)
    for i in range(3):
        minimum, maximum = image[..., i].min(), image[..., i].max()
        if minimum != maximum:
            image[..., i] -= minimum
            image[..., i] *= (cap if cap else maximum) / (maximum - minimum)
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
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    sns.heatmap(image, ax=ax)


def plot_histogram(ax, df, x, label):
    ax.set_title(label)
    sns.histplot(df, x=x, bins=40, alpha=0.8, kde=True, ax=ax)


def plot_image(ax, image, label):
    ax.set_title(label)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    ax.imshow(image)


def plot_images(images, labels, rows=5, cols=5, figsize=(16, 16)):
    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    if rows > 1 and cols > 1:
        axs = [y for x in axs for y in x]
    for i, image in enumerate(images[:rows * cols]):
        ax = axs[i] if rows + cols > 2 else axs
        plot_image(ax, image, labels[i])
    return fig, axs


def plot_ground(ground, labels, rows=1, cols=3, figsize=(24, 6)):
    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    plot_heatmap(axs[0], ground[:, :, 0], labels[0])
    plot_heatmap(axs[1], ground[:, :, 1], labels[1])
    plot_heatmap(axs[2], normalize_image(ground[:, :, 1] > 0), labels[2])
    return fig, axs


def export_plot(fig, path, close=True):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, transparent=True)
    if close:
        fig.clf()
        plt.close()


def colors(cmap, size):
    c = plt.cm.get_cmap(cmap, size)
    return [c(i) for i in range(size)]


def get_keys(keys):
    if isinstance(keys, list):
        return keys
    return keys.split('.')


def del_keys(dic, keys):
    keys = get_keys(keys)
    for key in keys[:-1]:
        dic = dic.get(key, {})
    return dic.pop(keys[-1], None)


def get_value(dic, keys, default=None):
    keys = get_keys(keys)
    for key in keys:
        dic = dic.get(key, {})
    return dic or default


def set_value(dic, keys, value):
    keys = get_keys(keys)
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = value
    return dic
