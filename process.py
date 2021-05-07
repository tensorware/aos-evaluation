import os
import json
import argparse

import glob as gb
import utils as ut
import numpy as np
import pandas as pd
import seaborn as sns

import ipywidgets as pyw
import matplotlib.pyplot as plt


def main(args):

    # zip files
    zip_files = os.path.join(args.path, '*.zip')
    for zip_path in sorted(gb.glob(zip_files, recursive=True)):

        # load data
        data = ut.load_data(zip_path)
        simulation = next(iter(data))

        # load images
        df = data[simulation]['images']
        df = df[df['type'] == 'monochrome']
        df = df.reset_index(drop=True)

        # load parameters
        parameters = data[simulation]['parameters']
        parameters['images'] = df.shape[0]
        print(f'process {simulation}', json.dumps(parameters, indent=4), '\n')

        # output folder
        output_folder = os.path.join(args.output, simulation)
        os.makedirs(output_folder, exist_ok=True)
        name_suffix = f'-{parameters["preset"]}-{parameters["view"]}'

        # integrate ground
        ground, alphas = ut.integrate_ground(df, parameters)
        df_alpha = pd.DataFrame(alphas.T, columns=['x', 'y', 'alpha'])
        df_alpha = df_alpha.sample(n=np.minimum(df_alpha.shape[0], 10**5), random_state=42)
        # df_alpha.to_csv(os.path.join(output_folder, f'alpha{name_suffix}.csv'), index=False)
        np.save(os.path.join(output_folder, f'ground{name_suffix}.npy'), ground)
        np.save(os.path.join(output_folder, f'alpha{name_suffix}.npy'), alphas)

        # plot stage image
        fig, ax = plt.subplots(figsize=(16, 16))
        ut.plot_image(ax, data[simulation]['stage'], 'stage')
        ut.export_plot(fig, os.path.join(output_folder, f'stage{name_suffix}.png'))

        # plot ground images
        fig, axs = plt.subplots(1, 3, figsize=(29, 7))
        fig.tight_layout()
        ut.plot_heatmap(axs[0], ground[:, :, 0], 'scanned (count)')
        ut.plot_heatmap(axs[1], ground[:, :, 1], 'captured (count)')
        ut.plot_heatmap(axs[2], ut.normalize_image(ground[:, :, 1] > 0), 'captured (binary)')
        ut.export_plot(fig, os.path.join(output_folder, f'ground{name_suffix}.png'))

        # plot alpha distribution
        fig, ax = plt.subplots(figsize=(12, 8))
        ut.plot_histogram(ax, df_alpha, 'alpha', f'field of view {parameters["view"]}°')
        ut.export_plot(fig, os.path.join(output_folder, f'alpha{name_suffix}.png'))

        # export parameters
        with open(os.path.join(output_folder, f'parameters{name_suffix}.json'), 'w') as f:
            json.dump(parameters, f, indent=4)


if __name__ == '__main__':

    # arguments
    argp = argparse.ArgumentParser(description='AOS-Evaluation')
    argp.add_argument('--path', default=os.path.join('data'), type=str, help='folder path of simulation zip files [PATH]')
    argp.add_argument('--output', default=os.path.join('results'), type=str, help='folder path of evaluation export files [PATH]')
    args = argp.parse_args()

    # main
    main(args)
