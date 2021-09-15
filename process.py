import os
import json
import argparse

import glob as gb
import utils as ut
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main(args):
    """ Execute:
        -----------------------------------------------------------------------------------------------------------------------------
        python process.py --input data/input/v10-default/forest-21 --output data/output/v10-default/forest-21 & \
        python process.py --input data/input/v10-default/forest-22 --output data/output/v10-default/forest-22 & \
        python process.py --input data/input/v10-default/forest-23 --output data/output/v10-default/forest-23 &
        -----------------------------------------------------------------------------------------------------------------------------
    """
    
    # zip files
    zip_files = os.path.join(args.input, '*.zip')
    for zip_path in sorted(gb.glob(zip_files, recursive=True)):
        
        # load data
        data = ut.load_data(zip_path)
        
        # load dataframes
        simulation = next(iter(data))
        df_images = data[simulation]['images']
        df_trees = data[simulation]['trees']

        # load parameters
        parameters = data[simulation]['parameters']
        print(f'process {simulation}', json.dumps(parameters, indent=4))
        
        # output folder
        output_folder = os.path.join(args.output, simulation)
        os.makedirs(output_folder, exist_ok=True)
        name_suffix = ut.get_value(parameters, 'preset')
        
        # generate subsamples
        N, M = ut.sample_data(parameters)
        for n in N.keys():
            
            # images by step size
            df_integrate = df_images.iloc[::N[n], :]
            print(f'export {simulation} for {df_integrate.shape[0]:4d} images taken every {M[n]:5.2f} meter (N={n:03d})')
            
            # integrate ground
            ground, alphas = None, None
            ground, alphas = ut.integrate_ground(df_integrate, parameters)
            
            # calculate statistics
            statistics = ut.calculate_statistics(df_integrate, df_trees, ground, parameters)

            # plot integrated ground
            fig, axs = ut.plot_ground(ground, ['scanned pixels (count)', 'visible pixels (count)', f'visibility ({statistics["ground_visibility"]:.2f})'])

            # plot tree positions
            ground_rect = [[0, ground.shape[0]], [0, ground.shape[1]]]
            trees_pos = ut.ground_positions(df_trees[['position.x', 'position.z']], ground_rect, parameters)
            axs[2].scatter(*trees_pos.T, marker='o', facecolor='firebrick', edgecolor='lightgray', linewidth=0.7)

            # export ground image
            ut.export_plot(fig, os.path.join(output_folder, f'ground-{name_suffix}-N{n:03d}.png'))
            
            # export data
            np.savez_compressed(os.path.join(output_folder, f'data-{name_suffix}-N{n:03d}.npz'), ground=ground, alphas=alphas, statistics=statistics)

        # plot stage image
        fig, ax = plt.subplots(figsize=(16, 8))
        ut.plot_image(ax, data[simulation]['stage'], 'stage')
        
        # export stage image
        ut.export_plot(fig, os.path.join(output_folder, f'stage-{name_suffix}.png'))
        
        # export parameters
        with open(os.path.join(output_folder, f'parameters-{name_suffix}.json'), 'w') as f:
            json.dump(parameters, f, indent=4)
        print()


if __name__ == '__main__':
    
    # arguments
    argp = argparse.ArgumentParser(description='AOS-Evaluation')
    argp.add_argument('--input', default=os.path.join('data', 'input'), type=str, help='folder path of simulation zip files [PATH]')
    argp.add_argument('--output', default=os.path.join('data', 'output'), type=str, help='folder path of evaluation export files [PATH]')
    args = argp.parse_args()
    
    # main
    main(args)
