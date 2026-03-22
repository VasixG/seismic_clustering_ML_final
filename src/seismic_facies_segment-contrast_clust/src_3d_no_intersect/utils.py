import yaml
import os
import numpy as np
import random
import torch
import importlib
import errno
import xarray as xr
from segysak.segy import segy_writer
import matplotlib.pyplot as plt

def read_optim_config(config_path):
    with open(config_path) as fh:
        config = yaml.load(fh, Loader=yaml.FullLoader)
    return config


def set_seeds(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def create_folder_if_not_exists(config):
    '''creates experiment folder and save the experiment config'''
    fold_path = os.path.join(config['experim_folder'],
                             config['sweep_name'])
    
    try:
        os.makedirs(fold_path)  # makedirs handles creating intermediate directories
        print(f"Folder '{fold_path}' created successfully.")
        save_conf_path = os.path.join(fold_path, 'config.yaml')
        with open(save_conf_path, 'w') as file:
            yaml.dump(config, file)
        weight_fold = os.path.join(fold_path, 'weights')
        os.makedirs(weight_fold)
        pics_fold = os.path.join(fold_path, 'pics')
        os.makedirs(pics_fold)
        graph_fold = os.path.join(fold_path, 'graphs')
        os.makedirs(graph_fold)
        # if config['save_cubes']:
        cubes_fold = os.path.join(fold_path, 'cubes')
        os.makedirs(cubes_fold)
    except OSError as e:
        if e.errno == errno.EEXIST:
            raise ValueError(f"Folder '{fold_path}' already exists.")  #Custom Error
        else:
            raise  # Re-raise other OSError exceptions (e.g., permission errors)
    
    return fold_path


def parse_sampling_strategy(sampling_strategy: str, *args, **kwargs):
    available_strat = ['RandomSampler', 'GridSampler', 'TPESampler', 
                       'CmaEsSampler', 'NSGAIISampler', 'QMCSampler', 
                        'GPSampler', 'BoTorchSampler', 'BruteForceSampler']
    if sampling_strategy not in available_strat:
        print('Sampler is not specified correctly, use default TPESampler')
        return None
    else:
        module_name = 'optuna.samplers' 
        module = importlib.import_module(module_name)
        sampler_class = getattr(module, sampling_strategy)
        sampler = sampler_class(*args, **kwargs)
        print('Sampler imported successfully!')
        return sampler


def save_segy(results_xr, save_fold, f_name):
    print('Save SEGY')
    results_xr.attrs['coord_scalar'] = 1.00000
    results_xr.attrs['sample_rate'] = 2.0000
    results_xr.attrs['source_file'] = 'Cube_resulting'
    cube_path = os.path.join(save_fold, f'{f_name}.segy')
    # the dictionary with byte location of the coords
    
    trace_header_map = dict(iline=189,
                         xline=193,
                         cdp_x=73,
                         cdp_y=77)
    segy_writer(
            results_xr,
            cube_path,
            trace_header_map=trace_header_map
        )
    
def save_segy_from_np_cube(dataset_path, numpy_path):
    
    # Load dataset and numpy array
    ds = xr.open_dataset(dataset_path)
    data_array = np.load(numpy_path)
    
    print(f"Replacing existing 'data' variable with new values")
    ds['data'] = (ds['data'].dims, data_array)  
    print(ds.dims)
    # Generate output filename and path
    f_name = os.path.split(numpy_path)[1].split(".")[0]
    fold_path = os.path.split(numpy_path)[0]
    
    # Save as SEGY
    save_segy(ds, fold_path, f_name)
    
    return ds


def down_cube_by_sel_ix_line(bolvanka_full, 
                             down_cube_template, 
                             np_out):
    if down_cube_template.data.shape == np_out.shape:
        print('The array shape is downsampled aldready!')
        down_cube_template['data'] = (down_cube_template.dims, 
                                      np_out)
        return down_cube_template
    bolvanka_full['data'] = (bolvanka_full.dims, np_out)
    result_cube = bolvanka_full.sel({'iline': down_cube_template.iline, 
                'xline': down_cube_template.xline, 
                # 'twt': template_cube.twt
                })
    return result_cube


class RunStudy:

    def __init__(self, config, result_folder, trial):

        self.config = config
        self.result_folder = result_folder
        self.plots_dict = {}
        self.fig_fold = os.path.join(self.result_folder, 'pics')
        # self.imgs_dict = {}

    def append(self, path, value):
        '''append new point in some plot in plots_dict'''
        # pass
        if path not in self.plots_dict:
            self.plots_dict[path] = []
        
        self.plots_dict[path].append(value)

    def save_plots(self):
        '''call it in the end of the trainig'''
        graph_fold = os.path.join(self.result_folder, 'graphs')
        for key, value in self.plots_dict.items():
            subfold, img_name = os.path.split(key)
            plot_fold = os.path.join(graph_fold, subfold)
            os.makedirs(plot_fold, exist_ok=True)
            fig, ax = plt.subplots(1)
            ax.plot(np.arange(len(value)), value, '--o')
            ax.grid(axis='both', linestyle='--', alpha=0.7)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(key.split('/')[-1])
            fig.tight_layout()
            fig.savefig(os.path.join(plot_fold, img_name))
            plt.close(fig)

    def upload(self, fig_name, fig):
        '''upload image provided_class'''
        full_path = os.path.join(self.fig_fold, fig_name)
        fig.savefig(full_path)
        plt.close(fig)

    def empty_storage(self):
        self.plots_dict.clear()  # More efficient than reassigning

    # def get_result_folder(self):

    #     experim_folder = self.config['experim_folder']
    #     self.result_fold = os.path.join(experim_folder, self.config['sweep_name'])

def read_or_create_metric(filepath, metric=0.0):
    """
    Read a float from a txt file, or create the file with default value if it doesn't exist.
    
    Args:
        filepath: Path to the txt file
        metric: Float value to write if file doesn't exist
    
    Returns:
        The float value read from file or the default value
    """
    if os.path.exists(filepath):
        # Read the float from existing file
        with open(filepath, 'r') as f:
            # try:
            value = float(f.read().strip())
                
            # except (ValueError, IOError) as e:
            # print(f"Error reading file, using default: {e}")
            return metric
    else:
        # Create file with default value
        with open(filepath, 'w') as f:
            f.write(str(metric))
        print(f"Created {filepath} with default value {metric}")
        return metric