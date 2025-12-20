import yaml
import os

def load_config(config_path):
    """
    Load and parse the YAML configuration file.

    Parameters:
        config_path (str): Path to the YAML configuration file.
    
    Returns:
        dict: Configuration data parsed from YAML.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        raise RuntimeError(f"Error loading configuration file '{config_path}': {e}")

def load_AbacusHOD_config(config_path):
    """
    Load AbacusHOD configuration from a YAML file.

    Parameters:
        config_path (str): Path to the YAML configuration file.
        
    Returns:
        sim_params (dict): Simulation parameters.
        HOD_params (dict): HOD parameters.
        clustering_params (dict): Clustering parameters.
    """
    config_full = load_config(config_path)
    sim_params = config_full.get("sim_params", {})
    HOD_params = config_full.get("HOD_params", {})
    clustering_params = config_full.get("clustering_params", {})
    return sim_params, HOD_params, clustering_params

def save_config(config, config_path):
    """
    Save configuration dictionary to a YAML file.

    Parameters:
        config (dict): Configuration data to be saved.
        config_path (str): Path to the YAML configuration file.
    """
    try:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, sort_keys=False)
    except Exception as e:
        raise RuntimeError(f"Error saving configuration to '{config_path}': {e}")
    
    
_DEFAULT_CONFIG = {
    'sim_params':{
        'cleaned_halos': True,
        'output_dir': '/pscratch/sd/s/siyizhao/tmp/',
        'sim_dir': '/global/cfs/projectdirs/desi/cosmosim/Abacus/',
        'sim_name': 'Abacus_pngbase_c300_ph000',
        'subsample_dir': '/pscratch/sd/s/siyizhao/AbacusSummit/subsample_desidr2_profile_withAB/',
        'z_mock': 2.5,
        'force_mt': True
        },
    'HOD_params':{
        'tracer_flags': {'ELG': False, 'LRG': False, 'QSO': True},
        'use_particles': False,
        'use_profiles': True,
        'want_AB': True,
        'want_rsd': True,
        'want_dv': True,
        'write_to_disk': False,
        'QSO_params': {'logM_cut': 12.03,
        'logM1': 13.11,
        'sigma': 0.08,
        'alpha': 1.03,
        'kappa': 0.39,
        'alpha_c': 0.92,
        'alpha_s': 2.7,
        'ic': 1.0,
        'profile_code': 1},
        'dv_draw_Q': '/global/homes/s/siyizhao/projects/fihobi/data/dv_draws/QSO_z2.3-2.8_CDF.npz'
        },
    'clustering_params':{
        'bin_params': {'logmax': 1.5, 'logmin': -1.0, 'nbins': 15},
        'clustering_type': 'all',
        'pi_bin_size': 40,
        'pimax': 40
        }
}

def config_Abacus(config=None, config_path=None):
    """
    Load or save AbacusHOD configuration.

    Parameters:
        config (dict, optional): Configuration data to be saved. If None, load from file.
        config_path (str): Path to the YAML configuration file.
        
    Returns:
        sim_params (dict): Simulation parameters.
        HOD_params (dict): HOD parameters.
        clustering_params (dict): Clustering parameters.
    """
    if config is not None:
        if config_path is None:
            sim_params = config.get("sim_params", {})
            output_dir = sim_params.get('output_dir', './')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                print(f"Created output directory: {output_dir}")
            config_path = output_dir + 'config.yaml'
        print('config provided, saving to:', config_path)
        save_config(config, config_path)
    else: # config is None
        if config_path is None:
            config_path = 'example.yaml'
            print("No config or file provided. Using an example configuration and saving it to:", config_path)
            save_config(_DEFAULT_CONFIG, config_path)
    return load_AbacusHOD_config(config_path)
