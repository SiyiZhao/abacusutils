#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate AbacusHOD mock. If path to CONFIG is provided, it will be used, else CONFIG in this file is used.

Usage
-----
$ python ./AbacusHODmock.py --help
    
Author: Siyi Zhao
"""
import argparse
import numpy as np
import sys, os
sys.path.insert(0, os.path.expanduser('~/lib/'))
from abacusnbody.hod.abacus_hod import AbacusHOD
import abacusnbody.hod.abacus_hod as ah
print(ah.__file__)
from config_helper import config_Abacus, save_config

# Example HOD parameters for QSO at z~2.5
logM_cut, logM1, sigma, alpha, kappa, alpha_c, alpha_s = [11.7209282, 11.1767657, 0.04491125, 0.90375827, 3.85540798, 1.81135532, 1.90593657]
CONFIG = {
    'sim_params':{
        'cleaned_halos': True,
        'output_dir': '/pscratch/sd/s/siyizhao/data_learnCosm/AbacusMocks/QSOz5-fnl100bf/',
        'sim_dir': '/global/cfs/projectdirs/desi/cosmosim/Abacus/',
        'sim_name': 'Abacus_pngbase_c302_ph000',
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
        'QSO_params': {'logM_cut': logM_cut,      
        'logM1': logM1,
        'sigma': sigma,
        'alpha': alpha,
        'kappa': kappa,
        'alpha_c': alpha_c,
        'alpha_s': alpha_s,
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

def find_path(Ball):
    if Ball.want_rsd:
        rsd_string = '_rsd'
        if Ball.want_dv:
            rsd_string += '_dv'
    else:
        rsd_string = ''
    path = (Ball.mock_dir) / ('galaxies' + rsd_string)
    return path

def compute_all(Ball, out=False, cfg=None, want_rsd=None, want_dv=None, want_clustering=False, nthread=1, verbose=False):
    """
    Generate the mock, and compute both wp and multipoles.
    If out=True, write the mock (and clustering) to disk, please provide cfg.
    """
    re = {}
    if want_rsd is not None:
        print(f"Original want_rsd: {Ball.want_rsd}")
        Ball.want_rsd = want_rsd
        print(f"Set want_rsd to {want_rsd}")
    if want_dv is not None:
        print(f"Original want_dv: {Ball.want_dv}")
        Ball.want_dv = want_dv
        print(f"Set want_dv to {want_dv}")
    want_rsd = Ball.want_rsd
    want_dv = Ball.want_dv
    if want_rsd & want_dv:
        fn_ext = '_dv'
    else:
        fn_ext = ''
    mock_dict = Ball.run_hod(tracers=Ball.tracers, want_rsd=want_rsd, Nthread = nthread, verbose = verbose, write_to_disk=out, fn_ext=fn_ext)
    re['mock_dict'] = mock_dict
    ## write config to disk
    if out:
        if cfg is None:
            try:
                sp, hp, cp = config_Abacus(config_path=Ball.output_dir + 'config.yaml')
                cfg = {'sim_params': sp, 'HOD_params': hp, 'clustering_params': cp}
            except Exception as e:
                print(f"Error loading config: {e}, please provide cfg when out=True.")
                sys.exit(1)
        cfg['HOD_params']['want_rsd'] = want_rsd
        cfg['HOD_params']['want_dv'] = want_dv
        path2dir = find_path(Ball)
        path2cfg = (path2dir) / ('config.yaml')
        save_config(cfg, path2cfg)
    if want_clustering:
        print("Computing clustering...")
        clustering = Ball.compute_multipole(mock_dict, rpbins=Ball.rpbins, pimax=Ball.pimax, sbins=Ball.rpbins[5:], nbins_mu=40, Nthread = nthread) 
        re['clustering'] = clustering
        if out:
            print("Saving clustering to disk...")     
            path2cluster = (path2dir) / ('clustering.npy')
            np.save(path2cluster, clustering)
            print("Save clustering to:", path2cluster)

    return re

def main(nthread, path2config, config_by_file=False):
    ## configure AbacusHOD
    if config_by_file:
        config = None
    else:
        config = CONFIG
    sim_params, HOD_params, clustering_params = config_Abacus(config=config,config_path=path2config)
    
    ## generate AbacusHOD object
    ball_profiles = AbacusHOD(sim_params, HOD_params, clustering_params)
    
    ## compute mock and clustering
    cfg = {'sim_params': sim_params, 'HOD_params': HOD_params, 'clustering_params': clustering_params}
    results=compute_all(ball_profiles, out=True, cfg=cfg, want_clustering=True, nthread=nthread, verbose=True)
    
    print("Finished generating AbacusHOD mock and computing clustering.")

class ArgParseFormatter(
    argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
):
    pass

def parse_args():
    # parsing arguments
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=ArgParseFormatter
    )
    parser.add_argument('-n', '--nthread', type=int, default=16, help='number of threads')
    parser.add_argument('-c', '--config', type=str, default=None, help='config path')
    parser.add_argument('-t', '--template', action='store_true', help='print template config and exit')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    nthread = args.nthread
    path2config = args.config
    template = args.template
    if template:
        config_Abacus(config=None, config_path=None)
        exit(0)
    if path2config is not None:
        config_by_file = True
    else:
        config_by_file = False

    main(nthread, path2config, config_by_file=config_by_file)
