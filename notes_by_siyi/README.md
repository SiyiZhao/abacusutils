# Notes

This directory contains notes when using the AbacusUtils library.

Author: Siyi Zhao 

---

## About Abacus 

[Abacus](https://abacussummit.readthedocs.io/en/latest/abacus.html) is a N-body code.
As the notes are mainly about using Abacus Utilities, I will use 'Abacus' as a general term to refer to the products from Abacus.

Products: 
- [AbacusSummit](https://abacussummit.readthedocs.io/en/latest/index.html) is a suite of cosmological N-body simulations run with the Abacus code. It is also named after the computer used to run the code, Summit. The simulations cover a range of cosmological parameters. Most of them are in 2Gpc/h boxes with 6912^3 particles.
- [AbacusPNG](https://arxiv.org/abs/2402.10881) is a suite of cosmological N-body simulations run with the Abacus code, started by the initial condition with local PNG.

The [Abacus Utilities](https://abacusutils.readthedocs.io/en/latest/) is a set of Python tools to read and analyze AbacusSummit data products.

## CompaSO halos

Abacus uses CompaSO halo finder by default.

I have an example script `CompaSO_halo.ipynb` to show how to read CompaSO halo catalogs and plot halo mass functions.

## AbacusHOD 

run Halo Occupation Distribution (HOD) models on Abacus halo catalogs to produce mock galaxy catalogs.

I have an example script `AbacusHODmock.py` to show how to use AbacusHOD to make a mock catalog, assuming:
- this repository is cloned under the `~/lib` directory;
- the prepared data has already generated.

### Prepare data

AbacusHOD needs prepare the data of halos and particles from Abacus simulations first. 

If you want to use particles to assign satellites, run `python -m abacusnbody.hod.prepare_sim --path2config $config`.
Elif you want to use profiles to assign satellites, run `python -m abacusnbody.hod.prepare_sim_profiles --path2config $config`.
There are also other options, to run the preparation script with different simulation boxes, redshifts, random seeds, etc. Check the help message by `python -m abacusnbody.hod.prepare_sim --help` for more details. However, it would overwrite the simulation name and redshifts in the config file.

(Following notes refer to the code of `abacusnbody/hod/prepare_sim.py`, there could be some differences in `abacusnbody/hod/prepare_sim_profiles.py`, if I notice it, I will add them.)

This step depends on following settings:
- 'sim_params': 
    - to find inputs: 'sim_dir', 'sim_name', 'z_mock';
    - 'cleaned_halos': whether to use cleaned halo catalogs;
    - 'halo_lc' if set: use lightcone halos;
    - the output directory will be under 'subsample_dir';
- 'MT': mass threshold, fixed to True; ([Commit 04d844f](https://github.com/SiyiZhao/abacusutils/commit/04d844f6e195cdb440c322d80550ad61b457a663) turn it to True, it's False in parent repo.)
- 'prepare_sim': 
    - 'Nthread_per_load': default 'auto', would calculate with the CPUs number and 'Nparallel_load';
    - 'Nparallel_load': max workers in parallel, it use `concurrent.futures.ProcessPoolExecutor` to work in parallel;
- 'HOD_params'
