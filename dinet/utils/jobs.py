import os
import shutil
import yaml

from dinet.utils import util
osp = os.path

from dinet import ANALYSIS_DIR, RESULTS_DIR

def rename_job(job: str, new_name: str):
    os.rename(osp.join(RESULTS_DIR, job), osp.join(RESULTS_DIR, new_name))
    for folder in util.glob2(ANALYSIS_DIR, "*", job):
        os.rename(folder, folder.replace(job, new_name))

def delete_job(job: str):
    shutil.rmtree(osp.join(RESULTS_DIR, job))
    for folder in util.glob2(ANALYSIS_DIR, "*", job):
        shutil.rmtree(folder)

def get_job_args(job: str) -> dict:
    config_path = osp.join(RESULTS_DIR, job, f"config.yaml")
    args = yaml.safe_load(open(config_path, "r"))
    return args

def get_dataset_for_job(job: str):
    return get_job_args(job)["data loading"]["dataset"]

