import argparse
from pathlib import Path
from util import util_main as UMN
from util import util_constants as UC

from distutils.util import strtobool
import os, time, subprocess
from concurrent.futures import ThreadPoolExecutor

def run_sbatch_script(script_path):
    print(f"Running {script_path}")
    subprocess.run(["sbatch", "-W", f"{script_path}"])

if __name__ == "__main__":
    #### arg parsing
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-ds", "--datasets", nargs="+", type=str, default=["polyrhythms", "secondary_dominants", "dynamics", "seventh_chords", "mode_mixture"], help="datasets")
    parser.add_argument("-nd", "--num_days", type=int, default=1, help="number of days")
    parser.add_argument("-pt", "--partition", type=str, default="preempt", help="partition to run on")
    parser.add_argument("-ms", "--model_sizes", nargs="+", type=str, default=["MERT-v1-95M", "MERT-v1-330M"], help="musicgen-small/musicgen-medium/musicgen-large/jukebox/baseline-chroma/baseline-concat/baseline-mel/baseline-mfcc")
    parser.add_argument("-l", "--layer_num", type=int, default=-1, help="1-indexed layer num (all if < 0, for jukebox)")
    parser.add_argument("-fsh", "--from_share", type=strtobool, default=True, help="load from share partition")
    parser.add_argument("-tsh", "--to_share", type=strtobool, default=False, help="save to share partition")
    parser.add_argument("-m", "--memmap", type=strtobool, default=True, help="save as memmap, else save as npy")
    parser.add_argument("-ub", "--use_64bit", type=strtobool, default=False, help="use 64 bit")
    parser.add_argument("-db", "--debug", type=strtobool, default=False, help="debug mode")
    parser.add_argument("-p", "--pickup", type=strtobool, default=False, help="pickup where script left off")
    parser.add_argument("-fn", "--fold_num", type=int, default=0, help="fold number to extract (-1 for no folds, 0 for all folds, else specific fold)")
    parser.add_argument("-ram", "--ram_mem", type=int, default=40, help="ram in gigs")
    parser.add_argument("-gpu", "--gpus", type=int, default=1, help="num of gpus to use")
    parser.add_argument("-sj", "--slurm_job", type=int, default=0, help="slurm job")
    parser.add_argument("-nj", "--num_jobs", type=int, default=1, help="number of jobs to run at a time")
    


    args = parser.parse_args()
    scripts = [] 
    project_root = Path(__file__).resolve().parent.parent
    cur_dir = Path(__file__).resolve().parent
    sh_dir = os.path.join(cur_dir, 'sh')
    if os.path.exists(sh_dir) == False:
        os.makedirs(sh_dir)

    py_path = os.path.join(project_root, 'extractor.py')


    script_idx = 0
    start_time = str(int(time.time() * 1000))

    for dataset in args.datasets:
        cur_normalize = True
        if 'dynamics' == dataset:
            cur_normalize = False
        ds_short = UC.DATASET_SHORT[dataset]
        for model_size in args.model_sizes:
            size_short = UC.MODEL_SIZES_SHORT[model_size]         
            job_str = f'ex-{ds_short}-{size_short}'
            slurm_strarr1 = ["#!/bin/bash"]
            slurm_strarr2 = [f"#SBATCH -p {args.partition}"]
            if args.partition != 'preempt':
                if args.partition != 'soundbendor':
                    slurm_strarr2 = ['#SBATCH -A eecs', f"#SBATCH -p {args.partition}"]
                else:
                    slurm_strarr2 = ['#SBATCH -A soundbendor', f"#SBATCH -p {args.partition}"]
            slurm_strarr3 = [f"#SBATCH --mem={args.ram_mem}G", f"#SBATCH --gres=gpu:{args.gpus}", f"#SBATCH -t {args.num_days}-00:00:00", f"#SBATCH --job-name={job_str}", "#SBATCH --export=ALL", f"#SBATCH --output=/nfs/guille/eecs_research/soundbendor/kwand/out_mtmidi_prb/{job_str}-%j.out", ""]
            slurm_strarr = slurm_strarr1 + slurm_strarr2 + slurm_strarr3
            p_str = f"python {py_path} -ds {dataset} -ms {model_size} -fsh {args.from_share} -tsh {args.to_share} -ub {args.use_64bit} -n {cur_normalize} -l {args.layer_num} -m {args.memmap} -db {args.debug} -p {args.pickup} -fn {args.fold_num}" 
            slurm_strarr.append(p_str)
            script_fname = f"{start_time}_{job_str}.sh"
            script_idx += 1
            script_path = os.path.join(sh_dir, script_fname)
            script_str = "\n".join(slurm_strarr)
            print(f"===== {dataset} | {model_size} | normalized: {cur_normalize} =====")
            print(f"Creating {script_fname}")
            with open(script_path, 'w') as f:
                f.write(script_str)
            subprocess.run(["chmod", "u+x", f"{script_path}"])
            scripts.append(script_path)

    with ThreadPoolExecutor(max_workers=args.num_jobs) as executor:
        executor.map(run_sbatch_script, scripts)




