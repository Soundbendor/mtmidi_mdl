import argparse
from pathlib import Path
from util import util_main as UMN
from util import util_constants as UC

from distutils.util import strtobool
import os, time, subprocess
from concurrent.futures import ThreadPoolExecutor

script_idxs = {}
script_idx = 0

def run_sbatch_script(script_path):
    cur_idx = script_idxs[script_path]
    print(f"Running {script_path} ({cur_idx}/{script_idx})")
    subprocess.run(["sbatch", "-W", f"{script_path}"])

if __name__ == "__main__":
    #### arg parsing
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-ds", "--datasets", nargs="+", type=str, default=["polyrhythms", "secondary_dominants", "dynamics", "seventh_chords", "mode_mixture", "time_signatures", "notes", "scales", "intervals", "simple_progressions", "chords"], help="datasets")
    parser.add_argument("-nd", "--num_days", type=int, default=1, help="number of days")
    parser.add_argument("-pt", "--partition", type=str, default="preempt", help="partition to run on")
    parser.add_argument("-ms", "--model_sizes", nargs="+", type=str, default=["MERT-v1-95M", "MERT-v1-330M", "wav2vec2-base", "wav2vec2-large", "musicgen-small", "musicgen-medium", "musicgen-large"], help="musicgen-small/musicgen-medium/musicgen-large/jukebox/MERT-v1-95M/MERT-v1-330M/wav2vec2-base/wav2vec2-large")
    parser.add_argument("-et", "--expr_type", type=str, default="mlp", help="experiment type")
    parser.add_argument("-zd", "--zero_dist", type=strtobool, default=False, help="find zero dist embeddings")
    parser.add_argument("-wdb", "--use_wandb", type=strtobool, default=True, help="sync to wandb")
    parser.add_argument("-cd", "--use_cuda", type=strtobool, default=True, help="use cuda")
    parser.add_argument("-bpr", "--biased_part_rto", type=strtobool, default=False, help="calculate biased participation ratio")
    parser.add_argument("-twn", "--twonn", type=strtobool, default=False, help="calculate twonn")
    parser.add_argument("-upr", "--unbiased_part_rto", type=strtobool, default=False, help="calculate biased participation ratio")
    parser.add_argument("-nstd", "--nonstandard", type=strtobool, default=False, help="do not divide data by feature-wise standard deviation")

    parser.add_argument("-st", "--stats", type=strtobool, default=False, help="calculate stats")
    parser.add_argument("-rs", "--restart_study", type=strtobool, default=False, help="force restart of optuna study")
    parser.add_argument("-sh", "--from_share", type=strtobool, default=True, help="load from share partition")
    parser.add_argument("-sf", "--suffix", type=int, default=1, help="suffix")
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

    py_path = os.path.join(project_root, 'probing.py')


    start_time = str(int(time.time() * 1000))
    
    using_part_rto = args.unbiased_part_rto == True or args.biased_part_rto == True 
    expr_short = UC.EXPR_SHORT[args.expr_type] 
    for dataset in args.datasets:
        ds_short = UC.DATASET_SHORT[dataset]
        for model_size in args.model_sizes:
            size_short = UC.MODEL_SIZES_SHORT[model_size]         
            job_str = f'{expr_short}-{ds_short}-{size_short}'
            if args.stats == True and using_part_rto == False:
                job_str = f'sts_{job_str}'
            if args.stats == False and using_part_rto == True:
                job_str = f'pto_{job_str}'
            if args.stats == False and using_part_rto == False:
                if args.twonn == False and args.zero_dist == False:
                    job_str = f'mdl_{job_str}'
                elif args.zero_dist == True:
                    job_str = f'zd_{job_str}'
                else:
                    job_str = f'twn_{job_str}'
            slurm_strarr1 = ["#!/bin/bash"]
            slurm_strarr2 = [f"#SBATCH -p {args.partition}"]
            if args.partition != 'preempt':
                if args.partition != 'soundbendor':
                    slurm_strarr2 = ['#SBATCH -A eecs', f"#SBATCH -p {args.partition}"]
                else:
                    slurm_strarr2 = ['#SBATCH -A soundbendor', f"#SBATCH -p {args.partition}"]
            slurm_strarr3 = [f"#SBATCH --mem={args.ram_mem}G", f"#SBATCH --gres=gpu:{args.gpus}", f"#SBATCH -t {args.num_days}-00:00:00", f"#SBATCH --job-name={job_str}", "#SBATCH --export=ALL", f"#SBATCH --output=/nfs/guille/eecs_research/soundbendor/kwand/out_mtmidi_mdl/{job_str}-%j.out", ""]
            slurm_strarr = slurm_strarr1 + slurm_strarr2 + slurm_strarr3
            p_str = f"python {py_path}  -st {args.stats} -upr {args.unbiased_part_rto}  -bpr {args.biased_part_rto} -twn {args.twonn} -nstd {args.nonstandard} -ds {dataset} -et {args.expr_type} -ms {model_size} -sh {args.from_share} -wdb {args.use_wandb} -cd {args.use_cuda} -sf {args.suffix} -zd {args.zero_dist}" 
            slurm_strarr.append(p_str)
            script_fname = f"{start_time}_{job_str}.sh"
            script_path = os.path.join(sh_dir, script_fname)
            script_str = "\n".join(slurm_strarr)
            script_idx += 1
            print(f"===== {args.expr_type} | {dataset} | {model_size} | STATS: {args.stats} | UPR: {args.unbiased_part_rto} | BPR: {args.biased_part_rto} | TWONN: {args.twonn} =====")
            print(f"Creating {script_fname}")
            with open(script_path, 'w') as f:
                f.write(script_str)
            subprocess.run(["chmod", "u+x", f"{script_path}"])
            scripts.append(script_path)
            script_idxs[script_path] = script_idx

    with ThreadPoolExecutor(max_workers=args.num_jobs) as executor:
        executor.map(run_sbatch_script, scripts)




