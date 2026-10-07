import sys,os,time,argparse,copy,types
import torch
from torch import nn
import jukemirlib as jml
import util.util_main as UMN
import util.util_constants as UC
import util.util_hf as UHF
import util.util_extractor as UEX
from dataclasses import dataclass
import librosa as lr
from librosa import feature as lrf
from typing import TYPE_CHECKING, Any, Optional, Union
import numpy as np
import torch
import random
from distutils.util import strtobool


# 1-indexed

# NEED TO IMPLEMENT LAST_TOKEN
# assume it's seqlen, dim but fix later
def get_jukebox_layer_embeddings(fpath=None, audio = None, dur = UC.WAV_DUR, meanpool = False, last_token = True, layers=list(range(1,73))):
    reps = None
    cur_args = {'layers': layers, 'duration': dur, 'meanpool': True, 'downsample_target_rate': UC.JUKEBOX_DOWNSAMP_RATE, 'downsample_method': None}
    if fpath != None:
        cur_args['fpath'] = fpath
    else:
        cur_args['audio'] = audio
    if fpath != None:
        acts = jml.extract(**cur_args)
    else:
        acts = jml.extract(**cur_args)
    jml.lib.empty_cache()
    ret = None
    if meanpool == True or last_token == False:
        ret = np.array([acts[i] for i in layers])
    if last_token == True:
        ret = np.array([acts[i][-1] for i in layers])
    return ret

def get_acts(model_size, cur_dataset, normalize = True, dur = UC.WAV_DUR, use_64bit = True, logfile_handle=None, recfile_handle = None, memmap = True, pickup = False, fold_num = -1, meanpool = False, last_token = True, from_dir = "", to_dir = ""):
    jukebox_layer_arr = list(range(UC.MODEL_NUM_LAYERS['jukebox']))
    using_hf = cur_dataset in UC.SYNTHEORY_DATASETS
    # musicgen stuff
    device = 'cpu'
    num_layers = None
    proc = None
    model = None
    model_sr = None
    text = ""
    wav_path = os.path.join(UMN.by_projpath('wav'), cur_dataset)
    if len(from_dir) > 0:
        wav_path = os.path.join(from_dir, cur_dataset)
    cur_pathlist = None
    out_ext = 'dat'
    if memmap == False:
        out_ext = 'npy'
    if using_hf == True:
        fold_num = -1 # don't care about fold folders
        cur_pathlist = UHF.load_syntheory_train_dataset(cur_dataset)
    else:
        cur_pathlist = UMN.filepath_list(wav_path, fold_num=fold_num, ignore_exts = set(['.csv']))

    device = 'cpu'
    if torch.cuda.is_available() == True:
        device = 'cuda'
        torch.cuda.empty_cache()
        torch.set_default_device(device)
    
    model_str = UMN.get_hf_model_str(model_size) 
    if 'jukebox' == model_size:
        jml.setup_models(cache_dir='/nfs/guille/eecs_research/soundbendor/kwand/jukemirlib')
        model_sr = 44100


    # existing files removing latest (since it may be partially written) and removing extension for each of checking
    existing_name_set = None
    if pickup == True:
        # pass -1 for fold_num to omit fold_num folder since remove_latest_file takes care of it
        _file_dir = UMN.get_model_acts_path(model_size, dataset=cur_dataset, return_relative = False, make_dir = False, other_projdir = to_dir, fold_num=-1)
        existing_files = UMN.remove_latest_file(_file_dir, is_relative = False, fold_num = fold_num)
        existing_name_set = set([UMN.get_basename(_f, with_ext = False) for _f in existing_files])
    for fidx,fpath in enumerate(cur_pathlist):
        if pickup == True:
            cur_name = UMN.get_basename(fpath, with_ext = False)
            if cur_name in existing_name_set:
                continue
        fdict = UEX.path_handler(fpath, model_sr = model_sr, normalize = normalize, dur = dur,using_hf = using_hf, logfile_handle=logfile_handle, meanpool = meanpool, last_token = last_token, out_ext = out_ext)
        #outpath = os.path.join(out_dir, outname)
        out_fname = fdict['out_fname']
        in_fpath = fdict['in_fpath']
        audio_ipt = fdict['audio']
        fold_num = fdict['fold_num']
        # store by model_size (and fold_num if not using_hf)
        rep_arr = None
        if memmap == True:
        print(f'--- extracting jukebox for {fpath} ---', file=logfile_handle)
        # note that layers are 1-indexed in jukebox
        # so let's 0-idx and then add 1 when feeding into jukebox fn

        # 1-idx for passing into fn
        j_idx = [l+1 for l in jukebox_layer_arr]
        print(f'extracting layers {j_idx}', file=logfile_handle)
        rep_arr = get_jukebox_layer_embeddings(fpath=None, audio = audio_ipt, meanpool = meanpool, last_token = last_token, dur = dur, layers=j_idx) 
        if memmap == True:
            cur_seqlen = -1
            if meanpool == False and last_token == False:
                cur_seqlen = rep_arr[0].shape[0]
                seqlen_fname = f'{out_fname}-seqlen.txt'
                #seqlen_folder = os.path.join(UC.SEQLEN_FOLDER, seqlen_fname)
                seqlen_folder = UMN.by_projpath2([UC.SEQLEN_FOLDER, model_size, cur_dataset], make_dir = True)
                with open(os.path.join(seqlen_folder,seqlen_fname), 'w') as sl_file:
                    sl_file.write(str(cur_seqlen))

            emb_file = UMN.get_acts_file(model_size, dataset=cur_dataset, fname=out_fname, use_64bit = use_64bit, write=True, use_shape = None, seqlen = cur_seqlen, meanpool = meanpool, last_token = last_token, other_projdir = to_dir, fold_num = fold_num)
            emb_file[jukebox_layer_arr] = rep_arr
            emb_file.flush()
        else:
            UMN.save_npy(rep_arr, out_fname, model_size, dataset=cur_dataset, other_projdir = to_dir)
        fname = fdict['fname']
        print(f'{fname},1', file=recfile_handle)





if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-ub", "--use_64bit", type=strtobool, default=False, help="use 64-bit")
    parser.add_argument("-ds", "--dataset", type=str, default="polyrhythms", help="dataset")
    parser.add_argument("-l", "--layer_num", type=int, default=-1, help="1-indexed layer num (all if < 0, for jukebox)")
    parser.add_argument("-mpl", "--meanpool", type=strtobool, default=False, help="meanpool over seq len (override for AR models)")
    parser.add_argument("-fsq", "--full_seq", type=strtobool, default=True, help="save full seq (override for both AR and Masked)")
    parser.add_argument("-n", "--normalize", type=strtobool, default=True, help="normalize audio")
    parser.add_argument("-m", "--memmap", type=strtobool, default=True, help="save as memmap, else save as npy")
    parser.add_argument("-db", "--debug", type=strtobool, default=False, help="debug mode")
    parser.add_argument("-p", "--pickup", type=strtobool, default=False, help="pickup where script left off")
    parser.add_argument("-tsh", "--to_share", type=strtobool, default=False, help="save on share partition")
    parser.add_argument("-fsh", "--from_share", type=strtobool, default=False, help="load on share partition")
    parser.add_argument("-fn", "--fold_num", type=int, default=0, help="fold number to extract (-1 for no folds, 0 for all folds, else specific fold)")

    
    args = parser.parse_args()
    use_64bit = args.use_64bit
    lnum = args.layer_num
    memmap = args.memmap
    normalize = args.normalize
    model_size = 'jukebox'
    dataset = args.dataset
    debug = args.debug
    pickup = args.pickup
    to_share = args.to_share
    from_share = args.from_share
    fold_num = args.fold_num
    # exit if not a "real" dataset
    logdir = UMN.by_projpath(subpath='log', make_dir = True)
    timestamp = int(time.time() * 1000)
    
    meanpool, last_token =  UEX.parse_seqtype_overrides(model_size, meanpool_override = args.meanpool, full_seq_override = args.full_seq)

    from_dir = ""
    to_dir = ""
    if args.from_share == True:
        from_dir = os.path.join(UC.SHARE_PATH, 'syntheory_plus')
    if args.to_share == True:
        to_dir = os.path.join(UC.SHARE_PATH, 'mtmidi_mdl')
    # miscellaneous logs
    log_fname = UEX.get_print_name(dataset, model_size, is_csv = False, normalize = normalize, timestamp = timestamp)
    rec_fname = UEX.get_print_name(dataset, model_size, is_csv = True, normalize = normalize, timestamp = timestamp)
    log_fpath = os.path.join(logdir, log_fname)
    rec_fpath = os.path.join(logdir, rec_fname)
    if debug == True:
        exit()
    if (dataset in UC.ALL_DATASETS) == False:
        sys.exit('not a dataset')
    else:
        lf = open(log_fpath, 'a')
        rf = open(rec_fpath, 'w')
        print(f'=== running extraction for {dataset} with {model_size} at {timestamp} ===', file=lf)
        get_acts(model_size, dataset, normalize = normalize, dur = UC.WAV_DUR, use_64bit = use_64bit, logfile_handle=lf, recfile_handle=rf, memmap = memmap, pickup = pickup, fold_num = fold_num, meanpool = meanpool, last_token = last_token, from_dir = from_dir, to_dir = to_dir)
        lf.close()
        rf.close()
