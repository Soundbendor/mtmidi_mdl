import sys,os,time,argparse,copy,types
import torch
from torch import nn
import jukemirlib as jml
import util.util_main as UMN
import util.util_constants as UC
import util.util_hf as UHF
from dataclasses import dataclass
import librosa as lr
from librosa import feature as lrf
from transformers import AutoProcessor, AutoModel, Wav2Vec2FeatureExtractor, MusicgenForConditionalGeneration
from typing import TYPE_CHECKING, Any, Optional, Union
import numpy as np
import torch
import random
from distutils.util import strtobool

dur = 4.0

# https://huggingface.co/m-a-p/MERT-v1-95M
# https://huggingface.co/m-a-p/MERT-v1-330M
# https://huggingface.co/facebook/wav2vec2-large
# https://huggingface.co/facebook/wav2vec2-base
# https://huggingface.co/docs/transformers/main/model_doc/wav2vec2
### porting old code from mtmidi
def get_baseline_features(audio, sr=22050, feat_type="concat"):
    feat = []
    if feat_type == "baseline-mel" or feat_type == "baseline-concat":
        # mel spectrogram
        # https://librosa.org/doc/latest/generated/librosa.feature.melspectrogram.html
        cur_mel = lrf.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length = 512)
        # returns (N=1, n_mels, t)
        feat.append(cur_mel)
    if feat_type == "baseline-chroma" or feat_type == "baseline-concat":
        # constant q chromagram
        # https://librosa.org/doc/0.10.2/generated/librosa.feature.chroma_cqt.html#librosa.feature.chroma_cqt
        # default fmin = 32.7
        # default norm = infinity norm normalization
        # default 36 bins per octave
        cur_chroma = lrf.chroma_cqt(y=audio, sr=sr, hop_length=512)
        # returns (N=1, n_chroma, t)
        feat.append(cur_chroma)
    if feat_type == "baseline-mfcc" or feat_type == "baseline-concat":
        # mfcc
        # https://librosa.org/doc/0.10.2/generated/librosa.feature.mfcc.html#librosa.feature.mfcc
        # default 20 mfccs
        # default orthonormal dct basis
        cur_mfcc = lrf.mfcc(y=audio, sr = sr, n_mfcc = 20)
        # returns (N=1, n_mfcc, t)
        feat.append(cur_mfcc)
    ft_vec = None
    for ft_idx,ft in enumerate(feat):
        # as in the original codebase, do 0,1,2-order diff across time dimension
        # and then take mean and std dev across time dimension
        # note that 0 order diff is just the same array
        for diff_n in range(3):
            cur_diff = np.diff(ft, n=diff_n, axis=1)
            cur_mean = np.mean(cur_diff, axis=1)
            cur_std = np.std(cur_diff, axis=1)
            cur = np.concatenate((cur_mean, cur_std))
            if diff_n == 0 and ft_idx == 0:
                ft_vec = copy.deepcopy(cur)
            else:
                ft_vec = np.concatenate([ft_vec, copy.deepcopy(cur)])
    # make it a 1 x cur_dim vector for consistency (i think)
    if len(ft_vec.shape) < 2:
        ft_vec = np.expand_dims(ft_vec,axis=0)
    return ft_vec

# 1-indexed
def get_jukebox_layer_embeddings(fpath=None, audio = None, layers=list(range(1,73))):
    reps = None
    if fpath != None:
        acts = jml.extract(fpath=fpath, layers=layers, duration=dur, meanpool=True, downsample_target_rate=jb_dsamp_rate, downsample_method=None)
    else:
        acts = jml.extract(audio=audio, layers=layers, duration=dur, meanpool=True, downsample_target_rate=jb_dsamp_rate, downsample_method=None)
    jml.lib.empty_cache()
    return np.array([acts[i] for i in layers])



def get_print_name(dataset, model_size, is_csv = False, normalize = True, timestamp = 0):
    base_fname = f'{dataset}_musicgen-{model_size}-{timestamp}'
    if normalize == True:
        base_fname = f'{dataset}_musicgen-{model_size}_norm-{timestamp}'
    ret = None
    if is_csv == False:
        ret = f'{base_fname}.log'
    else:
        ret = f'{base_fname}.csv'
    return ret

def path_handler(in_filepath, using_hf=False, model_sr = 44100, dur = 4., normalize = True, out_ext = 'dat', logfile_handle=None):
    out_fname = None
    audio = None
    out_fname = None
    fbasename = None
    fold_num = -1 
    if using_hf == False:
        print(f'loading {in_filepath}', file=logfile_handle)
        fbasename = UMN.get_basename(in_filepath, with_ext = False)
        fold_num = UMN.get_fold_num_from_filepath(in_filepath)
        out_fname = f'{fbasename}.{out_ext}'
        # don't need to load audio if jukebox
        audio = UMN.load_wav(in_filepath, dur = dur, normalize = normalize, sr = model_sr)
    else:
        hf_path = in_filepath['audio']['path']
        print(f"loading {hf_path}", file=lf)
        out_fname = UMN.ext_replace(hf_path, new_ext=out_ext)
        fbasename = UMN.ext_replace(hf_path, new_ext='')
        audio = UHF.get_from_entry_syntheory_audio(in_filepath, mono=True, normalize =normalize, dur = dur, sr=model_sr)
    return {'in_fpath': in_filepath, 'out_fname': out_fname, 'audio': audio, 'fname': fbasename, 'fold_num': fold_num}

def get_musicgen_lm_acts(model, proc, audio, text="", meanpool = True, model_sr = 32000, device = 'cpu'):
    procd = proc(audio = audio, text = text, sampling_rate = model_sr, padding=True, return_tensors = 'pt')
    procd.to(device)
    outputs = None
    with torch.no_grad():
        outputs = model(**procd, output_attentions=False, output_hidden_states=True)
    dhs = None
    
    #dat = None

    # hidden
    # outputs is a tuple of tensors with  shape (batch_size, seqlen, dimension) with 1 per layer
    # torch stack makes it so we have (num_layers, batch_size, seqlen, dimension)
    # then we average over seqlen in the meanpool case
    # then squeeze to get rid of the 1 dim (if batch_size == 1)
    # final shape is (num_layers, batch_size, dim)  (or (num_layers, dim) if bs=1)
    
    # attentions
    # outputs is a tuple of tensors with  shape (batch_size, num_heads, seqlen, seqlen) with 1 per layer
    # torch stack makes it so we have (num_layers, batch_size, num_heads, seqlen, sequlen)
    # then we average over seqlens in the meanpool case
    # then squeeze to get rid of the 1 dim (if batch_size == 1)
    # final shape is (num_layers, batch_size, num_heads) (or (num_layers, num_heads) if bs = 1)

    if meanpool == True:
        dhs = torch.stack(outputs.decoder_hidden_states).mean(axis=2).squeeze()
        #dat = torch.stack(outputs.decoder_attentions).mean(axis=(3,4)).squeeze()
    else:
        dhs = torch.stack(outputs.decoder_hidden_states).squeeze()
        #dat = torch.stack(outputs.decoder_attentions).squeeze()
    return dhs.detach().cpu().numpy()


def get_mert_w2v2_acts(model, proc, audio, meanpool = True, model_sr = 24000, device = 'cpu'):
    procd = proc(audio, sampling_rate = model_sr, padding=True, return_tensors = 'pt')
    procd.to(device)
    outputs = None
    with torch.no_grad():
        outputs = model(**procd, output_attentions=False, output_hidden_states=True)
    dhs = None
    
    #dat = None

    # hidden
    # outputs is a tuple of tensors with  shape (batch_size, seqlen, dimension) with 1 per layer
    # torch stack makes it so we have (num_layers, batch_size, seqlen, dimension)
    # then we average over seqlen in the meanpool case
    # then squeeze to get rid of the 1 dim (if batch_size == 1)
    # final shape is (num_layers, batch_size, dim)  (or (num_layers, dim) if bs=1)
    
    # attentions
    # outputs is a tuple of tensors with  shape (batch_size, num_heads, seqlen, seqlen) with 1 per layer
    # torch stack makes it so we have (num_layers, batch_size, num_heads, seqlen, sequlen)
    # then we average over seqlens in the meanpool case
    # then squeeze to get rid of the 1 dim (if batch_size == 1)
    # final shape is (num_layers, batch_size, num_heads) (or (num_layers, num_heads) if bs = 1)

    if meanpool == True:
        dhs = torch.stack(outputs.hidden_states).mean(axis=2).squeeze()
        #dat = torch.stack(outputs.decoder_attentions).mean(axis=(3,4)).squeeze()
    else:
        dhs = torch.stack(outputs.hidden_states).squeeze()
        #dat = torch.stack(outputs.decoder_attentions).squeeze()
    return dhs.detach().cpu().numpy()

def get_musicgen_encoder_embeddings(model, proc, audio, meanpool = True, model_sr = 32000, device='cpu'):
    procd = proc(audio = audio, sampling_rate = model_sr, padding=True, return_tensors = 'pt')
    procd.to(device)
    enc = model.get_audio_encoder()
    out = procd['input_values']
    
    # iterating through layers as in original syntheory codebase
    # https://github.com/brown-palm/syntheory/blob/main/embeddings/models.py
    for layer in enc.encoder.layers:
        out = layer(out)

    # output shape, (1, 128, 200), where 200 are the timesteps
    # so average across timesteps for max pooling


    if meanpool == True:
        # gives shape (128)
        out = torch.mean(out,axis=2).squeeze()
    else:
        # still need to squeeze
        # gives shape (128, 200)
        out = out.squeeze()
    return out.detach().cpu().numpy()


def get_acts(model_size, cur_dataset, normalize = True, dur = 4., use_64bit = True, logfile_handle=None, recfile_handle = None, memmap = True, pickup = False, fold_num = -1, from_dir = "", to_dir = ""):
    
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
    if 'musicgen' in model_size:
        proc = AutoProcessor.from_pretrained(model_str)
        model = MusicgenForConditionalGeneration.from_pretrained(model_str, device_map=device)
        model_sr = model.config.audio_encoder.sampling_rate
    elif 'MERT' in model_size:
        # MERT is not in transformers library, uses modeling_MERT.py
        # https://huggingface.co/docs/transformers/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor
        # default is 16k, trust_remote_code auto sets it to 24k, hopefully
        proc = Wav2Vec2FeatureExtractor.from_pretrained(model_str, do_normalize = False, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_str, trust_remote_code = True)
        model_sr = proc.sampling_rate
    elif 'wav2vec2' in model_size:
        # https://huggingface.co/docs/transformers/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor
        proc = Wav2Vec2FeatureExtractor.from_pretrained(model_str, do_normalize = False)
        model = AutoModel.from_pretrained(model_str)
        model_sr = proc.sampling_rate

    elif 'jukebox' == model_size:
        jml.setup_models(cache_dir='/nfs/guille/eecs_research/soundbendor/kwand/jukemirlib')


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
        fdict = path_handler(fpath, model_sr = model_sr, normalize = normalize, dur = dur,using_hf = using_hf, logfile_handle=logfile_handle, out_ext = out_ext)
        #outpath = os.path.join(out_dir, outname)
        out_fname = fdict['out_fname']
        in_fpath = fdict['in_fpath']
        audio_ipt = fdict['audio']
        fold_num = fdict['fold_num']
        # store by model_size (and fold_num if not using_hf)
        emb_file = None
        rep_arr = None
        if memmap == True:
            emb_file = UMN.get_acts_file(model_size, dataset=cur_dataset, fname=out_fname, use_64bit = use_64bit, write=True, use_shape = None, other_projdir = to_dir, fold_num = fold_num)
        if 'musicgen' in model_size and model_size != 'musicgen-audio':
            print(f'--- extracting musicgen_lm for {fpath} ---', file=logfile_handle)
            rep_arr =  get_musicgen_lm_acts(model, proc, audio_ipt, text="", meanpool = True, model_sr = model_sr, device=device)
        elif 'MERT' in model_size or 'wav2vec' in model_size:
            print(f'--- extracting mert/w2v2 for {fpath} ---', file=logfile_handle)
            rep_arr =  get_mert_w2v2_acts(model, proc, audio_ipt, meanpool = True, model_sr = model_sr, device=device)
        elif 'musicgen-audio' == model_size:
            rep_arr = get_musicgen_encoder_embeddings(model, proc, audio_ipt, meanpool = True, model_sr = model_sr, device=device)
        elif 'baseline' in model_size:
            rep_arr = get_baseline_features(audio, sr=sr, feat_type=model_size)
        elif model_size == 'jukebox':
            print(f'--- extracting jukebox for {f} with {layers_per} layers at a time ---', file=logfile_handle)
            # note that layers are 1-indexed in jukebox
            # so let's 0-idx and then add 1 when feeding into jukebox fn
            layer_gen = (list(range(l, min(um.model_num_layers['jukebox'], l + layers_per))) for l in range(0,um.model_num_layers['jukebox'], layers_per))
            has_last_layer = False
            if layer_num > 0:
                # 0-idx from 1-idxed argt
                layer_gen = ([l-1] for l in [layer_num])
            for layer_arr in layer_gen:
                # 1-idx for passing into fn
                j_idx = [l+1 for l in layer_arr]
                has_last_layer = um.model_num_layers['jukebox'] in j_idx
                print(f'extracting layers {j_idx}', file=logfile_handle)
                rep_arr = get_jukebox_layer_embeddings(fpath=fpath, audio = audio, layers=j_idx)
                emb_file[layer_arr,:] = rep_arr
                emb_file.flush()

        if model_size != 'jukebox':
            if memmap == True:
                emb_file[:,:] = rep_arr
                emb_file.flush()
            else:
                UMN.save_npy(rep_arr, out_fname, model_size, dataset=cur_dataset, other_projdir = to_dir)
        fname = fdict['fname']
        print(f'{fname},1', file=recfile_handle)





if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-ub", "--use_64bit", type=strtobool, default=False, help="use 64-bit")
    parser.add_argument("-ds", "--dataset", type=str, default="polyrhythms", help="dataset")
    parser.add_argument("-ms", "--model_size", type=str, default="musicgen-small", help="musicgen-small, musicgen-medium, or musicgen-large")
    parser.add_argument("-l", "--layer_num", type=int, default=-1, help="1-indexed layer num (all if < 0, for jukebox)")
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
    model_size = args.model_size
    dataset = args.dataset
    debug = args.debug
    pickup = args.pickup
    to_share = args.to_share
    from_share = args.from_share
    fold_num = args.fold_num
    # exit if not a "real" dataset
    logdir = UMN.by_projpath(subpath='log', make_dir = True)
    timestamp = int(time.time() * 1000)

    from_dir = ""
    to_dir = ""
    if args.from_share == True:
        from_dir = os.path.join(UC.SHARE_PATH, 'syntheory_plus')
    if args.to_share == True:
        to_dir = os.path.join(UC.SHARE_PATH, 'mtmidi_prb')
    # miscellaneous logs
    log_fname = get_print_name(dataset, model_size, is_csv = False, normalize = normalize, timestamp = timestamp)
    rec_fname = get_print_name(dataset, model_size, is_csv = True, normalize = normalize, timestamp = timestamp)
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
        get_acts(model_size, dataset, normalize = normalize, dur = dur, use_64bit = use_64bit, logfile_handle=lf, recfile_handle=rf, memmap = memmap, pickup = pickup, fold_num = fold_num, from_dir = from_dir, to_dir = to_dir)
        lf.close()
        rf.close()
