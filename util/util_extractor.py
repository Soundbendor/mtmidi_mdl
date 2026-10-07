from . import util_constants as UC
from . import util_main as UMN
from . import util_hf as UHF

def parse_seqtype_overrides(model_size, meanpool_override = False, full_seq_override = False):
    meanpool = False
    last_token = True
    if model_size in UC.MEANPOOL_MODELS or meanpool_override == True:
        meanpool = True
        last_token = False
    elif full_seq_override == True:
        meanpool = False
        last_token = False

    return meanpool, last_token



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

def path_handler(in_filepath, using_hf=False, model_sr = 44100, dur = UC.WAV_DUR, normalize = True, out_ext = 'dat', meanpool = False, last_token = True,logfile_handle=None):
    out_fname = None
    audio = None
    out_fname = None
    fbasename = None
    fold_num = -1 
    token_type = None
    
   
    
    if using_hf == False:
        print(f'loading {in_filepath}', file=logfile_handle)
        fbasename = UMN.get_basename(in_filepath, with_ext = False)
        fold_num = UMN.get_fold_num_from_filepath(in_filepath)
        out_fname = UMN.add_fname_suffix(fbasename, meanpool = meanpool, last_token = last_token, out_ext = out_ext) 
        # don't need to load audio if jukebox
        audio = UMN.load_wav(in_filepath, dur = dur, normalize = normalize, sr = model_sr)
    else:
        hf_path = UHF.get_from_entry_path(in_filepath) 
        print(f"loading {hf_path}", file=lf)
        #out_fname = UMN.ext_replace(hf_path, new_ext=out_ext)
        fbasename = UMN.ext_replace(hf_path, new_ext='')
        out_fname = UMN.add_fname_suffix(fbasename, meanpool = meanpool, last_token = last_token, out_ext = out_ext) 
        audio = UHF.get_from_entry_syntheory_audio(in_filepath, mono=True, normalize =normalize, dur = dur, sr=model_sr)
    return {'in_fpath': in_filepath, 'out_fname': out_fname, 'audio': audio, 'fname': fbasename, 'fold_num': fold_num}


