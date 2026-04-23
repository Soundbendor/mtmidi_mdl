import librosa
import datasets as HFDS
import numpy as np

def load_syntheory_train_dataset(ds_name, streaming = True):
    cur_ds =  HFDS.load_dataset("meganwei/syntheory", ds_name, split = 'train', streaming = streaming)
    return cur_ds


def get_from_entry_syntheory_audio(cur_entry, mono=True, normalize =True, dur = 4.0, sr = 32000):
    #cur_aud = train_ds[idx]['audio']
    cur_aud = cur_entry['audio']
    cur_sr = cur_aud['sampling_rate']
    cur_arr = None
    if cur_aud['array'].shape[0] > 1:
        cur_arr = np.mean(cur_aud['array'], axis=0)
    else:
        cur_arr = cur_aud['array'].flatten()
    if cur_sr != sr:
        cur_arr = librosa.resample(cur_arr, orig_sr=cur_sr, target_sr=sr)
    if normalize == True:
        cur_arr = librosa.util.normalize(cur_arr)
    want_samp = int(sr * dur)
    return cur_arr[:want_samp]
