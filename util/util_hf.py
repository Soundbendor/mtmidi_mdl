import librosa
import datasets as HFDS
import numpy as np


# audio files are now TorchCodec AudioDecoder
# https://meta-pytorch.org/torchcodec/stable/generated/torchcodec.decoders.AudioDecoder.html

# which has AudioStreamMetadata as the member .metadata
# https://meta-pytorch.org/torchcodec/stable/generated/torchcodec.decoders.AudioStreamMetadata.html#torchcodec.decoders.AudioStreamMetadata

# which is more complicated to decode
# https://meta-pytorch.org/torchcodec/stable/generated_examples/decoding/audio_decoding.html

def get_from_entry_path(cur_entry):
    #hf_path = cur_entry['audio']['path'] #old way
    hf_path = cur_entry['audio'].metadata.path # new way
    return hf_path

def load_syntheory_train_dataset(ds_name, streaming = True):
    cur_ds =  HFDS.load_dataset("meganwei/syntheory", ds_name, split = 'train', streaming = streaming)
    return cur_ds


def get_from_entry_syntheory_audio(cur_entry, mono=True, normalize =True, dur = 4.0, sr = 32000):
    #cur_aud = train_ds[idx]['audio']
    cur_aud = cur_entry['audio']
    #cur_sr = cur_aud['sampling_rate']
    cur_sr = cur_aud.metadata.sample_rate
    cur_samp = cur_aud.get_all_samples()
    cur_arr = None
    if cur_aud['array'].shape[0] > 1:
        cur_arr = np.mean(cur_samp.data.numpy(), axis=0)
    else:
        cur_arr = cur_samp.data.numpy().flatten()
    if cur_sr != sr:
        cur_arr = librosa.resample(cur_arr, orig_sr=cur_sr, target_sr=sr)
    if normalize == True:
        cur_arr = librosa.util.normalize(cur_arr)
    want_samp = int(sr * dur)
    return cur_arr[:want_samp]
