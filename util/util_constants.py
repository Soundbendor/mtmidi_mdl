from pathlib import Path
import os
import numpy as np

# from https://github.com/brown-palm/syntheory/blob/4f222359e750ec55425c12809c1a0358b74fce49/embeddings/models.py#L114
JUKEBOX_DOWNSAMP_RATE = 15
# length of soundfile
WAV_DUR = 4.0
LOG2E = np.log2(np.exp(1.))

NUM_FOLDS = 20
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ACTS_FOLDER = 'acts'
SAMPLER_FOLDER = 'samplers'
CM_FOLDER = 'cm'
CM_NONSTANDARD_FOLDER = 'cm-nstd'
RESULTS_FOLDER = 'res'
RESULTS_NONSTANDARD_FOLDER = 'res-nstd'
MODELS_FOLDER = 'model_models'
DATA_STATS_FOLDER = 'data_stats'
BIASED_PART_RTO_FOLDER = 'b_pr'
UNBIASED_PART_RTO_FOLDER = 'ub_pr'
TWONN_FOLDER = 'twonn'
TWONN_UNUSED_PCT = 0.1
TWONN_SIZE_FOLDER = 'twonn_sz'
BIASED_PART_RTO_NONSTANDARD_FOLDER = 'b_pr-nstd'
UNBIASED_PART_RTO_NONSTANDARD_FOLDER = 'ub_pr-nstd'
RDB_FOLDER = 'rdb'
OPT_DIRECTION = 'minimize'
EARLY_STOPPING_CHECK_INTERVAL = 1
EARLY_STOPPING_PATIENCE = 100
NUM_STEPS = 5
SCHED_PATIENCE = 15
WANDB_PING_TIMEOUT = 10
MEMMAP = True
NUM_EPOCHS = 10000
PATIENCE_DELTA = 10.**(-4)
TRAIN_PCT = 0.92
TEST_SUBPCT = 0.5
NUM_CLASS_THRESH = 3
BATCH_SIZE = 2048
TWONN_BATCH_SIZE = 64
LEARNING_RATE = 10.**(-2)
# no l2 weight decay (set to -2 in original which meant turn off)
DATALOADER_SHUFFLE = True
IS_64BIT = False
SEED = 39
USE_WEIGHTS = False
MLPPROBE_HIDDEN_DIMS = [512]


# for less than 11 classes
CM_FIGSIZE_S = (5,5)
# for 11 classes and up
CM_FIGSIZE_M = (7,7)
# for a lot of classses
CM_FIGSIZE_L = (18,15)

SHARE_PATH = os.path.join(os.sep, 'nfs','hpc', 'share', 'kwand') 
WANDB_PATH = os.path.join(os.sep, 'nfs','guille', 'eecs_research', 'soundbendor', 'kwand', 'wandb') 


EXPR_PRETTY_NAMES = {'mlp': 'MLP', 'linear': 'Linear Layer'}
MODEL_SIZES = ["baseline-concat", "baseline-chroma", "baseline-mfcc", "baseline-mel", "musicgen-audio", "musicgen-small", "musicgen-medium", "musicgen-large", "jukebox"]

EXPR_SHORT = {"mlp": "mlp", "standard_scaler": "sts", 'linear': 'lin'}

MODEL_SIZES_SHORT = {"baseline-concat": "bcat", "baseline-chroma": "bchr", "baseline-mfcc": "bmfcc", "baseline-mel": "bmel", "musicgen-audio": "mga", "musicgen-small": "mgs", "musicgen-medium": "mgm", "musicgen-large": "mgl", "jukebox": "j", "MERT-v1-95M": 'm95', "MERT-v1-330M": 'm330', "wav2vec2-base": "wb", "wav2vec2-large": "wl"}



DATASET_SHORT = {"polyrhythms": "pl",
                 "dynamics": "dyn",
                 "seventh_chords": "ch7",
                 "mode_mixture": "mm",
                 "secondary_dominants": "sd",
                 "tempos": "tpo",
                 "time_signatures": "ts",
                 "chords": "chd",
                 "notes": "not",
                 "scales": "scl",
                 "intervals": "ivl",
                 "simple_progressions": "spg"
                 }

DATASET_PPRINT = {"polyrhythms": "Polyrhythms",
                 "dynamics": "Dynamics",
                 "seventh_chords": "Seventh Chords",
                 "mode_mixture": "Mode Mixture",
                 "secondary_dominants": "Secondary Dominants",
                 "tempos": "Tempos",
                 "time_signatures": "Time Signatures",
                 "chords": "Chords",
                 "notes": "Notes",
                 "scales": "Scales",
                 "intervals": "Intervals",
                 "simple_progressions": "Simple Progressions"
                 }



# https://github.com/huggingface/transformers/blob/80996194bec45b16d4472a099e64b57e049bc6fd/src/transformers/models/musicgen/convert_musicgen_transformers.py#L120
FFN_DIM = {"baseline-concat": 960, "baseline-chroma": 72, "baseline-mfcc": 120, "baseline-mel": 768, "musicgen-audio": 128, "musicgen-small": 1024, "musicgen-medium": 1536, "musicgen-large": 2048, "jukebox": 4800, 'MERT-v1-95M': 768, 'MERT-v1-330M': 1024, 'wav2vec2-base': 768, 'wav2vec2-large': 1024}

# initial embeddings for mgs/mgm/mgl/mert/wav2vec2
MODEL_NUM_LAYERS = {"baseline-concat": 1, "baseline-chroma": 1, "baseline-mfcc": 1, "baseline-mel": 1, "musicgen-audio": 1, "musicgen-small": 25, "musicgen-medium": 49, "musicgen-large": 49, "jukebox": 72, 'MERT-v1-95M': 13, 'MERT-v1-330M': 25, 'wav2vec2-base': 13, 'wav2vec2-large': 25}

SINGLE_LAYER_MODELS = set(["baseline-concat", "baseline-chroma", "baseline-mfcc", "baseline-mel", "musicgen-audio"])
### porting a lot of old code from mtmidi

MUSICGEN_SR = 32000
JUKEBOX_SR = 44100
MERT_SR = 24000
W2V2_SR = 16000
# same as mtmidi
# but secondary_dominant -> secondary_dominants
# modemix_chordprog -> mode_mixture
# chords7 -> seventh_chords
SYNTHEORY_PLUS_DATASETS = set(['polyrhythms', 'dynamics', 'seventh_chords', 'secondary_dominants', 'mode_mixture'])

SYNTHEORY_DATASETS = set(['tempos', 'time_signatures', 'chords', 'notes', 'scales', 'intervals', 'simple_progressions'])
CHORDPROG_DATASETS = set(['secondary_dominant', 'modemix_chordprog', 'simple_progressions'])

MODELS = ['baseline-concat', 'baseline-chroma', 'baseline-mfcc', 'baseline-mel', 'musicgen-audio', 'musicgen-small', 'musicgen-medium', 'musicgen-large', 'jukebox', 'MERT-v1-95M', 'MERT-v1-330M', 'wav2vec2-base', 'wav2vec2-large']


CLS_PPRINT = {'f1_macro': 'F1 (macro)',
              'f1_micro': 'F1 (micro)',
              'layer_idx': 'Layer Index',
              'loss': 'NLL',
              'loss_base2': 'NLL (base 2)',
              'accuracy_score': 'Accuracy',
              'balanced_accuracy_score': 'Balanced Accuracy'
              }

MODEL_PPRINT = {'baseline-concat': "Concat.",
                 'baseline-chroma': "Chroma",
                 'baseline-mfcc': "MFCC",
                 'baseline-mel': "Mel",
                 'musicgen-audio': "EnCodec",
                 'musicgen-small': "MusicGen-small",
                 'musicgen-medium': "MusicGen-medium",
                 'musicgen-large': "MusicGen-large",
                 'jukebox': "Jukebox",
                 'MERT-v1-95M': "MERT-95M",
                 'MERT-v1-330M': "MERT-330M",
                 'wav2vec2-base': "Wav2Vec2-base",
                 'wav2vec2-large': "Wav2Vec2-large"
                 }

# at least 10 samples per step
DATASET_PREQ_STEPS = {"polyrhythms": 7,
                      "dynamics": 7,
                      "seventh_chords": 7,
                      "mode_mixture": 7,
                      "secondary_dominants": 7,
                      "tempos": -1,
                      "time_signatures": 5,
                      "chords": 7,
                      "notes": 6,
                      "scales": 7,
                      "intervals": 7,
                      "simple_progressions": 7
                 }

DATASET_PREQ_TRAIN_SIZES = {k:(1./np.power(2, np.arange(v+1)))[::-1] for (k,v) in DATASET_PREQ_STEPS.items()}
#datasets that are regression
REG_DATASETS = set(['tempos'])
# datasets to train on middle on
TOM_DATASETS = set(['tempos'])
ALL_DATASETS = SYNTHEORY_DATASETS.union(SYNTHEORY_PLUS_DATASETS)

