from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ACTS_FOLDER = 'acts'
SAMPLER_FOLDER = 'samplers'
SCALERS_FOLDER = 'scalers'
SCALERS_DOC_FOLDER = 'scalers_doc'
CM_FOLDER = 'cm'
RESULTS_FOLDER = 'res'
RESULTS_TRAIN_FOLDER = 'res_train'
MODELS_FOLDER = 'model_models'
DATA_STATS_FOLDER = 'data_stats'
PART_RTO_FOLDER = 'part_rto'
RDB_FOLDER = 'rdb'
NUM_FOLDS = 20
EARLY_STOPPING_CHECK_INTERVAL = 1
EARLY_STOPPING_BOREDOM = 500
SCHED_PATIENCE = 15
GRAD_MAX_NORM = 0.01
MEMMAP = True
WARMUP_BATCH_COUNT = 50
WARMUP_FACTOR = 0.01
NUM_EPOCHS = 5000
SCALER_EPOCHS = 100
BATCH_SIZE = 1024
LEARNING_RATE = 10.**(-3)
# no l2 weight decay (set to -2 in original which meant turn off)
DATALOADER_SHUFFLE = True
TRAIN_FOLDS = list(range(1,15))
VALID_FOLDS = list(range(15,18))
TEST_FOLDS = list(range(18,21))
IS_64BIT = False
SEED = 39
TRAIN_PCT = 0.7
TEST_SUBPCT = 0.5
OPT_DIRECTION = 'maximize'
STANDARD_SCALER_CONSTANT_FEATURE_MASK = True
LINEARPROBE_INITIAL_DROPOUT = False

MLPPROBE_INITIAL_DROPOUT = True
MLPPROBE_HIDDEN_DROPOUT = True
MLPPROBE_HIDDEN_DIMS = [512]

EBIC_GAMMA = 1.0

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


CLS_PPRINT = {'aic_rep': 'AIC (rep.)',
              'aic_rep_avg': 'AIC (rep., per-sample)',
              'bic_rep': 'BIC (rep.)',
              'bic_rep_avg': 'BIC (rep., per-sample)',
              'ebic_rep': 'EBIC (rep.)',
              'ebic_rep_avg': 'EBIC (rep., per-sample)',
              'aic_cls': 'AIC (cls)',
              'aic_cls_avg': 'AIC (cls, per-sample)',
              'bic_cls': 'BIC (rep.)',
              'bic_cls_avg': 'BIC (cls, per-sample)',
              'ebic_cls': 'EBIC (cls)',
              'ebic_cls_avg': 'EBIC (cls, per-sample)',
              'aic_pr': 'AIC (PR)',
              'aic_pr_avg': 'AIC (PR, per-sample)',
              'bic_pr': 'BIC (PR)',
              'bic_pr_avg': 'BIC (PR, per-sample)',
              'ebic_pr': 'EBIC (PR)',
              'ebic_pr_avg': 'EBIC (PR, per-sample)',

              'f1_macro': 'F1 (macro)',
              'f1_micro': 'F1 (micro)',
              'layer_idx': 'Layer Index',
              'loss': 'NLL',
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

#datasets that are regression
REG_DATASETS = set(['tempos'])
# datasets to train on middle on
TOM_DATASETS = set(['tempos'])
ALL_DATASETS = SYNTHEORY_DATASETS.union(SYNTHEORY_PLUS_DATASETS)

