import os
import polars as pl
from util import util_main as UMN
from util import util_constants as UC
import sys

mv_test = True
#datdir = '/nfs/guille/eecs_research/soundbendor/kwand/mtmidi_mdl/acts'
datdir = '/nfs/hpc/share/kwand/mtmidi_mdl/acts/'
datasets = ['notes', 'scales', 'intervals', 'simple_progressions', 'tempos', 'chords', 'time_signatures']
if len(sys.argv) > 1:
    dataset = sys.argv[1]

csvdir = UMN.by_projpath(subpath='csv',make_dir = False)

#os.mkdir(dstestdir)

#for model_size in UC.MODEL_NUM_LAYERS.keys():
#for model_size in ['musicgen-large']:
for dataset in datasets:
    dsdatdir = os.path.join(datdir, dataset)
    csvfile = os.path.join(csvdir, f'{dataset}-metadata.csv')
    df = pl.read_csv(csvfile)
    for model_size in ["MERT-v1-95M", "MERT-v1-330M", "wav2vec2-base", "wav2vec2-large",'musicgen-small', 'musicgen-medium', 'musicgen-large']:

        for i in range(len(df)):
            cur_fp = os.path.join(dsdatdir, model_size)
            cur_name = df[i]['name'][0]
            cur_dat = f'{cur_name}.dat'
            from_fp = os.path.join(cur_fp, cur_dat)
            cur_fold = df[i]['fold'][0]
            fold_folder = f'fold_{cur_fold}'
            fold_fp = os.path.join(cur_fp, fold_folder)
            to_fp = os.path.join(fold_fp, cur_dat)
            if os.path.isdir(fold_fp) == False:
                os.mkdir(fold_fp)
            #print(f'moving from {from_fp} to {to_fp}')
            os.rename(from_fp, to_fp)



