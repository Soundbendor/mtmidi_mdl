import os
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from util import util_main as UMN
from util import util_constants as UC
from util import util_rdb as UR
from util import util_optuna as UO
import sys
from PIL import Image, ImageDraw

across_model = True
across_ds = True
PCA_FOLDER = 'pca'
GIF_FOLDER = 'res_gif'
GRAPH_FOLDER = 'res_graph'
suffix = 1
subset = 'train'
dur = 750
seed = UC.SEED
GRAPH_SUBFOLDER = 'pca'
loop = 1
models = ["musicgen-small", "musicgen-medium", "musicgen-large", "MERT-v1-95M", "MERT-v1-330M", "wav2vec2-base", "wav2vec2-large"]
#models = ["MERT-v1-95M"]
#models = ["musicgen-medium", "musicgen-large", "MERT-v1-95M", "MERT-v1-330M", "wav2vec2-base", "wav2vec2-large"]

root_folder = UC.PROJECT_ROOT

exclude_ds = set(['tempos'])
pca_graph_folder = os.path.join(root_folder, GRAPH_FOLDER, PCA_FOLDER)
pca_gif_folder = os.path.join(root_folder, GIF_FOLDER, PCA_FOLDER)
#dses = [x for x in UC.DATASET_SHORT.keys() if x not in exclude_ds]
dses = ["polyrhythms", "dynamics", "notes", "scales", "seventh_chords", "time_signatures", "intervals", "simple_progressions", "chords"]
#dses = ['polyrhythms']

if across_ds == True:
    for ds in dses:
        ds_src_folder = os.path.join(pca_graph_folder, ds)
        ds_dest_folder = UMN.by_projpath_multi(subpaths=[GIF_FOLDER, PCA_FOLDER, ds],make_dir = True)
        for m in models:
            img_arr = []
            outf = f'{m}_sd{seed}_{subset}-{suffix}.gif'
            outpath = os.path.join(ds_dest_folder, outf)
            num_layers = UC.MODEL_NUM_LAYERS[m]
            for l in range(num_layers):
                infile = f'l{l}_sd{seed}_{subset}-{suffix}.png'
                inpath = os.path.join(ds_src_folder, m, infile)
                img_arr.append(Image.open(inpath))
            img_arr[0].save(outpath, save_all=True, append_images=img_arr[1:], duration=dur, loop= loop)
            for im in img_arr:
                im.close()

if across_model == True:
    for m in models:
        num_layers = UC.MODEL_NUM_LAYERS[m]
        for l in range(num_layers):
            img_arr = []
            outf = f'l{l}_sd{seed}_{subset}-{suffix}.gif'
            m_dest_folder = UMN.by_projpath_multi(subpaths=[GIF_FOLDER, PCA_FOLDER, m],make_dir = True)
            outpath = os.path.join(m_dest_folder, outf)
            for ds in dses:
                ds_src_folder = os.path.join(pca_graph_folder, ds)
                infile = f'l{l}_sd{seed}_{subset}-{suffix}.png'
                inpath = os.path.join(ds_src_folder, m, infile)
                img_arr.append(Image.open(inpath))
            img_arr[0].save(outpath, save_all=True, append_images=img_arr[1:], duration=dur, loop= loop)
            for im in img_arr:
                im.close()


