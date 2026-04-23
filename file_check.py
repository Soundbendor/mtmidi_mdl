import os
import polars as pl

datasets = ['polyrhythms', 'dynamics', 'seventh_chords', 'mode_mixture', 'secondary_dominants']

emb_types = ['wav2vec2-large', 'wav2vec2-base', 'MERT-v1-330M', 'MERT-v1-95M', 'jukebox', 'musicgen-large', 'musicgen-medium', 'musicgen-small', 'musicgen-audio', 'baseline-mel', 'baseline-mfcc', 'baseline-chroma', 'baseline-concat']


for ds in datasets:
    cur_path = os.path.join('csv', f'{ds}-metadata.csv')
    actpath = os.path.join('acts', ds)
    df = pl.read_csv(cur_path)
    for i in range(1,21):
        cur = df.filter(pl.col('fold') == i)
        for m in emb_types:
            actpath2 = os.path.join(actpath, m, f'fold_{i}')
            fo = os.listdir(actpath2)
            if len(cur) != len(fo):
                print(m, 'bad')
        
