class_arr = ['ionian', 'dorian', 'phrygian', 'lydian', 'mixolydian', 'aeolian', 'locrian']
mode_to_idx = {x:i for (i,x) in enumerate(class_arr)}
idx_to_mode = {y:x for (x,y) in mode_to_idx.items()}
num_modes = len(class_arr)
