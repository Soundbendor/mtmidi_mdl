timesig_to_idx ={"(3, 8)":0, "(6, 8)":1, "(9, 8)":2, "(12, 8)":3, "(2, 4)":4, "(3, 4)":5, "(4, 4)":6, "(2, 2)" :7} 
class_arr = [k for (k,v) in timesig_to_idx.items()]
idx_to_timesig = {y:x for (x,y) in timesig_to_idx.items()}
num_timesig = len(timesig_to_idx.keys())
