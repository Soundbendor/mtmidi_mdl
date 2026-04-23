num_intervals = 12
class_arr = [k+1 for k in range(num_intervals)] #intervals are 1-indexed
interval_to_idx = {x:(x-1) for x in class_arr} # let the actual indices be 0-idx
idx_to_interval = {(y,x) for (x,y) in interval_to_idx.items()}
