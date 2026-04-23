# translated from original csv 'chord_progression' to be more concise
prog_arr = ['maj-1564', 'maj-4536', 'min-1671', 'maj-1645', 'min-4711', 'min-1764', 'maj-1465', 'maj-6415', 'maj-2516', 'maj-5641', 'min-1637', 'maj-5415', 'min-1451', 'maj-4156', 'min-1673', 'min-7671', 'maj-1451', 'min-1341', 'min-1251']
major_arr = [False, True]
idx_to_prog = {i: x for (i,x) in enumerate(prog_arr)}
prog_to_idx = {x:i for (i,x) in idx_to_prog.items()}
idx_to_major = {int(x): x for x in major_arr}
major_to_idx = {x:i for (i,x) in idx_to_major.items()}
num_progs = len(prog_arr)
num_types = len(major_arr)
