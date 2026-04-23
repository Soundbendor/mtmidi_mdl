is_modemix_arr = [False, True]

#### FOR CLASSIFYING BY INDIVIDUAL PROGRESSION ####
# gets orig_prog strings as defined above if is_modemix = False
# else gets sub_prog strings as defined above if is_modemix = True
def progtup_to_progstr(progtup, is_modemix=False):
    arr_str = []
    scaletype = progtup[0]
    if is_modemix == True:
        if scaletype == 'min':
            arr_str.append('mm2')
        else:
            arr_str.append('mm1')
    else:
        arr_str.append(scaletype)
    degstr = ''.join([str(deg) for deg in progtup[1:]])
    arr_str.append(degstr)
    retstr = '-'.join(arr_str)
    return retstr

chordprog_arr = [
        ('maj', 1,4,5,1),
        ('maj', 1,4,6,5),
        ('maj', 1,5,6,4),
        ('maj', 1,6,4,5),
        ('maj', 2,5,1,6),
        ('maj', 4,1,5,6),
        ('maj', 5,4,1,5),
        ('maj', 5,6,4,1),
        ('maj', 6,4,1,5),
        ]



subprog_arr = [progtup_to_progstr(y,is_modemix=x) for y in chordprog_arr for x in is_modemix_arr]
subprog_to_idx = {x:i for (i,x) in enumerate(subprog_arr)}
idx_to_subprog = {i:x for (x,i) in subprog_to_idx.items()}


#### FOR CLASSIFYING MODE MIXTURE ####
num_is_modemix = len(is_modemix_arr)
is_modemix_to_idx = {x:int(x) for x in is_modemix_arr}
idx_to_is_modemix = {x:i for (i,x) in is_modemix_to_idx.items()}

