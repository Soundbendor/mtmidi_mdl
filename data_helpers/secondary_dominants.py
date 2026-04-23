
#### FOR CLASSIFYING BY SUBSTITUTION TYPE ####
# shorthand for substitution types
sub_types = {'orig': 'N', 'secondary_dominant': 'S', 'tritone_sub': 'T'}
sub_type_arr = ['N', 'S', 'T']
sub_type_to_idx= {x:i for (i,x) in enumerate(sub_type_arr)}
idx_to_sub_type = {i:x for (x,i) in sub_type_to_idx.items()}

num_subtypes = len(sub_type_arr)

#### FOR CLASSIFYING BY CHORD PROGRESSION ####

def progtup_to_progstr(progtup, scale_type='', sub_type='N'):
    arr_str = []
    cur_elt = progtup[0]
    degstr = ''.join([str(deg) for deg in progtup[1:]])
    ret_str = ''
    if len(scale_type) > 0:
        ret_str = f'{scale_type}-'
    ret_str = ret_str + f'e_{cur_elt}-{degstr}'
    if len(sub_type) > 0:
        ret_str = ret_str + f'-s_{sub_type}'
    return ret_str

# first element of tuple refers to which elt of seq to transform into a secondary dominant (1-indexed)
# applies to both major and minor
scale_type_arr = ['maj', 'min']
chordprog_arr = [
        (2,1,6,2,5),
        (2,1,1,4,5),
        (2,1,2,5,1),
        (2,1,3,6,5)
        ]
subprog_arr = []
for scale_type in scale_type_arr:
    for sub_type in sub_type_arr:
        for chordprog in chordprog_arr:
            cur_str = progtup_to_progstr(chordprog, scale_type=scale_type, sub_type=sub_type)
            subprog_arr.append(cur_str)
subprog_to_idx = {x:i for (i,x) in enumerate(subprog_arr)}
idx_to_subprog = {i:x for (x,i) in subprog_to_idx.items()}
num_subprog = len(subprog_arr)

