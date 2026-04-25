import os, pickle
import optuna

from . import util_main as UMN
from . import util_constants as UC

search_space = {'learning_rate_exp': [-4] }

def study_callback(study, trial):
    study_sampler_path = study.user_attrs['sampler_filepath']
    with open(study_sampler_path, 'wb') as f:
        pickle.dump(study.sampler, f)

def get_layer_search_space(model_size):
    ret = []
    num_layers = UC.MODEL_NUM_LAYERS[model_size]
    ret = list(range(num_layers))
    return ret

def create_study_name(parser_args):
    return f'{parser_args.expr_type}-{parser_args.dataset}_{parser_args.model_size}-{parser_args.suffix}'

def record_dict_in_study(studydict, cur_dict):
    flat_dict = UMN.dict_arrayargs_to_str(cur_dict)
    for k,v in flat_dict.items():
        studydict['study'].set_user_attr(k,v)

def create_or_load_study(parser_args, seed=UC.SEED, evaluation = False):
    ret = {}

    cur_study_name = create_study_name(parser_args)
    sampler_dir = UMN.by_projpath(UC.SAMPLER_FOLDER, True)
    rdb_dir = UMN.by_projpath(UC.RDB_FOLDER, True)
    sampler_filepath = os.path.join(sampler_dir, f'{cur_study_name}.pkl')
    rdb_filepath = os.path.join(rdb_dir, f'{cur_study_name}.db')
    resuming = False
    cur_sampler = None
    if evaluation == False and os.path.exists(rdb_filepath) == True and os.path.exists(sampler_filepath) == True and parser_args.restart_study == False:
        resuming = True
        cur_sampler = pickle.load(open(sampler_filepath, 'rb'))
    rdb_url = "sqlite:///" + rdb_filepath
    ret['study_name'] = cur_study_name
    ret['sampler_filepath'] = sampler_filepath
    ret['rdb_filepath'] = rdb_filepath
    ret['resuming_study'] = resuming
    ret['study_seed'] = seed

    if cur_sampler == None:
        cur_search_space = None
        if parser_args.model_size in UC.SINGLE_LAYER_MODELS:
            cur_search_space = {k:v for (k,v) in singlelayer_search_space.items()}
        else:
            cur_search_space = {k:v for (k,v) in multilayer_search_space.items()}
        cur_search_space['layer_idx'] = get_layer_search_space(parser_args.model_size) 
        cur_sampler = optuna.samplers.GridSampler(cur_search_space, seed=seed)

    if evaluation == False:
        ret['study'] = optuna.create_study(study_name=cur_study_name, sampler = cur_sampler, storage=rdb_url, direction=UC.OPT_DIRECTION, load_if_exists = (resuming == True and parser_args.restart_study == False))
        ret['study'].set_user_attr('sampler_filepath', sampler_filepath)
    return ret


def get_run_name(configdict, layer_idx, other = None, is_short = False):
    _dataset = configdict['dataset']
    suffix = configdict['suffix']
    _model_size = configdict['model_size']
    if is_short == True:
        _dataset = UC.DATASET_SHORT[_dataset]
        _model_size = UC.MODEL_SIZES_SHORT[_model_size]
    ret = None
    if other == None:
        ret = f'{_dataset}_{_model_size}_l{layer_idx}-{suffix}'
    else:
        ret = f'{_dataset}_{_model_size}_l{layer_idx}_{other}-{suffix}'
    return ret 



#format string to data normalization be appendable to a run name
def data_norm_string_format(dnorm, is_short = False ):
    dn_int = 1 if dnorm == True else 0
    if is_short == False:
        return f'datanorm{dn_int}'
    else: 
        return f'dn{dn_int}'



#format string to make batch size be appendable to a run name
def batch_size_string_format(bsz, is_short = False ):
    bs_int = abs(int(bsz))
    if is_short == False:
        return f'batchsize{bs_int}'
    else: 
        return f'bs{bs_int}'

#format string to make learning rate be appendable to a run name
def learning_rate_string_format(lr_exp, is_short = False ):
    lr_int = abs(int(lr_exp))
    if is_short == False:
        return f'learningrate{lr_int}'
    else: 
        return f'lr{lr_int}'


def get_run_and_short_names(configdict, layer_idx, name_params):
    other_long_arr = []
    other_short_arr = []


    
    if 'learning_rate_exp' in name_params.keys():
        lrexp_long = learning_rate_string_format(name_params['learning_rate_exp'] , is_short = False)
        lrexp_short = learning_rate_string_format(name_params['learning_rate_exp'] , is_short = True)
        other_long_arr.append(lrexp_long)
        other_short_arr.append(lrexp_short)

    
    if 'batch_size' in name_params.keys():
        bsz_long = batch_size_string_format(name_params['batch_size'] , is_short = False)
        bsz_short = batch_size_string_format(name_params['batch_size'] , is_short = True)
        other_long_arr.append(bsz_long)
        other_short_arr.append(bsz_short)
    
    if 'data_norm' in name_params.keys():
        dnorm_long = data_norm_string_format(name_params['data_norm'] , is_short = False)
        dnorm_short = data_norm_string_format(name_params['data_norm'] , is_short = True)
        other_long_arr.append(dnorm_long)
        other_short_arr.append(dnorm_short)


    other_long = '_'.join(other_long_arr)
    other_short = '_'.join(other_short_arr)
    run_name = get_run_name(configdict, layer_idx, other = other_long, is_short = False) 
    short_name = get_run_name(configdict, layer_idx, other = other_short, is_short = True)
    return run_name, short_name



