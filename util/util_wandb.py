import os

import wandb
import matplotlib.pyplot as plt
from optuna.integration.wandb import WeightsAndBiasesCallback as WBC

from . import util_main as UMN
from . import util_constants as UC

# https://optuna-integration.readthedocs.io/en/stable/reference/generated/optuna_integration.WeightsAndBiasesCallback.html

# wandb_kwargs is the things passed to wandb.init()
# https://docs.wandb.ai/models/ref/python/functions/init

# to login
# https://docs.wandb.ai/models/ref/python/functions/login

entity='soundbendor'
cur_dir = os.path.dirname(os.path.realpath(__file__))
def login():
    _key = ''
    with open(os.path.join(cur_dir, 'wandbkey'), 'r') as f:
        _tmp = f.readlines()
        _key = _tmp[0].strip()
    wandb.login(key = _key)

# call directly for standard_scaler
def init(wdict, override = None):
    if override is not None:
        wdict.update(override)
    run = None
    cur_args = {'entity': wdict['entity'], 
                'project': wdict['project'],
                'dir': wdict['dir'],
                'id': wdict['id'],
                'name': wdict['name'],
                'config': wdict['config'],
                'settings': wdict['settings'],
                'mode': 'online',
                'reinit': True
                }

    try:
        run = wandb.init(**cur_args)
        print('wandb init in ONLINE mode')
    except:
        cur_args['mode'] = 'offline'
        run = wandb.init(**cur_args)
        print('wandb init in OFFLINE mode')
    return run

def build_config(parser_args, datadict, subsetdict):
    _config = {k:v for (k,v) in vars(parser_args).items()}
    model_shape = UMN.get_acts_shape(_config['model_size'])
    _config['num_epochs'] = UC.NUM_EPOCHS
    _config['is_64bit'] = UC.IS_64BIT
    _config['model_dim'] = model_shape[1]
    _config['model_num_layers'] = model_shape[0]
    _config['dataloader_shuffle'] = UC.DATALOADER_SHUFFLE
    if parser_args.expr_type == 'mlp':
        _config['probe_hidden_dims'] = UC.MLPPROBE_HIDDEN_DIMS
    elif parser_args.expr_type == 'linear':
        _config['probe_hidden_dims'] = []
    _config['early_stopping_check_interval'] = UC.EARLY_STOPPING_CHECK_INTERVAL
    _config['early_stopping_patience'] = UC.EARLY_STOPPING_PATIENCE

    _config['num_preq_steps'] = subsetdict['num_preq_steps']
    _config['train_pct'] = subsetdict['train_pct']
    _config['test_subpct'] = subsetdict['test_subpct']

    _config['is_balanced'] = datadict['is_balanced']
    _config['use_weights'] = (subsetdict['weights'].shape[0] > 0) and (UC.USE_WEIGHTS == True)
    return _config


def build_initdict(parser_args, _config):
    _d = {'entity': entity, 'project': f'mtmidi_mdl-{parser_args.expr_type}', 'dir': UC.WANDB_PATH, 'settings': wandb.Settings(init_timeout=120)}
    _d['config'] = _config
    return _d

def finish_run(cur_run):
    cur_run.finish()

def get_main_callback(initdict, as_multirun = True): 
    return WBC(wandb_kwargs=initdict, as_multirun = as_multirun)

def trial_name_callback(study, trial):
    default_id = f"trial-{trial.number}_layer-{trial.params.get('layer_index', '')}"
    default_name = f"t{trial.number}_l{trial.params.get('layer_index', '')}"
    if wandb.run is not None:
        #wandb.run.id = trial.user_attrs.get('run_name', default_id) # immutable
        wandb.run.name = trial.user_attrs.get('short_name', default_name)
        wandb.run.save()

def add_to_summary(cur_run, add_dict):
    for (k,v) in add_dict.items():
        cur_run.summary[k] = v

def log_array(cur_run, cur_key, cur_arr):
    for i,v in enumerate(cur_arr):
        cur_run.log({cur_key: cur_arr[i]}, step=i)

def log_accum_metrics(cur_run, accum_metrics):
    for i,metricdict in enumerate(accum_metrics):
        cur_run.log(metricdict, step=i)

