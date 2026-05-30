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

def get_key():
    _key = ''
    with open(os.path.join(cur_dir, 'wandbkey'), 'r') as f:
        _tmp = f.readlines()
        _key = _tmp[0].strip()
    return _key

def login():
    _key = get_key()
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
                'force': True,
                'name': wdict['name'],
                'config': wdict['config'],
                'settings': wdict['settings'],
                'mode': 'online',
                'reinit': True
                }

    try:
        _key = get_key()
        # hacky ping
        api = wandb.Api(api_key = _key, timeout = WANDB_PING_TIMEOUT)
        viewer = api.viewer()
        run = wandb.init(**cur_args)
        print('wandb init in ONLINE mode')
    except:
        cur_args['mode'] = 'offline'
        run = wandb.init(**cur_args)
        print('wandb init in OFFLINE mode')
    return run


def build_initdict(parser_args, _config):
    _d = {'entity': entity, 'project': f'online_mdl-{parser_args.expr_type}', 'dir': UC.WANDB_PATH, 'settings': wandb.Settings(init_timeout=120)}
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

