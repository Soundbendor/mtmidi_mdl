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
    run = wandb.init(
            entity = wdict['entity'], 
            project = wdict['project'],
            dir = wdict['dir'],
            id = wdict['id'],
            name = wdict['name'],
            config = wdict['config'],
            settings=wdict['settings'],
            reinit = True
            )
    return run

def build_config(parser_args, datadict, subsetdict):
    _config = {k:v for (k,v) in vars(parser_args).items()}
    model_shape = UMN.get_acts_shape(_config['model_size'])
    _config['num_epochs'] = UC.NUM_EPOCHS
    _config['is_64bit'] = UC.IS_64BIT
    _config['model_dim'] = model_shape[1]
    _config['model_num_layers'] = model_shape[0]
    _config['dataloader_shuffle'] = UC.DATALOADER_SHUFFLE
    _config['standard_scaler_constant_feature_mask'] = UC.STANDARD_SCALER_CONSTANT_FEATURE_MASK
    if parser_args.expr_type == 'mlp':
        _config['probe_hidden_dims'] = []
        _config['early_stopping_check_interval'] = UC.EARLY_STOPPING_CHECK_INTERVAL
        _config['early_stopping_boredom'] = UC.EARLY_STOPPING_BOREDOM
    elif parser_args.expr_type == 'linear':
        _config['probe_hidden_dims'] = []
        _config['early_stopping_check_interval'] = UC.EARLY_STOPPING_CHECK_INTERVAL
        _config['early_stopping_boredom'] = UC.EARLY_STOPPING_BOREDOM

    elif parser_args.expr_type == 'mlp_2l':
        _config['probe_hidden_dims'] = UC.MLPPROBE_HIDDEN_DIMS
        _config['early_stopping_check_interval'] = UC.EARLY_STOPPING_CHECK_INTERVAL
        _config['early_stopping_boredom'] = UC.EARLY_STOPPING_BOREDOM

    _config['train_folds'] = subsetdict['train_folds']
    _config['valid_folds'] = subsetdict['valid_folds']
    _config['test_folds'] = subsetdict['test_folds']
    _config['is_balanced'] = datadict['is_balanced']
    _config['use_weights'] = subsetdict['weights'].shape[0] > 0
    return _config


def build_initdict(parser_args, _config):
    _d = {'entity': entity, 'project': f'mtmidi_prb-{parser_args.expr_type}', 'dir': UC.WANDB_PATH, 'settings': wandb.Settings(init_timeout=120)}
    _d['config'] = _config
    return _d

def log_scaler_batch_mean_var(cur_run, scalerdict):
    means = scalerdict['mean_vecs_batch'].detach().cpu().numpy().T
    variances = scalerdict['var_vecs_batch'].detach().cpu().numpy().T 
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    for ax, data, title in zip(axes, [means, variances], ["Running Mean", "Running Variance"]):
        im = ax.imshow(data, cmap="coolwarm", aspect="auto")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax)
    
    fig.tight_layout()
    cur_run.log({"standardscaler_means_vars": wandb.Image(fig)})
    plt.close(fig)

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

def log_accum_metrics(cur_run, accum_metrics):
    for i,metricdict in enumerate(accum_metrics):
        cur_run.log(metricdict, step=i)

