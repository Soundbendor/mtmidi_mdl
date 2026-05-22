import torch, torch.utils.data as TUD
from torch import nn
import optuna, pickle, numpy as np  

import util.util_main as UMN
import util.util_metrics as UME
import util.util_constants as UC
import util.util_data as UD
import util.util_wandb as UW
import util.util_optuna as UO
import util.util_probing as UP
import util.util_rdb as UR

from models.mlpprobe import MLPProbe
from models.standard_scaler import StandardScaler
from probe_dataset import ProbeDataset

from functools import partial
from distutils.util import strtobool
import os, sys, time, argparse, copy

def calculate_mean_stdev(generator, train_subset, train_size, emb_dim, shuffle = True, device='cpu'):
    train_dl = TUD.DataLoader(train_subset, batch_size = train_size, shuffle=shuffle, generator=generator)
    
    _mean = None
    _std = None

    with torch.no_grad():
        for batch_idx, data in enumerate(train_dl):
            _ipt, ground_truth = data

        
            if _ipt.shape[0] != train_size:
                print(f'did not load entire split of size {train_size}')
                break
            _mean = _ipt.mean(axis=0)
            _std = _ipt.std(axis=0)
            if _mean.shape[0] != emb_dim or _std.shape[0] != emb_dim:
                _mean = None
                _std = None
                print(f'did not match emb_dim of size {emb_dim}')
    return _mean, _std




def train_model(model, mean, stdev, train_size, generator, opt_fn, loss_fn, train_subset, batch_size=64, shuffle = True, is_classification = True, device='cpu'):
    train_dl = TUD.DataLoader(train_subset, batch_size = batch_size, shuffle=shuffle, generator=generator)
    
    total_loss = 0.
    iters = 0


    for batch_idx, data in enumerate(train_dl):
        opt_fn.zero_grad() 
        _ipt, ground_truth = data
        ipt = (_ipt - mean)/stdev


        model_pred = model(ipt)

        loss = None
        if is_classification == True:
            loss = loss_fn(model_pred, ground_truth)
        else:
            loss = loss_fn(model_pred.flatten(), ground_truth.flatten())
        
        loss.backward()
        opt_fn.step()
        cur_loss = loss.item()
        total_loss += (cur_loss/train_size)
    return total_loss 

def valid_test_model(model, mean, stdev, generator, loss_fn, valid_subset, batch_size=64, shuffle = True, is_classification = True, device='cpu'):
    valid_dl = TUD.DataLoader(valid_subset, batch_size = batch_size, shuffle=shuffle, generator=generator)
    
    total_loss = 0.
    # for accumulating ground truths and predictions
    truths = None
    preds = None
    
    model.eval()

    with torch.no_grad():
        for batch_idx, data in enumerate(valid_dl):
            
            _ipt, ground_truth = data
            ipt = (_ipt - mean)/stdev


            model_pred = model(ipt)
           
            if loss_fn != None:
                if is_classification == True:
                    loss = loss_fn(model_pred, ground_truth)
                else:
                    loss = loss_fn(model_pred.flatten(), ground_truth.flatten())
                
                cur_loss = loss.item()
                total_loss += cur_loss

            truths, preds = UP.accumulate_truths_preds(truths, ground_truth, preds, model_pred, batch_idx, is_classification)

    return total_loss, truths, preds

def _objective(trial, datadict, subsetdict, configdict, wandbdict, device='cpu'):

    

    layer_idx = trial.suggest_categorical('layer_idx', list(range(configdict['model_num_layers'])))
     
   
    # init model
    # init rng
    torch_gen = torch.Generator(device=device)
    torch_gen.manual_seed(configdict['torch_seed'])
    # init opt/loss

    opt_fn = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loss = None
    valid_loss = None
    if datadict['is_classification'] == True:
        train_loss = nn.CrossEntropyLoss(reduction='sum')
        valid_loss = nn.CrossEntropyLoss(reduction='sum')
    else:
        train_loss = nn.MSELoss(reduction='sum')
        valid_loss = nn.MSELoss(reduction='sum')

    # other init
    using_early_stopping =  configdict['early_stopping_check_interval'] > 0
      
    cur_mean = UP.load_mean(configdict, layer_idx)
    cur_stdev = UP.load_std(configdict, layer_idx)
    # wandbstuff
    cur_run = None
    run_name = None
    short_name = None
    if configdict['use_wandb'] == True:
        param_dict = {'l2_weight_decay_exp': 0, 'learning_rate_exp': UC.LEARNING_RATE, 'batch_size': UC.BATCH_SIZE, 'data_norm': True, 'layer_idx': layer_idx}
        run_name, short_name = UO.get_run_and_short_names(configdict, layer_idx, param_dict) 
        cur_run = UW.init(wandbdict, {'id': run_name, 'name': short_name})
        UW.add_to_summary(cur_run, param_dict)

    num_steps = subsetdict['num_preq_steps']
    # now for the actual train/valid loops

    cur_test_subset = subsetdict['test_subset']
    cur_test_subset.dataset.set_layer_idx(layer_idx)
    
    # outer prequential loop
    test_metrics = []
    train_losses = []
    for preq_idx in  range(num_steps+1):
        model = MLPProbe(in_dim =configdict['model_dim'], out_dim = datadict['num_classes'], hidden_dims = configdict['probe_hidden_dims'])
        
        full_train = False
        if preq_idx == num_steps:
            full_train = True
        cur_train_subset = None
        cur_valid_subset = None
        cur_train_size = None
        if full_train == False:
            cur_train_subset = subsetdict['preq'][i]['train_subset']
            cur_valid_subset = subsetdict['preq'][i]['encode_subset']
            cur_train_size = subsetdict['preq'][i]['train_size']
        else:
            cur_train_subset = subsetdict['preq_all_subset']
            cur_valid_subset = subsetdict['valid_subset']
            cur_train_size = subsetdict['preq_all_size']
        
        cur_train_subset.dataset.set_layer_idx(layer_idx)
        cur_valid_subset.dataset.set_layer_idx(layer_idx)

        patience = 0
        
        best_loss = float('inf')
        accum_metrics = []
        valid_nlls = []
        train_avg_nlls = []
        best_model_dict = None
        actual_training_epochs = None

        for epoch_idx in range(configdict['num_epochs']):
            # train/valid
            train_avg_loss = train_model(model, cur_mean, cur_stdev, cur_train_size, torch_gen, opt_fn, train_loss, cur_train_subset, batch_size=batch_size, shuffle = configdict['dataloader_shuffle'], is_classification = datadict['is_classification'], device=device)
            valid_loss, valid_truths, valid_preds = valid_test_model(model, cur_mean, cur_stdev, torch_gen, valid_loss, cur_valid_subset, batch_size=batch_size, shuffle = configdict['dataloader_shuffle'], is_classification = datadict['is_classification'], device=device)

            train_avg_nlls.append(train_avg_loss)
            # early stopping
            if using_early_stopping == False:
                best_loss = valid_loss
            else:
                if epoch_idx % configdict['early_stopping_check_interval'] == 0:
                    if best_loss < valid_loss:
                        best_loss = valid_loss
                        patience = 0
                        best_model_dict = copy.deepcopy(model.state_dict())
                    else:
                        patience += 1
                if patience >= configdict['early_stopping_patience']:
                    actual_training_epochs = epoch_idx + 1
                    model.load_state_dict(best_model_dict)
                    break
                elif epoch_idx == (configdict['num_epochs'] - 1):
                    # end of training, just report what you have
                    actual_training_epochs = epoch_idx + 1
                    best_loss = valid_loss
        # training on full training set, should not encode actual validation set
        if full_train == False:
            nlls.append(best_loss)

        test_loss, test_truths, test_preds = valid_test_model(model, cur_mean, cur_stdev, torch_gen, valid_loss, cur_test_subset, batch_size=batch_size, shuffle = configdict['dataloader_shuffle'], is_classification = datadict['is_classification'], device=device)
        test_metrics = UME.get_metrics(test_truths, test_preds, nlls, layer_idx, trial_number, datadict, subsetdict, configdict, save_to_csv = True, make_cm = True)

    online_mdl = test_metrics['online_mdl']
    # bookkeeping
    trial.set_user_attr(key='actual_training_epochs', value=actual_training_epochs)
    trial.set_user_attr(key='test_loss', value=test_loss)
    trial.set_user_attr(key='online_mdl', value=online_mdl)
    # naming
    trial.set_user_attr(key='run_name', value=run_name)
    trial.set_user_attr(key='short_name', value=short_name)

    # wandb stuff
    if configdict['use_wandb'] == True:
        UW.log_array(cur_run, 'train_avg_nll', train_avg_nlls)
        UW.finish_run(cur_run)
    return online_mdl

            


if __name__ == "__main__":
    #### arg parsing
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-ds", "--dataset", type=str, default="polyrhythms", help="dataset")
    parser.add_argument("-ms", "--model_size", type=str, default="musicgen-small", help="musicgen-small/musicgen-medium/musicgen-large/jukebox/MERT-v1-95M/MERT-v1-330M/wav2vec2-base/wav2vec2-large")
    parser.add_argument("-et", "--expr_type", type=str, default="mlp", help="experiment type")
    parser.add_argument("-wdb", "--use_wandb", type=strtobool, default=True, help="sync to wandb")
    parser.add_argument("-cd", "--use_cuda", type=strtobool, default=True, help="use cuda")
    parser.add_argument("-ev", "--eval", type=strtobool, default=False, help="eval")
    parser.add_argument("-st", "--stats", type=strtobool, default=False, help="calculate stats")
    parser.add_argument("-eb", "--eval_best", type=strtobool, default=False, help="eval on the best trial per model")
    parser.add_argument("-pr", "--part_rto", type=strtobool, default=False, help="calculate participation ratio")
    parser.add_argument("-rs", "--restart_study", type=strtobool, default=False, help="force restart of optuna study")
    parser.add_argument("-sh", "--from_share", type=strtobool, default=False, help="load from share partition")
    parser.add_argument("-sj", "--slurm_job", type=int, default=0, help="slurm job")
    parser.add_argument("-sf", "--suffix", type=int, default=1, help="suffix")
    parser.add_argument("-tsd", "--torch_seed", type=int, default=UC.SEED, help="torch random seed")
    parser.add_argument("-ssd", "--split_seed", type=int, default=UC.SPLIT_SEED, help="seed for splitting")

    args = parser.parse_args()

    #### some initialization
    device = 'cpu'
    if args.use_cuda == True and torch.cuda.is_available() == True:
        device = 'cuda'
        torch.cuda.empty_cache()
        torch.set_default_device(device)
    torch.manual_seed(args.torch_seed)
    from_dir = ""
    if args.from_share == True:
        from_dir = os.path.join(UC.SHARE_PATH, 'mtmidi_mdl')
    datadict = UD.load_data_dict(args.dataset)

    cur_ds = ProbeDataset(datadict, args.model_size, layer_idx=0, from_dir = from_dir, to_torch = True, device = device)
    subsetdict = UP.get_preq_valid_test_subsets(cur_ds, datadict, train_pct = UC.TRAIN_PCT, test_subpct = UC.TEST_SUBPCT, seed = args.split_seed)

    # wandb stuff
    configdict = UW.build_config(args, datadict, subsetdict)
    wandbdict = UW.build_initdict(args, configdict)
    if args.stats == True:
        torch_gen = torch.Generator(device=device)
        torch_gen.manual_seed(configdict['torch_seed'])
        train_subset = subsetdict['preq_all_subset']
        train_size = subsetdict['preq_all_size']
        for layer_idx in range(configdict['model_num_layers']): 
            train_subset.dataset.set_layer_idx(layer_idx)
            cur_mean, cur_std = calculate_mean_stdev(torch_gen, train_subset, train_size, configdict['model_dim'] , shuffle = True, device=device)
            print(layer_idx, cur_pr, cur_mean, cur_std)
            UP.save_mean(cur_mean, configdict, layer_idx, is_train = True)
            UP.save_std(cur_std, configdict, layer_idx, is_train = True)
    elif args.eval == False:
        # TRAINING ==========
        if args.use_wandb == True:
            UW.login()
        # optuna stuff
        studydict = UO.create_or_load_study(args, seed=UC.SEED, evaluation = False)
        UO.record_dict_in_study(studydict, configdict)
        objective = partial(_objective, datadict=datadict, subsetdict=subsetdict, configdict=configdict, wandbdict=wandbdict, device=device)
        callback_arr = [UO.study_callback]
        studydict['study'].optimize(objective, timeout = None, n_trials = None, n_jobs=1, gc_after_trial = True, callbacks=callback_arr)

