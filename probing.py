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
from probe_dataset import ProbeDataset
from two_nn import calc_twonn,find_zero_dists

from functools import partial
from distutils.util import strtobool
import os, sys, time, argparse, copy

def set_seed(seed):
    torch.manual_seed(seed)

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



def calculate_biased_participation_ratio(generator, train_subset, train_size, cur_mean, cur_stdev, emb_dim, nstd = False, shuffle = True, device='cpu'):
    train_dl = TUD.DataLoader(train_subset, batch_size = train_size, shuffle=shuffle, generator=generator)
    

    with torch.no_grad():
        for batch_idx, data in enumerate(train_dl):
            _ipt, ground_truth = data
            if _ipt.shape[0] != train_size:
                print(f'did not load entire split of size {train_size}')
                break

            ipt = None
            if nstd == False:
                ipt = (_ipt - cur_mean)/cur_stdev
            else:
                ipt = (_ipt - cur_mean)


        
            cur_cov = (ipt.T @ ipt)/(train_size - 1)
            if cur_cov.shape[0] != emb_dim or cur_cov.shape[1] != emb_dim:
                print(f'cov matrix did not match emb_dim of size {emb_dim}')
            cur_cov2 = cur_cov @ cur_cov
            if cur_cov2.shape[0] != emb_dim or cur_cov2.shape[1] != emb_dim:
                print(f'cov*cov matrix did not match emb_dim of size {emb_dim}')
            cur_nom = torch.pow(torch.trace(cur_cov), 2)
            cur_denom = torch.trace(cur_cov2)
            ret = cur_nom/cur_denom
    return ret



def train_model(model, mean, stdev, train_size, generator, opt_fn, loss_fn, train_subset, nstd = False, batch_size=64, shuffle = True, is_classification = True, device='cpu'):
    train_dl = TUD.DataLoader(train_subset, batch_size = batch_size, shuffle=shuffle, generator=generator)
    
    total_loss = 0.


    for batch_idx, data in enumerate(train_dl):
        opt_fn.zero_grad() 
        _ipt, ground_truth = data
        ipt = None
        if nstd == False:
            ipt = (_ipt - mean)/stdev
        else:
            ipt = (_ipt - mean)


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

def valid_test_model(model, mean, stdev, generator, loss_fn, valid_subset, nstd = False, batch_size=64, shuffle = True, is_classification = True, device='cpu'):
    valid_dl = TUD.DataLoader(valid_subset, batch_size = batch_size, shuffle=shuffle, generator=generator)
    
    total_loss = 0.
    # for accumulating ground truths and predictions
    truths = None
    preds = None
    
    model.eval()

    with torch.no_grad():
        for batch_idx, data in enumerate(valid_dl):
            
            _ipt, ground_truth = data
            ipt = None
            if  nstd == False:
                ipt = (_ipt - mean)/stdev
            else:
                ipt = (_ipt - mean)



            model_pred = model(ipt)
           
            if loss_fn != None:
                loss = None
                if is_classification == True:
                    loss = loss_fn(model_pred, ground_truth)
                else:
                    loss = loss_fn(model_pred.flatten(), ground_truth.flatten())
                
                cur_loss = loss.item()
                total_loss += cur_loss

            truths, preds = UP.accumulate_truths_preds(truths, ground_truth, preds, model_pred, batch_idx, is_classification)

    return total_loss, truths, preds


def calc_twonn_curve(layer_idx, datadict, subsetdict, configdict, device='cpu'):
    num_steps = subsetdict['num_preq_steps']
    preq_all_size = subsetdict['preq_all_size']
    
    twonn_ids = []
    twonn_sizes = []
    successful = True
    for preq_idx in range(num_steps+1):

        # init rng
        torch_gen = torch.Generator(device=device)
        torch_gen.manual_seed(configdict['seed'])
        set_seed(configdict['seed'])





        # setting subsets
        full_train = False
        if preq_idx == num_steps:
            full_train = True
        cur_train_subset = None
        cur_train_size = None
        if full_train == False:
            cur_train_subset = subsetdict['preq'][preq_idx]['train_subset']
            cur_train_size = subsetdict['preq'][preq_idx]['train_size']
        else:
            cur_train_subset = subsetdict['preq_all_subset']
            cur_train_size = subsetdict['preq_all_size']
        
        cur_train_subset.dataset.set_layer_idx(layer_idx)
        train_pct = 100. * (cur_train_size/preq_all_size)

        print(f'layer/preq ({layer_idx},{preq_idx}): train({cur_train_size}) ({train_pct:.4f})')
        train_dl = TUD.DataLoader(cur_train_subset, batch_size = cur_train_size, shuffle=True, generator=torch_gen)
    
        cur_id = -1.
        with torch.no_grad():
            for batch_idx, data in enumerate(train_dl):
                _ipt, ground_truth = data

            
                if _ipt.shape[0] != cur_train_size:
                    print(f'did not load entire split of size {train_size}')
                    successful = False
                else: 
                    cur_id = calc_twonn(_ipt.to(device), batch_size = UC.TWONN_BATCH_SIZE, unused_pct = UC.TWONN_UNUSED_PCT).item()
        print('id:', cur_id)
        if cur_id >= 0.:
            twonn_ids.append(cur_id)
            twonn_sizes.append(cur_train_size)
        else:
            successful = False
    # bookkeeping
    if successful == True:
        UP.save_twonn_ids(twonn_ids, configdict, layer_idx)
        UP.save_twonn_sizes(twonn_sizes, configdict, layer_idx)
    return successful



def _objective(trial, datadict, subsetdict, configdict, wandbdict, device='cpu'):

    
    trial_number = trial.number

    layer_idx = trial.suggest_categorical('layer_idx', list(range(configdict['model_num_layers'])))
     
   

    # other init
    using_early_stopping =  configdict['early_stopping_check_interval'] > 0
      
    cur_mean = torch.from_numpy(UP.load_mean(configdict, layer_idx)).to(device)
    cur_stdev = torch.from_numpy(UP.load_std(configdict, layer_idx)).to(device)
    # wandbstuff
    cur_run = None
    run_name = None
    short_name = None
    if configdict['use_wandb'] == True:
        param_dict = {'learning_rate_exp': np.log10(configdict['learning_rate']), 'batch_size': configdict['batch_size'], 'data_norm': True, 'layer_idx': layer_idx, 'seed': configdict['seed']}
        run_name, short_name = UO.get_run_and_short_names(configdict, layer_idx, param_dict) 
        cur_run = UW.init(wandbdict, {'id': run_name, 'name': short_name})
        UW.add_to_summary(cur_run, param_dict)

    num_steps = subsetdict['num_preq_steps']


    # now for the actual train/valid loops
    cur_test_subset = subsetdict['test_subset']
    cur_test_subset.dataset.set_layer_idx(layer_idx)
    
    # outer prequential loop
    accum_metrics = []
    valid_nlls = []
    test_nll = -1. 
    train_avg_nlls = []
    actual_training_epochs = []

    preq_all_size = subsetdict['preq_all_size']

    for preq_idx in  range(num_steps+1):

        # init rng
        torch_gen = torch.Generator(device=device)
        torch_gen.manual_seed(configdict['seed'])
        set_seed(configdict['seed'])

        # init model
        model = MLPProbe(in_dim =configdict['model_dim'], out_dim = datadict['num_classes'], hidden_dims = configdict['probe_hidden_dims'])

        # init optimizer and loss
        opt_fn = torch.optim.Adam(model.parameters(), lr=configdict['learning_rate'])

        train_loss_fn = None
        valid_loss_fn = None
        if datadict['is_classification'] == True:
            train_loss_fn = nn.CrossEntropyLoss(reduction='sum')
            valid_loss_fn = nn.CrossEntropyLoss(reduction='sum')
        else:
            train_loss_fn = nn.MSELoss(reduction='sum')
            valid_loss_fn = nn.MSELoss(reduction='sum')



        # setting subsets
        full_train = False
        if preq_idx == num_steps:
            full_train = True
        cur_train_subset = None
        cur_valid_subset = None
        cur_train_size = None
        cur_valid_size = None
        if full_train == False:
            cur_train_subset = subsetdict['preq'][preq_idx]['train_subset']
            cur_valid_subset = subsetdict['preq'][preq_idx]['encode_subset']
            cur_train_size = subsetdict['preq'][preq_idx]['train_size']
            cur_valid_size = subsetdict['preq'][preq_idx]['encode_size']
        else:
            cur_train_subset = subsetdict['preq_all_subset']
            cur_valid_subset = subsetdict['valid_subset']
            cur_train_size = subsetdict['preq_all_size']
            cur_valid_size = subsetdict['valid_size']
        
        cur_train_subset.dataset.set_layer_idx(layer_idx)
        cur_valid_subset.dataset.set_layer_idx(layer_idx)

        train_pct = 100. * (cur_train_size/preq_all_size)
        valid_pct = 100. * (cur_valid_size/preq_all_size)

        print(f'layer/preq ({layer_idx},{preq_idx}): train/valid ({cur_train_size},{cur_valid_size}) ({train_pct:.4f},{valid_pct:.4f})')

        patience = 0
        
        best_loss = float('inf')

        best_model_dict = None

        for epoch_idx in range(configdict['num_epochs']):
            # train/valid
            train_avg_loss = train_model(model, cur_mean, cur_stdev, cur_train_size, torch_gen, opt_fn, train_loss_fn, cur_train_subset, nstd=configdict['nonstandard'], batch_size=configdict['batch_size'], shuffle = configdict['dataloader_shuffle'], is_classification = datadict['is_classification'], device=device)
            valid_loss, valid_truths, valid_preds = valid_test_model(model, cur_mean, cur_stdev, torch_gen, valid_loss_fn, cur_valid_subset, nstd=configdict['nonstandard'], batch_size=configdict['batch_size'], shuffle = configdict['dataloader_shuffle'], is_classification = datadict['is_classification'], device=device)

            train_avg_nlls.append(train_avg_loss)
            # early stopping
            if using_early_stopping == False:
                best_loss = valid_loss
            else:
                last_epoch = epoch_idx == (configdict['num_epochs'] - 1)
                if epoch_idx % configdict['early_stopping_check_interval'] == 0:
                    if valid_loss < np.max([(best_loss - configdict['patience_delta']), 0.]):
                        best_loss = valid_loss
                        patience = 0
                        best_model_dict = copy.deepcopy(model.state_dict())
                    else:
                        patience += 1
                if patience >= configdict['early_stopping_patience'] or last_epoch == True:
                    actual_training_epochs = epoch_idx + 1
                    model.load_state_dict(best_model_dict)
                    break
                elif last_epoch == True:
                    # end of training, just report what you have
                    actual_training_epochs = epoch_idx + 1
                    best_loss = valid_loss
        # training on full training set, should not encode actual validation set
        if full_train == False:
            valid_nlls.append(best_loss)
        else:
            test_nll = best_loss

        test_loss, test_truths, test_preds = valid_test_model(model, cur_mean, cur_stdev, torch_gen, valid_loss_fn, cur_test_subset, nstd = configdict['nonstandard'], batch_size=configdict['batch_size'], shuffle = configdict['dataloader_shuffle'], is_classification = datadict['is_classification'], device=device)
        test_metrics = UME.get_metrics(test_truths, test_preds, valid_nlls, preq_idx, layer_idx, trial_number, datadict, subsetdict, configdict, save_to_csv = True, make_cm = True)
        accum_metrics.append(test_metrics)

    # bookkeeping

    metric_dict = UME.accum_metrics_to_metric_dict(accum_metrics)
    for k,v in metric_dict.items():
        trial.set_user_attr(key=k, value=v)
    
    final_mdl = metric_dict['online_mdl'][-1]
    trial.set_user_attr(key='valid_nlls', value = valid_nlls)
    trial.set_user_attr(key='test_nll', value = test_nll)
    trial.set_user_attr(key='final_mdl', value = final_mdl)
    trial.set_user_attr(key='train_avg_nlls', value = train_avg_nlls)
    trial.set_user_attr(key='actual_training_epochs', value = actual_training_epochs)
    
    # naming
    trial.set_user_attr(key='run_name', value=run_name)
    trial.set_user_attr(key='short_name', value=short_name)
    # wandb stuff
    if configdict['use_wandb'] == True:
        UW.log_array(cur_run, 'train_avg_nll', train_avg_nlls)
        UW.finish_run(cur_run)
    return final_mdl

def get_zero_dists(layer_idx, datadict, subsetdict, configdict, device='cpu'):
    
    successful = True
    # init rng
    torch_gen = torch.Generator(device=device)
    torch_gen.manual_seed(configdict['seed'])
    set_seed(configdict['seed'])
            
    cur_train_subset = subsetdict['preq_all_subset']
    cur_train_size = subsetdict['preq_all_size']
        
    cur_train_subset.dataset.set_layer_idx(layer_idx)
    cur_train_subset.dataset.set_emit_name(True)

    train_dl = TUD.DataLoader(cur_train_subset, batch_size = cur_train_size, shuffle=True, generator=torch_gen)
   
    zd_dict = None
    with torch.no_grad():
        for batch_idx, data in enumerate(train_dl):
            _ipt, ground_truth, data_names = data

        
            if _ipt.shape[0] != cur_train_size:
                print(f'did not load entire split of size {train_size}')
                successful = False
            else: 
                zd_dict = find_zero_dists(_ipt.to(device), data_names, batch_size = UC.TWONN_BATCH_SIZE)
    # bookkeeping
    if successful == True:
        UP.save_zero_dist_csv(zd_dict, configdict, layer_idx)
    return successful


if __name__ == "__main__":
    #### arg parsing
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-ds", "--dataset", type=str, default="polyrhythms", help="dataset")
    parser.add_argument("-ms", "--model_size", type=str, default="musicgen-small", help="musicgen-small/musicgen-medium/musicgen-large/jukebox/MERT-v1-95M/MERT-v1-330M/wav2vec2-base/wav2vec2-large")
    parser.add_argument("-et", "--expr_type", type=str, default="mlp", help="experiment type")
    parser.add_argument("-zd", "--zero_dist", type=strtobool, default=False, help="find zero dist embeddings")
    parser.add_argument("-dbg", "--debug", type=strtobool, default=False, help="debug")
    parser.add_argument("-wdb", "--use_wandb", type=strtobool, default=True, help="sync to wandb")
    parser.add_argument("-cd", "--use_cuda", type=strtobool, default=True, help="use cuda")
    parser.add_argument("-st", "--stats", type=strtobool, default=False, help="calculate stats")
    parser.add_argument("-bpr", "--biased_part_rto", type=strtobool, default=False, help="calculate biased participation ratio")
    parser.add_argument("-twn", "--twonn", type=strtobool, default=False, help="calculate twonn")
    parser.add_argument("-upr", "--unbiased_part_rto", type=strtobool, default=False, help="calculate biased participation ratio")
    parser.add_argument("-nstd", "--nonstandard", type=strtobool, default = False, help="do not divide data by feature-wise standard deviation")
    parser.add_argument("-rs", "--restart_study", type=strtobool, default=False, help="force restart of optuna study")
    parser.add_argument("-sh", "--from_share", type=strtobool, default=True, help="load from share partition")
    parser.add_argument("-sj", "--slurm_job", type=int, default=0, help="slurm job")
    parser.add_argument("-sf", "--suffix", type=int, default=1, help="suffix")

    args = parser.parse_args()

    #### some initialization
    device = 'cpu'
    if args.use_cuda == True and torch.cuda.is_available() == True:
        device = 'cuda'
        torch.cuda.empty_cache()
        torch.set_default_device(device)
    from_dir = ""
    if args.from_share == True:
        from_dir = os.path.join(UC.SHARE_PATH, 'mtmidi_mdl')
    datadict = UD.load_data_dict(args.dataset)

    cur_ds = ProbeDataset(datadict, args.model_size, layer_idx=0, from_dir = from_dir, to_torch = True, device = device)

    configdict = UP.build_config(args, datadict)
    subsetdict = UP.get_preq_valid_test_subsets(cur_ds, datadict, configdict)
    wandbdict = UW.build_initdict(args, configdict)

    if args.debug == True:
        pass
    elif args.stats == True:
        train_subset = subsetdict['preq_all_subset']
        train_size = subsetdict['preq_all_size']
        for layer_idx in range(configdict['model_num_layers']): 
            torch_gen = torch.Generator(device=device)
            torch_gen.manual_seed(configdict['seed'])
            train_subset.dataset.set_layer_idx(layer_idx)
            cur_mean, cur_std = calculate_mean_stdev(torch_gen, train_subset, train_size, configdict['model_dim'] , shuffle = True, device=device)
            print(layer_idx, cur_mean, cur_std)
            UP.save_mean(cur_mean, configdict, layer_idx)
            UP.save_std(cur_std, configdict, layer_idx)
    elif args.biased_part_rto == True:
        train_subset = subsetdict['preq_all_subset']
        train_size = subsetdict['preq_all_size']
        for layer_idx in range(configdict['model_num_layers']): 
            torch_gen = torch.Generator(device=device)
            torch_gen.manual_seed(configdict['seed'])
            train_subset.dataset.set_layer_idx(layer_idx)

            cur_mean = torch.from_numpy(UP.load_mean(configdict, layer_idx)).to(device)
            cur_stdev = torch.from_numpy(UP.load_std(configdict, layer_idx)).to(device)
            cur_bpr = calculate_biased_participation_ratio(torch_gen, train_subset, train_size, cur_mean, cur_stdev, configdict['model_dim'] , nstd = configdict['nonstandard'], shuffle = True, device=device)
            print(layer_idx, cur_bpr)
            UP.save_biased_part_rto(cur_bpr, configdict, layer_idx)
    elif args.twonn == True:
        for layer_idx in range(configdict['model_num_layers']):
            cur_success = calc_twonn_curve(layer_idx, datadict, subsetdict, configdict, device=device)
            print(layer_idx, cur_success)
    elif args.zero_dist == True:
        for layer_idx in range(configdict['model_num_layers']):
            cur_success = get_zero_dists(layer_idx, datadict, subsetdict, configdict, device=device)
            print(layer_idx, cur_success)
    else:
        # TRAINING ==========
        if args.use_wandb == True:
            UW.login()
        # optuna stuff
        studydict = UO.create_or_load_study(args, configdict, evaluation = False)
        UO.record_dict_in_study(studydict, configdict)
        objective = partial(_objective, datadict=datadict, subsetdict=subsetdict, configdict=configdict, wandbdict=wandbdict, device=device)
        callback_arr = [UO.study_callback]
        studydict['study'].optimize(objective, timeout = None, n_trials = None, n_jobs=1, gc_after_trial = True, callbacks=callback_arr)

