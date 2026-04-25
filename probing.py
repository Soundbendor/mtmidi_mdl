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

def calculate_participation_ratio(generator, train_subset, train_size, emb_dim, cur_mean = None, cur_std = None, shuffle = True, device='cpu'):
    train_dl = TUD.DataLoader(train_subset, batch_size = train_size, shuffle=shuffle, generator=generator)
    
    part_rto = torch.tensor(-1.)
    _mean = None
    _std = None

    with torch.no_grad():
        for batch_idx, data in enumerate(train_dl):
            _ipt, ground_truth = data

        
            if _ipt.shape[0] != train_size:
                print(f'did not load entire split of size {train_size}')
                break
            if cur_mean == None:
                _mean = _ipt.mean(axis=0)
            else:
                _mean = cur_mean
            if cur_std == None:
                _std = _ipt.std(axis=0)
            else:
                _std = cur_std
            ipt = (_ipt - _mean)/_std
            cov = torch.cov(ipt.T)
            if cov.shape[0] != emb_dim or cov.shape[1] != emb_dim or _mean.shape[0] != emb_dim or _std.shape[0] != emb_dim:
                print(f'did not match emb_dim of size {emb_dim}')
                break
            else:
                cur_shape = cov.shape
                print(f'successfully formed cov of shape {cur_shape} for ({train_size},{emb_dim})')
                tr_sq = torch.square(torch.diag(cov).sum())
                sum_sq = torch.square(cov).sum()
                part_rto = tr_sq/sum_sq
    return part_rto, _mean, _std




def train_model(model, scaler, generator, opt_fn, loss_fn, train_subset, scheduler, batch_size=64, global_batch_count = 0, warmup_batch_count = 100, shuffle = True, is_classification = True, device='cpu'):
    train_dl = TUD.DataLoader(train_subset, batch_size = batch_size, shuffle=shuffle, generator=generator)
    
    total_loss = 0.
    iters = 0

    cur_batch_count = global_batch_count
    if scaler != None:
        scaler.eval()

    for batch_idx, data in enumerate(train_dl):
        opt_fn.zero_grad() 
        _ipt, ground_truth = data
        ipt = None

        if scaler != None:
            scaler.partial_fit(_ipt)
            ipt = scaler.transform(_ipt)
        else:
            ipt = _ipt

        model_pred = model(ipt)

        loss = None
        if is_classification == True:
            loss = loss_fn(model_pred, ground_truth)
        else:
            loss = loss_fn(model_pred.flatten(), ground_truth.flatten())
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=UC.GRAD_MAX_NORM)
        opt_fn.step()
        cur_loss = loss.item()
        total_loss += cur_loss
        iters += 1
        if scheduler != None:
            if cur_batch_count <= warmup_batch_count:
                scheduler.step()
            cur_batch_count += 1
    avg_loss = total_loss/float(iters)
    return avg_loss, cur_batch_count

def valid_test_model(model, scaler, generator, loss_fn, valid_subset, batch_size=64, shuffle = True, is_classification = True, device='cpu'):
    valid_dl = TUD.DataLoader(valid_subset, batch_size = batch_size, shuffle=shuffle, generator=generator)
    
    total_loss = 0.
    iters = 0
    # for accumulating ground truths and predictions
    truths = None
    preds = None
    
    model.eval()
    if scaler != None:
        scaler.eval()

    with torch.no_grad():
        for batch_idx, data in enumerate(valid_dl):
            
            _ipt, ground_truth = data
            ipt = None

            if scaler != None:
                ipt = scaler.transform(_ipt)
            else:
                ipt = _ipt

            model_pred = model(ipt)
           
            # don't need loss for testing
            if loss_fn != None:
                if is_classification == True:
                    loss = loss_fn(model_pred, ground_truth)
                else:
                    loss = loss_fn(model_pred.flatten(), ground_truth.flatten())
                
                cur_loss = loss.item()
                total_loss += cur_loss
                iters += 1

            truths, preds = UP.accumulate_truths_preds(truths, ground_truth, preds, model_pred, batch_idx, is_classification)

    return total_loss, truths, preds

def _objective(trial, datadict, subsetdict, configdict, wandbdict, device='cpu'):

    # suggested params
    layer_idx = trial.suggest_categorical('layer_idx', list(range(configdict['model_num_layers'])))
    


    lr_exp = trial.suggest_int('learning_rate_exp', -5, -1, step=1)
    learning_rate = 10**lr_exp

    run_name = UO.get_run_name(configdict, layer_idx, is_short = False)
    trial_number = trial.number
    
    subsetdict['train_subset'].dataset.set_layer_idx(layer_idx)
    subsetdict['valid_subset'].dataset.set_layer_idx(layer_idx)
     
    # load pre-trained scaler
    scaler = None

    if data_norm == True:
        scaler = StandardScaler(with_mean = True, with_std = True, use_64bit = configdict['is_64bit'], dim=configdict['model_dim'], use_constant_feature_mask = configdict['standard_scaler_constant_feature_mask'], device = device)

    # init model
    model = MLPProbe(in_dim =configdict['model_dim'], out_dim = datadict['num_classes'], hidden_dims = configdict['probe_hidden_dims'])
    # init rng
    torch_gen = torch.Generator(device=device)
    torch_gen.manual_seed(configdict['torch_seed'])
    # init opt/loss

    opt_fn = None
    opt_fn = torch.optim.Adam(model.parameters(), lr=learning_rate)

    plateau_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_fn, patience=UC.SCHED_PATIENCE)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(opt_fn, start_factor=UC.WARMUP_FACTOR, end_factor=1.0, total_iters=UC.WARMUP_BATCH_COUNT, last_epoch=-1)
    train_loss = None
    valid_loss = None
    if datadict['is_classification'] == True:
        if datadict['is_balanced'] == True:
            train_loss = nn.CrossEntropyLoss(reduction='mean')
            valid_loss = nn.CrossEntropyLoss(reduction='sum')
        else:
            train_loss = nn.CrossEntropyLoss(reduction='mean')
            valid_loss = nn.CrossEntropyLoss(reduction='sum')
            #train_loss = nn.CrossEntropyLoss(reduction='mean', weight = torch.from_numpy(subsetdict['weights']).to(device=device, dtype=(torch.float32 if configdict['is_64bit'] == False else torch.float64)))
            #valid_loss = nn.CrossEntropyLoss(reduction='sum', weight = torch.from_numpy(subsetdict['weights']).to(device=device, dtype=(torch.float32 if configdict['is_64bit'] == False else torch.float64)))
    else:
        train_loss = nn.MSELoss(reduction='mean')
        valid_loss = nn.MSELoss(reduction='sum')

    # other init
    using_early_stopping =  configdict['early_stopping_check_interval'] > 0
    boredom = 0
    cur_batch_count = 0
        
    best_score = float('-inf')
    ret_score = float('-inf')
    best_loss = float('inf')
    ret_loss = float('inf')
    accum_metrics = []
    best_model_dict = None
    best_scaler_dict = None
    actual_training_epochs = None
    
    # wandbstuff
    cur_run = None
    run_name = None
    short_name = None
    if configdict['use_wandb'] == True:
        param_dict = {'l2_weight_decay_exp': l2_weight_decay_exp, 'learning_rate_exp': lr_exp, 'batch_size': batch_size, 'data_norm': data_norm, 'layer_idx': layer_idx}
        run_name, short_name = UO.get_run_and_short_names(configdict, layer_idx, param_dict) 
        cur_run = UW.init(wandbdict, {'id': run_name, 'name': short_name})
        UW.add_to_summary(cur_run, param_dict)
    # now for the actual train/valid loops
    for epoch_idx in range(configdict['num_epochs']):
        # train/valid
        train_avg_loss, cur_batch_count = train_model(model, scaler, torch_gen, opt_fn, train_loss, subsetdict['train_subset'], warmup_sched, batch_size=batch_size, global_batch_count = cur_batch_count, warmup_batch_count = UC.WARMUP_BATCH_COUNT, shuffle = configdict['dataloader_shuffle'], is_classification = datadict['is_classification'], device=device)
        total_loss, valid_truths, valid_preds = valid_test_model(model, scaler, torch_gen, valid_loss, subsetdict['valid_subset'], batch_size=batch_size, shuffle = configdict['dataloader_shuffle'], is_classification = datadict['is_classification'], device=device)
        if cur_batch_count > UC.WARMUP_BATCH_COUNT:
            plateau_sched.step(total_loss)
        # get validation metrics
        valid_metrics = UME.get_metrics(valid_truths, valid_preds, total_loss, layer_idx, trial_number, datadict, subsetdict, configdict, save_to_csv = False, make_cm = False)
        accum_metrics.append(valid_metrics)
        cur_score = UME.get_optimization_metric(valid_metrics, datadict)

        # early stopping
        if using_early_stopping == False:
            ret_score = cur_score
            ret_loss = total_loss
        else:
            if epoch_idx % configdict['early_stopping_check_interval'] == 0:
                if cur_score > best_score:
                    best_score = cur_score
                    best_loss = total_loss
                    boredom = 0
                    best_model_dict = copy.deepcopy(model.state_dict())
                    if scaler != None:
                        best_scaler_dict = copy.deepcopy(scaler.state_dict())
                else:
                    boredom += 1
            if boredom >= configdict['early_stopping_boredom']:
                actual_training_epochs = epoch_idx + 1
                ret_score = best_score
                ret_loss = best_loss
                break
            elif epoch_idx == (configdict['num_epochs'] - 1):
                # end of training, just report what you have
                actual_training_epochs = epoch_idx + 1
                best_model_dict = copy.deepcopy(model.state_dict())
                if scaler != None:
                    best_scaler_dict = copy.deepcopy(scaler.state_dict())
                ret_score = cur_score
                ret_loss = total_loss

    # model saving
    if best_model_dict != None:
        UP.save_model_dict(best_model_dict, configdict, layer_idx, trial_number)
    if best_scaler_dict != None:
        UP.save_scaler_dict(best_scaler_dict, configdict, layer_idx, trial_number)
    # bookkeeping
    trial.set_user_attr(key='actual_training_epochs', value=actual_training_epochs)
    trial.set_user_attr(key='valid_loss', value=ret_loss)
    # naming
    trial.set_user_attr(key='run_name', value=run_name)
    trial.set_user_attr(key='short_name', value=short_name)

    # wandb stuff
    if configdict['use_wandb'] == True:
        UW.log_accum_metrics(cur_run, accum_metrics)
        UW.add_to_summary(cur_run, {'returned_score': ret_score})
        UW.finish_run(cur_run)
    return ret_score

            


if __name__ == "__main__":
    #### arg parsing
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-ds", "--dataset", type=str, default="polyrhythms", help="dataset")
    parser.add_argument("-ms", "--model_size", type=str, default="musicgen-small", help="musicgen-small/musicgen-medium/musicgen-large/jukebox/baseline-chroma/baseline-concat/baseline-mel/baseline-mfcc")
    parser.add_argument("-et", "--expr_type", type=str, default="mlp", help="experiment type")
    parser.add_argument("-wdb", "--use_wandb", type=strtobool, default=True, help="sync to wandb")
    parser.add_argument("-cd", "--use_cuda", type=strtobool, default=True, help="use cuda")
    parser.add_argument("-ev", "--eval", type=strtobool, default=False, help="eval")
    parser.add_argument("-st", "--stats", type=strtobool, default=False, help="calculate stats")
    parser.add_argument("-eb", "--eval_best", type=strtobool, default=False, help="eval on the best trial per model")
    parser.add_argument("-pr", "--part_rto", type=strtobool, default=False, help="calculate participation ratio")
    parser.add_argument("-en", "--eval_nll", type=strtobool, default=False, help="do eval on training dataset for nll comps")
    parser.add_argument("-rs", "--restart_study", type=strtobool, default=False, help="force restart of optuna study")
    parser.add_argument("-sh", "--from_share", type=strtobool, default=False, help="load from share partition")
    parser.add_argument("-sj", "--slurm_job", type=int, default=0, help="slurm job")
    parser.add_argument("-sf", "--suffix", type=int, default=1, help="suffix")
    parser.add_argument("-tsd", "--torch_seed", type=int, default=UC.SEED, help="torch random seed")
    parser.add_argument("-ssd", "--split_seed", type=int, default=UC.SEED, help="seed for splitting")

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
        from_dir = os.path.join(UC.SHARE_PATH, 'mtmidi_prb')
    datadict = UD.load_data_dict(args.dataset)

    cur_ds = ProbeDataset(datadict, args.model_size, layer_idx=0, from_dir = from_dir, to_torch = True, device = device)
    subsetdict = UP.get_train_test_subsets(cur_ds, datadict, train_folds = UC.TRAIN_FOLDS, valid_folds =UC.VALID_FOLDS, test_folds = UC.TEST_FOLDS, train_pct = UC.TRAIN_PCT, test_subpct = UC.TEST_SUBPCT, seed = args.split_seed)

    # wandb stuff
    configdict = UW.build_config(args, datadict, subsetdict)
    wandbdict = UW.build_initdict(args, configdict)
    if args.stats == True:
        torch_gen = torch.Generator(device=device)
        torch_gen.manual_seed(configdict['torch_seed'])
        train_subset = subsetdict['train_subset']
        for layer_idx in range(configdict['model_num_layers']): 
            subsetdict['pilot_train_subset'].dataset.set_layer_idx(layer_idx)
            cur_pr, cur_mean, cur_std = calculate_participation_ratio(torch_gen, subsetdict['pilot_train_subset'], subsetdict['pilot_train_size'], configdict['model_dim'] , cur_mean = None, cur_std = None, shuffle = True, device=device)
            print(layer_idx, cur_pr, cur_mean, cur_std)
            UP.save_part_rto(cur_pr, configdict, layer_idx, is_train = True)
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

    elif args.part_rto == True or args.eval == True:
        # EVALUATION ========== 

        # load study and get best params given rdb
        eval_params = []

        cur_study = UO.create_or_load_study(args, seed=UC.SEED, evaluation = True)
        if args.eval_best == True:
            best_param_dict, best_trial_dict, attr_dict = UR.get_best_params_of_study(cur_study)
            cur_params = UR.make_eval_param_dict(best_param_dict, best_trial_dict)
            eval_params.append(cur_params)
        else:
            for layer_idx in range(configdict['model_num_layers']):
                best_param_dict, best_trial_dict, attr_dict = UR.get_best_params_of_layer_idx(cur_study, layer_idx)
                cur_params = UR.make_eval_param_dict(best_param_dict, best_trial_dict)
                eval_params.append(cur_params)
        for param_dict in eval_params:
            layer_idx = param_dict['layer_idx']
            trial_number = param_dict['trial_number']
            batch_size = param_dict['batch_size']
            data_norm = param_dict['data_norm']
            
            run_name = UO.get_run_name(configdict, layer_idx, is_short = False)

            # some more init
            # init rng
            torch_gen = torch.Generator(device=device)
            torch_gen.manual_seed(configdict['torch_seed'])

            # set layer indices
            subsetdict['train_subset'].dataset.set_layer_idx(layer_idx)
            subsetdict['valid_subset'].dataset.set_layer_idx(layer_idx)
            subsetdict['test_subset'].dataset.set_layer_idx(layer_idx)


            test_subset = None
            test_size = -1
            if args.eval_nll == True:
                test_subset = subsetdict['train_subset']
                test_size = subsetdict['train_size']
            else:
                test_subset = subsetdict['test_subset']
                test_size = subsetdict['test_size']

            # init/load models
            scaler = None
            if data_norm == True:
                scaler = StandardScaler(with_mean = True, with_std = True, use_64bit = configdict['is_64bit'], dim=configdict['model_dim'], use_constant_feature_mask = configdict['standard_scaler_constant_feature_mask'], device = device)
                UP.load_scaler_dict(scaler, configdict, layer_idx, trial_number, device=device)
                scaler.eval()
            if args.part_rto == True:
                cur_pr = calculate_participation_ratio(scaler, torch_gen, test_subset, test_size, configdict['model_dim'], shuffle = True, device=device)
                if cur_pr <= 0.:
                    print(f'calculation failed at layer {layer_idx}')
                    break
                else:
                    print(f'saving pr ({cur_pr}, train: {args.eval_nll}) for {args.dataset} at layer {layer_idx}')
                    UP.save_part_rto(cur_pr, configdict, layer_idx, trial_number)
            elif args.eval == True:
                model = MLPProbe(in_dim =configdict['model_dim'], out_dim = datadict['num_classes'],  hidden_dims = configdict['probe_hidden_dims'])

                UP.load_model_dict(model, configdict, layer_idx, trial_number, device=device)

                # get loss
                test_loss = None
                if datadict['is_classification'] == True:
                    if datadict['is_balanced'] == True:
                        test_loss = nn.CrossEntropyLoss(reduction='sum')
                    else:
                        test_loss = nn.CrossEntropyLoss(reduction='sum')
                        #test_loss = nn.CrossEntropyLoss(reduction='sum', weight = torch.from_numpy(subsetdict['weights']).to(device=device, dtype=(torch.float32 if configdict['is_64bit'] == False else torch.float64)))
                else:
                    test_loss = nn.MSELoss(reduction='sum')

                print(f'evaluating (train: {args.eval_nll}) for {args.dataset} at layer {layer_idx}')
                test_total_loss, test_truths, test_preds = valid_test_model(model, scaler, torch_gen, test_loss, test_subset , batch_size=batch_size, shuffle = configdict['dataloader_shuffle'], is_classification = datadict['is_classification'], device=device)
                # get test metrics
                test_metrics = UME.get_metrics(test_truths, test_preds, test_total_loss, layer_idx, trial_number, datadict, subsetdict, configdict, save_to_csv = True, make_cm = True)

