import polars as pl
from . import util_main as UMN
from . import util_constants as UC
import os,json
import sqlite3


study_tables = ['alembic_version', 'trial_params', 'studies','trial_system_attributes', 'study_directions', 'trial_user_attributes','study_system_attributes','trial_values','study_user_attributes','trials','trial_heartbeats','version_info','trial_intermediate_values']

# ====== DATABASE HELPERS =======

def get_dbconn(study_dict):
    return sqlite3.connect(study_dict['rdb_filepath'])

def get_trial_id_from_number(conn,number):
    tv = pl.read_database(query = f'select trial_id from trials where number={number}', connection=conn)
    return tv['trial_id'][0]

def get_number_from_trial_id(conn,trial_id):
    tv = pl.read_database(query = f'select number from trials where trial_id={trial_id}', connection=conn)
    return tv['number'][0]



def get_best_trial_id_val(conn):
    tv = pl.read_database(query = 'select trial_id,value from trial_values', connection=conn)
    best_row = tv[tv.select(pl.col('value').arg_max())[0,0]]
    return best_row

def get_best_id_val_from_trial_ids(conn, trial_ids):
    id_str = ','.join([str(int(x)) for x in trial_ids])
    tv = pl.read_database(query = f'select trial_id,value from trial_values where trial_id in ({id_str})', connection=conn)
    best_row = tv[tv.select(pl.col('value').arg_max())[0,0]]
    return best_row


def get_layer_idx_trial_ids(conn, layer_idx):
    tv = pl.read_database(query=f"select * from trial_params where param_name='layer_idx' and param_value={layer_idx}", connection=conn)
    return tv['trial_id'].to_numpy()


def get_best_id_val_of_layer_idx(conn, layer_idx):
    idx_arr = get_layer_idx_trial_ids(conn, layer_idx)
    best_row = get_best_id_val_from_trial_ids(conn, idx_arr)
    return best_row

def get_best_trial_params_of_study(conn):
    best_row = get_best_trial_id_val(conn)
    best_id = best_row['trial_id'][0]
    best_trial_number = get_number_from_trial_id(conn, best_id)
    best_val = best_row['value'][0]
    best_params = pl.read_database(query = f'select param_name,param_value,distribution_json from trial_params where trial_id={best_id}', connection=conn)
    best_dict = {'trial_id': best_id, 'trial_number': best_trial_number, 'value': best_val}
    return (best_params, best_dict)

def get_best_trial_params_of_layer_idx(conn, layer_idx):
    best_row = get_best_id_val_of_layer_idx(conn, layer_idx)
    best_id = best_row['trial_id'][0]
    best_trial_number = get_number_from_trial_id(conn, best_id)
    best_val = best_row['value'][0]
    best_params = pl.read_database(query = f'select param_name,param_value,distribution_json from trial_params where trial_id={best_id}', connection=conn)
    best_dict = {'trial_id': best_id, 'trial_number': best_trial_number, 'value': best_val}
    return (best_params, best_dict)


def close_dbconn(conn):
    conn.close()

def parse_best_params(best_params_df):
    param_dict = {x['param_name']: {'_value': x['param_value'], 'dict': json.loads(x['distribution_json'])} for x in best_params_df.to_dicts()}
    for param,pdict in param_dict.items():
        cur_dist = pdict['dict']['name']
        if cur_dist == 'CategoricalDistribution':
            cur_choices = pdict['dict']['attributes']['choices']
            choice_idx = int(pdict['_value'])
            param_dict[param]['value'] = cur_choices[choice_idx]
        elif cur_dist == "IntDistribution":
            param_dict[param]['value'] = int(pdict["_value"])
        else:
            param_dict[param]['value'] = pdict["_value"]
    return param_dict

def get_study_attr(conn):
    study_attr = pl.read_database(query = f'select * from study_user_attributes', connection=conn)
    ret_dict = {x['key']: json.loads(x['value_json']) for x in study_attr.to_dicts()}
    return ret_dict

def get_best_params_of_study(study_dict):
    conn = get_dbconn(study_dict)
    (best_params, best_dict) = get_best_trial_params_of_study(conn)
    param_dict = parse_best_params(best_params)
    attr_dict = get_study_attr(conn)
    close_dbconn(conn)
    #param_dict['best_value'] = best_val
    return (param_dict, best_dict, attr_dict)

def get_best_params_of_layer_idx(study_dict, layer_idx):
    conn = get_dbconn(study_dict)
    (best_params, best_dict) = get_best_trial_params_of_layer_idx(conn, layer_idx)
    param_dict = parse_best_params(best_params)
    attr_dict = get_study_attr(conn)
    close_dbconn(conn)
    #param_dict['best_value'] = best_val
    return (param_dict, best_dict, attr_dict)

# ====== FETCHERS OF PARAMS FOR EVAL ======
def make_eval_param_dict(best_param_dict, best_trial_dict):
    ret = {}
    ret['trial_number'] = best_trial_dict['trial_number']

    for m in ['layer_idx', 'dropout', 'learning_rate_exp', 'data_norm', 'batch_size']:
        ret[m] = best_param_dict[m]['value']
    return ret
   
