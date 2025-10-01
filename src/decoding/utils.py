# Decoder utils
import numpy as np
from src.decoding import *

def init_decoder(decoder_type, decoder_params):
    if decoder_type in ['linear']:
        decoder = LinearDecoder(model_name=decoder_params['decoder_type'], **decoder_params)
    elif decoder_type in ['nonlinear']:
        print(f'[DEBUG] {decoder_params}')
        decoder = NonLinearDecoder(**decoder_params)
    
    return decoder

def avg_perf_dataset(model_name, data_name_lst, df):
    # Step 1: Filter the DataFrame based on model_name
    filtered_df = df[df["model_name"] == model_name]

    # Step 2: Group by data_seed (averaging over model_seed)
    averaged_df = filtered_df.groupby("model_seed").mean()
    # Step 3: Convert to dictionary format {data_name: {metric: avg_value}}
    result_dict = {}
    for sess_name in data_name_lst:
        results_dict_seed = averaged_df[f'{sess_name}'].to_dict()
        results_dict_lst = {}
        for metric, dict in results_dict_seed.items():
            results_dict_lst[metric] = np.array(list(dict.values())).mean()
        result_dict[sess_name] = results_dict_lst

    return result_dict

def avg_perf(result_dict):
    results = {
        'recon_MSE' : [],
        'recon_MAE' : [],
        'recon_R2'  : [],
        'recon_Corr': [],
        'decoder_MSE': [],
        'decoder_MAE': [],
        'decoder_R2' : [],
        'decoder_Corr': [],
        'unidecoder_MSE': [],
        'unidecoder_MAE': [],
        'unidecoder_R2': [],
        'unidecoder_Corr': [],
        'supdecoder_MSE': [],
        'supdecoder_MAE': [],
        'supdecoder_R2' : [],
        'supdecoder_Corr': []
    }
    # Iterate through each dataset in result_dict
    for metrics_dict in result_dict.values():
        for metric, value in metrics_dict.items():
            results[metric].append(value)

    return results

def train_behv_decoder_separate(dataset_info, z_hat_dict, behv_dict, params):
    z_hat_train_dict = z_hat_dict['train']
    z_hat_valid_dict = z_hat_dict['valid']
    z_hat_test_dict  = z_hat_dict['test']

    behv_train_dict = behv_dict['train']
    behv_valid_dict = behv_dict['valid']
    behv_test_dict  = behv_dict['test']

    decoder_type         = params['decoder_type']
    decoder_params       = params['decoder_params']
    decoder_train_params = params['decoder_train_params']
    print(f'[INFO] Train Separate-decoder ===>')
    # Step 1: Initialize results recording dictionary
    results_train_dict = {}
    results_test_dict  = {}

    # Step 2: Train each agent individually
    for i, sess_name in enumerate(dataset_info.keys()):
        z_hat_train = z_hat_train_dict[sess_name]
        z_hat_valid = z_hat_valid_dict[sess_name]
        z_hat_test  = z_hat_test_dict[sess_name]

        behv_train  = behv_train_dict[sess_name]
        behv_valid  = behv_valid_dict[sess_name]
        behv_test   = behv_test_dict[sess_name]

        if decoder_type == 'nonlinear':
            decoder_params['input_dim'] = z_hat_train[0].shape[1]
            print(f"[INFO] latent dimension {decoder_params['input_dim']}")
            decoder_params['output_dim'] = len(params['model_params']['which_behv_dims'])

        decoder = init_decoder(decoder_type, decoder_params)
        decoder.fit(z_hat_train, behv_train, z_hat_valid, behv_valid, **decoder_train_params)
        y_train_pred = decoder.predict(z_hat_train)
        y_test_pred  = decoder.predict(z_hat_test)
        results_train = decoder.score(behv_train, y_train_pred)
        results_test  = decoder.score(behv_test, y_test_pred)
        print(f"[RESULT] {sess_name} Training R2 is {results_train['R2']} and Correlation is {results_train['Corr']}")
        print(f"[RESULT] {sess_name} Testing R2 is {results_test['R2']} and Correlation is {results_test['Corr']}")
        results_train_dict[sess_name] = results_train 
        results_test_dict[sess_name]  = results_test
    decoding_results_dict = {
                                'train': results_train_dict,
                                'test':  results_test_dict                        
                            }
    return decoding_results_dict

def train_behv_decoder_uni(dataset_info, z_hat_dict, behv_dict, params):
    z_hat_train_dict = z_hat_dict['train']
    z_hat_valid_dict = z_hat_dict['valid']
    z_hat_test_dict  = z_hat_dict['test']

    behv_train_dict = behv_dict['train']
    behv_valid_dict = behv_dict['valid']
    behv_test_dict  = behv_dict['test']
    
    decoder_type = params['decoder_type']
    decoder_params       = params['decoder_params']
    decoder_train_params = params['decoder_train_params']

    print(f'[INFO] Train Uni-decoder ===>')
    ## Step 1: concate all agent together
    z_hat_train_lst_all = []
    z_hat_valid_lst_all = []
    behv_train_lst_all  = []
    behv_valid_lst_all  = []
    for i, sess_name in enumerate(dataset_info.keys()):
        z_hat_train = z_hat_train_dict[sess_name]
        z_hat_valid = z_hat_valid_dict[sess_name]
        behv_train  = behv_train_dict[sess_name]
        behv_valid  = behv_valid_dict[sess_name]
        z_hat_train_lst_all += z_hat_train 
        z_hat_valid_lst_all += z_hat_valid
        behv_train_lst_all  += behv_train
        behv_valid_lst_all  += behv_valid
    
    ## Step 2: initialize one behavior decoder
    if decoder_type == 'nonlinear':
        decoder_params['input_dim'] = z_hat_train_lst_all[0].shape[1]
        print(f"[INFO] latent dimension {decoder_params['input_dim']}")
        decoder_params['output_dim'] = len(params['model_params']['which_behv_dims'])
    
    decoder = init_decoder(decoder_type, decoder_params)

    ## Step 3: Fit with the data
    decoder.fit(z_hat_train_lst_all, behv_train_lst_all, 
                z_hat_valid_lst_all, behv_valid_lst_all,
                **decoder_train_params)
    results_train_dict = {} 
    results_test_dict  = {}
    for i, sess_name in enumerate(dataset_info.keys()):
        z_hat_train = z_hat_train_dict[sess_name]
        z_hat_valid = z_hat_valid_dict[sess_name]
        z_hat_test  = z_hat_test_dict[sess_name]

        behv_train = behv_train_dict[sess_name]
        behv_valid = behv_valid_dict[sess_name]
        behv_test  = behv_test_dict[sess_name]

        y_train_pred = decoder.predict(z_hat_train)
        y_test_pred  = decoder.predict(z_hat_test)

        results_train = decoder.score(behv_train, y_train_pred)
        results_test  = decoder.score(behv_test, y_test_pred)

        results_train_dict[sess_name] = results_train 
        results_test_dict[sess_name]  = results_test 

        print(f"[RESULT] {sess_name} Training R2 is {results_train['R2']} and Correlation is {results_train['Corr']}")
        print(f"[RESULT] {sess_name} Testing R2 is {results_test['R2']} and Correlation is {results_test['Corr']}")

    decoder_results_dict = {
                            'train': results_train_dict,
                            'test' : results_test_dict
                            }
    return decoder_results_dict

def get_supbehv_decoder(dataset_info, behv_dict, behv_hat_dict, params):
    behv_train_dict = behv_dict['train']
    behv_valid_dict = behv_dict['valid']
    behv_test_dict  = behv_dict['test']

    behv_hat_train_dict = behv_hat_dict['train']
    behv_hat_valid_dict = behv_hat_dict['valid']
    behv_hat_test_dict  = behv_hat_dict['test']

    decoder_type   = params['decoder_type']      
    decoder_params = params['decoder_params']
    print(f'[INFO] Evaluate default behavior predictions ===>')
    if decoder_type == 'nonlinear':
        decoder_params['input_dim'] = 1 # useless
        decoder_params['output_dim'] = len(params['model_params']['which_behv_dims'])

    decoder = init_decoder(decoder_type, decoder_params)
    results_sup_train_dict = {}
    results_sup_test_dict  = {}

    for i, sess_name in enumerate(dataset_info.keys()):
        behv_train = behv_train_dict[sess_name]
        behv_test  = behv_test_dict[sess_name]
        behv_hat_train = behv_hat_train_dict[sess_name]
        behv_hat_test  = behv_hat_test_dict[sess_name]

        results_sup_train = decoder.score(behv_train, behv_hat_train)
        results_sup_test  = decoder.score(behv_test, behv_hat_test)

        print(f"[RESULT] {sess_name} Supervised training R2 is {results_sup_train['R2']} and Correlation is {results_sup_train['Corr']}")
        print(f"[RESULT] {sess_name} Supervised testing R2 is {results_sup_test['R2']} and Correlation is {results_sup_test['Corr']}")
        results_sup_train_dict[sess_name] = results_sup_train
        results_sup_test_dict[sess_name]  = results_sup_test

    decoding_results_dict = {
                                'train': results_sup_train_dict,
                                'test' : results_sup_test_dict
                            }
    return decoding_results_dict

def train_behv_decoder(dataset_info, z_hat_dict, behv_dict, behv_hat_dict, params):
    #### Step 1: train seperate decoders
    separatedecoder_results_dict = train_behv_decoder_separate(dataset_info, z_hat_dict, behv_dict, params)
    #### Step 2: train one decoder
    unidecoder_results_dict = train_behv_decoder_uni(dataset_info, z_hat_dict, behv_dict, params)
    #### Step 3: evaluate shared-DFINE predicted behavior and original behavior
    dfinesup_results_dict = get_supbehv_decoder(dataset_info, behv_dict, behv_hat_dict, params)
    decoding_results_dict = {
                                'train_sep': separatedecoder_results_dict['train'],
                                'test_sep' : separatedecoder_results_dict['test'],
                                'train_sup': dfinesup_results_dict['train'],
                                'test_sup' : dfinesup_results_dict['test'],
                                'train_uni': unidecoder_results_dict['train'],
                                'test_uni' : unidecoder_results_dict['test']
                            }
    return decoding_results_dict

def update_csv_behv(df_train, df_test, df_index, dataset_info, decoding_results_dict):
    for i, sess_name in enumerate(dataset_info.keys()):
        df_train.loc[df_index, (f'{sess_name}', "decoder_MSE")]  = decoding_results_dict["train_sep"][sess_name]["MSE"]
        df_train.loc[df_index, (f'{sess_name}', "decoder_MAE")]  = decoding_results_dict["train_sep"][sess_name]["MAE"]
        df_train.loc[df_index, (f'{sess_name}', "decoder_R2")]   = decoding_results_dict["train_sep"][sess_name]["R2"]
        df_train.loc[df_index, (f'{sess_name}', "decoder_Corr")] = decoding_results_dict["train_sep"][sess_name]["Corr"]

        df_train.loc[df_index, (f'{sess_name}', "unidecoder_MSE")]  = decoding_results_dict["train_uni"][sess_name]["MSE"]
        df_train.loc[df_index, (f'{sess_name}', "unidecoder_MAE")]  = decoding_results_dict['train_uni'][sess_name]["MAE"]
        df_train.loc[df_index, (f'{sess_name}', "unidecoder_R2")]   = decoding_results_dict["train_uni"][sess_name]["R2"]
        df_train.loc[df_index, (f'{sess_name}', "unidecoder_Corr")] = decoding_results_dict["train_uni"][sess_name]["Corr"]
        
        df_train.loc[df_index, (f'{sess_name}', "supdecoder_MSE")] = decoding_results_dict["train_sup"][sess_name]["MSE"]
        df_train.loc[df_index, (f'{sess_name}', "supdecoder_MAE")] = decoding_results_dict["train_sup"][sess_name]["MAE"]
        df_train.loc[df_index, (f'{sess_name}', "supdecoder_R2")]  = decoding_results_dict["train_sup"][sess_name]["R2"]
        df_train.loc[df_index, (f'{sess_name}', "supdecoder_Corr")]= decoding_results_dict["train_sup"][sess_name]["Corr"]

        df_test.loc[df_index, (f'{sess_name}', "decoder_MSE")] = decoding_results_dict["test_sep"][sess_name]["MSE"]
        df_test.loc[df_index, (f'{sess_name}', "decoder_MAE")] = decoding_results_dict["test_sep"][sess_name]["MAE"]
        df_test.loc[df_index, (f'{sess_name}', "decoder_R2")]  = decoding_results_dict["test_sep"][sess_name]["R2"]
        df_test.loc[df_index, (f'{sess_name}', "decoder_Corr")]= decoding_results_dict["test_sep"][sess_name]["Corr"]

        df_test.loc[df_index, (f'{sess_name}', "unidecoder_MSE")] = decoding_results_dict["test_uni"][sess_name]["MSE"]
        df_test.loc[df_index, (f'{sess_name}', "unidecoder_MAE")] = decoding_results_dict["test_uni"][sess_name]["MAE"]
        df_test.loc[df_index, (f'{sess_name}', "unidecoder_R2")]  = decoding_results_dict["test_uni"][sess_name]["R2"]
        df_test.loc[df_index, (f'{sess_name}', "unidecoder_Corr")]= decoding_results_dict["test_uni"][sess_name]["Corr"]

        df_test.loc[df_index, (f'{sess_name}', "supdecoder_MSE")] = decoding_results_dict["test_sup"][sess_name]["MSE"]
        df_test.loc[df_index, (f'{sess_name}', "supdecoder_MAE")] = decoding_results_dict["test_sup"][sess_name]["MAE"]
        df_test.loc[df_index, (f'{sess_name}', "supdecoder_R2")]  = decoding_results_dict["test_sup"][sess_name]["R2"]
        df_test.loc[df_index, (f'{sess_name}', "supdecoder_Corr")]= decoding_results_dict["test_sup"][sess_name]["Corr"]
