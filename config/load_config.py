import yaml 
import os

import numpy as np

def load_model_config(args):
    with open(args.model_config, 'r') as f:
        config = yaml.safe_load(f)
    
    return (
        config.get('model_params', {}),  # Model parameters dictionary
        config.get('train_params', {})   # Training parameters dictionary
    )

def load_data_config(args):
    with open(args.data_config) as f:
        config = yaml.safe_load(f)
        data_params = yaml.safe_load(f)
    
    data_config, data_params = (config.get('data_config',{}), config.get('data_params', {}))
    # Override brain area if provided in command line
    if args.brain_area:
        data_config['brain_area'] = args.brain_area.lower()
    if args.data_folder:
        data_config['data_folder'] = args.data_folder
    
    # Generate full paths
    data_path_lst = []
    for subj in data_config['subjects']:
        for date in subj['dates']:
            path = os.path.join(
                data_config['data_folder'],
                data_config['path_template'].format(
                    subject=subj['subject'],
                    date=date,
                    brain_area=data_config['brain_area']
                )
            )
            data_path_lst.append(path)
    data_config['data_path_lst'] = data_path_lst
    
    return data_config, data_params

def load_decoder_config(args):
    with open(args.decoder_config, 'r') as file:
        config = yaml.safe_load(file)
    
    # Convert the YAML structure to the desired Python format
    decoder_params = {}
    for key, value in config['decoder'].items():
        decoder_params[key] = value
    
    # Handle the params_grid specially to create the numpy array
    decoder_train_params = {}
    for key, value in config['decoder_train'].items():
        if key == 'params_grid':
            grid_config = config['decoder_train']['params_grid']['alpha']
            decoder_train_params['params_grid'] = {'alpha': np.arange(grid_config['start'], grid_config['stop'], grid_config['step'])}
        else:
            decoder_train_params[key] = value
    
    return decoder_params, decoder_train_params