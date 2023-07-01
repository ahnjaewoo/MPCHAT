import os
import json

from transformers import WEIGHTS_NAME

def load_checkpoint_args(args, logger):
    recover_args = {'global_step': 0, 'step': 0, 'last_checkpoint_dir': None,
                    'last_best_checkpoint_dir': None, 'last_best_score': None}

    if os.path.exists(args.output_dir):
        save_file = os.path.join(args.output_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                texts = f.read().split('\n')
                last_saved = texts[0]
                last_saved = last_saved.strip()
                last_best_saved = texts[1].split('best: ')[-1].strip()
                last_best_score = json.loads(texts[2])

        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        if last_saved:
            folder_name = os.path.splitext(last_saved.split('/')[0])[0] # in the form of checkpoint-00001 or checkpoint-00001/pytorch_model.bin
            recover_args['last_checkpoint_dir'] = os.path.join(args.output_dir, folder_name)
            recover_args['epoch'] = int(folder_name.split('-')[1])
            recover_args['global_step'] = int(folder_name.split('-')[2])
            recover_args['last_best_checkpoint_dir'] = os.path.join(args.output_dir, last_best_saved)
            recover_args['last_best_score'] = last_best_score
            assert os.path.isfile(os.path.join(recover_args['last_checkpoint_dir'], WEIGHTS_NAME)), "Last_checkpoint detected, but file not found!"

    if recover_args['last_checkpoint_dir'] is not None: # recovery
        args.model_name_or_path = recover_args['last_checkpoint_dir']
        logger.info(" -> Recovering model from {}".format(recover_args['last_checkpoint_dir']))

    return recover_args
