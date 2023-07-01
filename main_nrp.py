import os

import json
import glob
import logging
import argparse

import torch
from utils.misc import (
    set_seed,
)
from utils.metric_logger import TensorboardLogger
from data.mpchat_nrp import MpchatClipClipNrpDataset, MpchatClipSbertNrpDataset
from models.nrp_models import ClipClipNrp, ClipSbertNrp
from modules.checkpoint import load_checkpoint_args
from modules.train_nrp import train, evaluate

from transformers import (
    AutoTokenizer,
    CLIPProcessor,
    WEIGHTS_NAME,
)

MODEL_CLASSES = {
    'clip-clip': (ClipClipNrp, MpchatClipClipNrpDataset),
    'clip-sbert': (ClipSbertNrp, MpchatClipSbertNrpDataset),
}
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()

    ## Required (or pre-defined) params
    parser.add_argument("--dialog_data_dir", default=None, type=str, required=True, help="The dialogue data dir")
    parser.add_argument("--dialog_image_data_dir", default=None, type=str, required=True, help="The dialogue image data dir")
    parser.add_argument("--persona_image_data_dir", default=None, type=str, required=True, help="The persona image data dir")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--model_type", default='clip-clip', choices=['clip-clip', 'clip-sbert'])
    parser.add_argument("--model_name_or_path", default='', type=str,
                        help="Path to pre-trained model or shortcut name")

    ## Configs
    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--num_workers", default=6, type=int)
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true', help="Whether to use 16-bit (mixed) precision instead of 32-bit")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true', help="Whether to run test on the test set.")
    parser.add_argument("--freeze_image_encoder", action='store_true', help="Whether to freeze image encoder or not")
    parser.add_argument("--freeze_text_encoder", action='store_true', help="Whether to freeze image encoder or not")
    parser.add_argument("--remove_empty_images", action='store_true', help="Whether to remove empty images or not")
    parser.add_argument("--sum_persona_images", action='store_true', help="Whether to sum persona images or not")

    # Misc: other params (model, input, etc)
    parser.add_argument("--clip_model_name", default='openai/clip-vit-base-patch32', type=str, help="CLIP model name")
    parser.add_argument("--sbert_model_name", default='sentence-transformers/multi-qa-distilbert-cos-v1', type=str, help="SBERT model name")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=4, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--max_num_responses", type=int, default=100, help="maximum number of multimodal personas")
    parser.add_argument("--max_seq_length", type=int, default=77)
    parser.add_argument("--max_num_imgs", type=int, default=5)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument('--logging_steps', type=int, default=100, help="Log every X updates steps.")
    parser.add_argument('--save_epoch', type=int, default=5, help="Save checkpoint every X epochs.")
    parser.add_argument('--save_after_epoch', type=int, default=-1, help="Save checkpoint after epoch.")

    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        raise NotImplementedError
    args.device = device

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S', level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # set seed
    set_seed(args.seed, args.n_gpu)

    # Output config
    os.makedirs(args.output_dir, exist_ok=True)

    # Load saved checkpoint
    recover_args = load_checkpoint_args(args, logger)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.sbert_model_name)
    clip_processor = CLIPProcessor.from_pretrained(args.clip_model_name)
    model_class, dataset_class = MODEL_CLASSES[args.model_type]

    # Prepare model
    model = model_class(args, clip_processor)
    if recover_args['last_checkpoint_dir'] is not None or args.model_name_or_path != '': # recovery
        model_logging = model.load_state_dict(torch.load(os.path.join(args.model_name_or_path, 'pytorch_model.bin')))
        logger.info(f"{model_logging}")

    # Freeze model
    if args.freeze_image_encoder:
        for param in model.context_image_encoder.parameters():
            param.requires_grad = False
        for param in model.persona_image_encoder.parameters():
            param.requires_grad = False

    if args.freeze_text_encoder:
        for param in model.context_text_encoder.parameters():
            param.requires_grad = False
        for param in model.persona_text_encoder.parameters():
            param.requires_grad = False
        for param in model.response_encoder.parameters():
            param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    logger.info('Total Parameters: {}'.format(total_params))

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # load eval dataset
    eval_dataset = dataset_class(args, tokenizer, clip_processor, 'val')

    # load tensorboard
    tb_log_dir = os.path.join(args.output_dir, 'train_logs')
    meters = TensorboardLogger(
        log_dir=tb_log_dir,
        delimiter="  ",
    )

    # training
    if args.do_train:
        train_dataset = dataset_class(args, tokenizer, clip_processor, 'train')
        global_step, tr_loss = train(args, train_dataset, eval_dataset, model, meters, recover_args, logger)
        logger.info("global_step = %s, average loss = %s", global_step, tr_loss)

    # test
    if args.do_test:
        test_dataset = dataset_class(args, tokenizer, clip_processor, 'test')
        checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))

        try:
            with open(os.path.join(args.output_dir, "last_checkpoint"), "r") as f:
                texts = f.read().split('\n')
                best_saved = texts[1].split('best: ')[-1].strip()
            checkpoints = [ckpt for ckpt in checkpoints if best_saved in ckpt]
        except:
            logger.info("Cannot load checkpoint!")
            pass
        logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        test_log_json = []
        for checkpoint in checkpoints:
            epoch = checkpoint.split('-')[-2]
            global_step = checkpoint.split('-')[-1]
            model.load_state_dict(torch.load(os.path.join(checkpoint, 'pytorch_model.bin')))
            model.to(args.device)
            test_scores = evaluate(args, model, test_dataset, 'test', logger, prefix=global_step)

            epoch_log = {'epoch': epoch, 'test_scores': test_scores}
            test_log_json.append(epoch_log)

            if args.local_rank in [-1, 0]:
                with open(args.output_dir + '/test_logs.json', 'w') as fp:
                    json.dump(test_log_json, fp)

    # close the tb logger
    meters.close()
    logger.info("Good Job Computer!")

if __name__ == '__main__':
    main()
