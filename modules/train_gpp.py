import os

import json
import time
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import torch
from torch.utils.data import (
    RandomSampler,
    SequentialSampler,
    DataLoader,
)
from utils.misc import compute_metrics_from_logits

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)

def train(args, train_dataset, eval_dataset, model, meters, recover_args, logger):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, num_workers=args.num_workers, sampler=train_sampler, batch_size=args.train_batch_size, pin_memory=True)
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = []
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if
                    not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay}
    ]

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                     num_warmup_steps=args.warmup_steps,
                                     num_training_steps=t_total)

    if recover_args['global_step'] > 0 and os.path.isfile(os.path.join(recover_args['last_checkpoint_dir'], 'optimizer.pth')): # recovery
        last_checkpoint_dir = recover_args['last_checkpoint_dir']
        logger.info(
            "Load optimizer from {}".format(last_checkpoint_dir))
        optimizer_to_load = torch.load(
            os.path.join(last_checkpoint_dir, 'optimizer.pth'),
            map_location=torch.device("cpu"))
        optimizer.load_state_dict(optimizer_to_load.pop("optimizer"))
        scheduler.load_state_dict(optimizer_to_load.pop("scheduler"))

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = recover_args['global_step']
    start_epoch = recover_args['epoch'] + 1 if global_step > 0 else 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()

    best_scores = {
        'epoch': 0,
        'global_step': 0,
        'scores': {'recall@1': 0.0}
    }
    if recover_args['last_best_score'] is not None:
        best_scores = recover_args['last_best_score']

    log_json = []
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(start_epoch, int(args.num_train_epochs)):
        t_start = time.time()
        tbar = tqdm(train_dataloader, ncols=70)
        for step, batch in enumerate(tbar):
            tbar.set_description(f'Training loss = {logging_loss}')
            model.train()

            context_input_ids = batch[0].to(args.device, non_blocking=True)
            context_attention_mask = batch[1].to(args.device, non_blocking=True)
            response_input_ids = batch[2].to(args.device, non_blocking=True)
            response_attention_mask = batch[3].to(args.device, non_blocking=True)
            persona_input_ids = batch[4].to(args.device, non_blocking=True)
            persona_attention_mask = batch[5].to(args.device, non_blocking=True)
            final_persona_input_ids = batch[6].to(args.device, non_blocking=True)
            final_persona_attention_mask = batch[7].to(args.device, non_blocking=True)
            dialog_img_feat = batch[8].to(args.device, non_blocking=True)
            persona_img_feats = batch[9].to(args.device, non_blocking=True)
            final_persona_img_feats = batch[10].to(args.device, non_blocking=True)
            dialog_img_mask = batch[11].to(args.device, non_blocking=True)
            persona_img_mask = batch[12].to(args.device, non_blocking=True)

            if args.fp16:
                with torch.cuda.amp.autocast(enabled=True):
                    outputs = model(
                        context_input_ids=context_input_ids,
                        context_attention_mask=context_attention_mask,
                        response_input_ids=response_input_ids,
                        response_attention_mask=response_attention_mask,
                        persona_input_ids=persona_input_ids,
                        persona_attention_mask=persona_attention_mask,
                        final_persona_input_ids=final_persona_input_ids,
                        final_persona_attention_mask=final_persona_attention_mask,
                        dialog_img_feat=dialog_img_feat,
                        persona_img_feats=persona_img_feats,
                        final_persona_img_feats=final_persona_img_feats,
                        dialog_img_mask=dialog_img_mask,
                        persona_img_mask=persona_img_mask,
                        mode='train',
                    )
                    loss = outputs[0]
            else:
                outputs = model(
                    context_input_ids=context_input_ids,
                    context_attention_mask=context_attention_mask,
                    response_input_ids=response_input_ids,
                    response_attention_mask=response_attention_mask,
                    persona_input_ids=persona_input_ids,
                    persona_attention_mask=persona_attention_mask,
                    final_persona_input_ids=final_persona_input_ids,
                    final_persona_attention_mask=final_persona_attention_mask,
                    dialog_img_feat=dialog_img_feat,
                    persona_img_feats=persona_img_feats,
                    final_persona_img_feats=final_persona_img_feats,
                    dialog_img_mask=dialog_img_mask,
                    persona_img_mask=persona_img_mask,
                    mode='train',
                )
                loss = outputs[0]

            if args.n_gpu > 1: loss = loss.mean()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            logging_loss = round(loss.item(), 5)
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # do gradient clipping
                if args.max_grad_norm > 0:
                   torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scheduler.step()  # Update learning rate schedule
                if args.fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                model.zero_grad()
                global_step += 1

                # update tensorboard
                meters.update_metrics({'batch_metrics': {'loss': loss}})
                meters.update_params({'params': {'lr': optimizer.param_groups[0]['lr']}})

                if args.logging_steps > 0 and (global_step + 1) % args.logging_steps == 0:
                    meters.get_logs(global_step+1)

        # Evaluation
        logger.info("Epoch: %d, global_step: %d" % (epoch, global_step))
        eval_scores = evaluate(args, model, eval_dataset, 'val', logger, prefix=global_step)

        # Select recall@1 score as metric
        if eval_scores['recall@1'] > best_scores['scores']['recall@1']:
            best_scores['scores'] = eval_scores
            best_scores['epoch'] = epoch
            best_scores['global_step'] = global_step

        # save checkpoints
        if (args.local_rank in [-1, 0]) and (args.save_epoch>0 and epoch % args.save_epoch == 0) and (epoch > args.save_after_epoch):
            output_dir = os.path.join(args.output_dir, 'checkpoint-{}-{}'.format(epoch, global_step))
            if not os.path.exists(output_dir): os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
            optimizer_to_save = {
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }

            save_num = 0
            while (save_num < 3):
                try:
                    logger.info("Saving model attempt: {}".format(save_num))
                    torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    torch.save(optimizer_to_save, os.path.join(output_dir, 'optimizer.pth'))
                    save_file = os.path.join(args.output_dir, 'last_checkpoint')
                    with open(save_file, 'w') as f:
                        f.write('checkpoint-{}-{}/pytorch_model.bin\n'.format(epoch, global_step))
                        f.write(f'best: checkpoint-{best_scores["epoch"]}-{best_scores["global_step"]}\n')
                        json.dump(best_scores, f)
                    break
                except:
                    save_num += 1
            logger.info("Saving model checkpoint {0} to {1}".format(epoch, output_dir))

        epoch_log = {'epoch': epoch, 'eval_scores': eval_scores, 'best_scores': best_scores['scores']}
        log_json.append(epoch_log)

        logger.info("PROGRESS: {}%".format(round(100*(epoch + 1) / args.num_train_epochs, 4)))

        if args.local_rank in [-1, 0]:
            with open(args.output_dir + '/eval_logs.json', 'w') as fp:
                json.dump(log_json, fp)

            t_end = time.time()
            logger.info('Epoch: %d, Train Time: %.3f' % (epoch, t_end - t_start))

    return global_step, tr_loss / global_step

def evaluate(args, model, eval_dataset, mode, logger, prefix=''):
    t_start = time.time()
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    test_sampler = SequentialSampler(eval_dataset)
    test_dataloader = DataLoader(eval_dataset, num_workers=args.num_workers, sampler=test_sampler, batch_size=args.eval_batch_size, pin_memory=True)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    results_dict = defaultdict(list)
    for batch in tqdm(test_dataloader, ncols=70):
        model.eval()

        context_input_ids = batch[0].to(args.device, non_blocking=True)
        context_attention_mask = batch[1].to(args.device, non_blocking=True)
        response_input_ids = batch[2].to(args.device, non_blocking=True)
        response_attention_mask = batch[3].to(args.device, non_blocking=True)
        persona_input_ids = batch[4].to(args.device, non_blocking=True)
        persona_attention_mask = batch[5].to(args.device, non_blocking=True)
        final_persona_input_ids = batch[6].to(args.device, non_blocking=True)
        final_persona_attention_mask = batch[7].to(args.device, non_blocking=True)
        dialog_img_feat = batch[8].to(args.device, non_blocking=True)
        persona_img_feats = batch[9].to(args.device, non_blocking=True)
        final_persona_img_feats = batch[10].to(args.device, non_blocking=True)
        labels = batch[11].to(args.device, non_blocking=True)
        dialog_img_mask = batch[12].to(args.device, non_blocking=True)
        persona_img_mask = batch[13].to(args.device, non_blocking=True)

        with torch.no_grad():
            loss, logits = model(
                context_input_ids=context_input_ids,
                context_attention_mask=context_attention_mask,
                response_input_ids=response_input_ids,
                response_attention_mask=response_attention_mask,
                persona_input_ids=persona_input_ids,
                persona_attention_mask=persona_attention_mask,
                final_persona_input_ids=final_persona_input_ids,
                final_persona_attention_mask=final_persona_attention_mask,
                dialog_img_feat=dialog_img_feat,
                persona_img_feats=persona_img_feats,
                final_persona_img_feats=final_persona_img_feats,
                labels=labels,
                dialog_img_mask=dialog_img_mask,
                persona_img_mask=persona_img_mask,
                mode=mode,
            )
            results_dict['loss'].append(loss.cpu().detach().numpy())
            results_dict['logits'].append(logits.cpu().detach().numpy())
            results_dict['labels'].append(labels.cpu().detach().numpy())

    for key, value in results_dict.items():
        if results_dict[key][0].shape == ():
            results_dict[key] = np.array(value)
        else:
            results_dict[key] = np.concatenate(value, axis=0)

    recall, mrr = compute_metrics_from_logits(torch.tensor(results_dict['logits']),
                                              torch.tensor(results_dict['labels']))

    total_scores = {
        'loss': round(np.mean(results_dict['loss']).item(), 4),
        'mrr': round(mrr, 4),
    }
    for k,v in recall.items():
        total_scores[k] = round(v, 4)

    logger.info("Eval Results:")
    logger.info(f'Eval Score: {total_scores}')

    t_end = time.time()
    logger.info('Eval Time Cost: %.3f' % (t_end - t_start))

    return total_scores
