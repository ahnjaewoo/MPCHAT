import os
import json
import pickle

import torch
from torch.utils.data import (
    Dataset,
)

from utils.misc import pil_loader

class MpchatClipClipGppDataset(Dataset):
    def __init__(self,
                 args,
                 tokenizer,
                 clip_processor,
                 mode):
        super(MpchatClipClipGppDataset, self).__init__()
        assert mode in ['train', 'val', 'test']

        self.args = args
        self.clip_processor = clip_processor
        self.mode = mode
        self.examples = []

        with open(os.path.join(args.dialog_data_dir, 'mpchat_gpp.json'), 'r') as fp:
            data = json.load(fp)[f'{mode}']

        for dialog_idx, dialog in enumerate(data):
            main_author = dialog['main_author']
            turn_indices = []
            for turn_idx, author in enumerate(dialog['authors']):
                if main_author == author and dialog['gpp_grounded_persona'][turn_idx] is not None:
                    turn_indices.append(turn_idx)

            for turn_idx in turn_indices:
                context = ' '.join(dialog['messages'][:turn_idx])
                response = dialog['messages'][turn_idx]
                final_persona_sentence = f"{dialog['gpp_grounded_persona'][turn_idx]['title']}"
                persona_sentences = ' '.join([f"{x['title']}" for x in dialog['gpp_candidate_personas'][turn_idx]])
                persona_fpaths = [os.path.join(args.persona_image_data_dir, x['file_name']) for x in dialog['gpp_candidate_personas'][turn_idx]]
                if dialog['has_image']:
                    fname_context = dialog['file_name']
                    dialog_fpath = os.path.join(args.dialog_image_data_dir, fname_context)
                else:
                    dialog_fpath = ''
                final_persona_fpath = os.path.join(args.persona_image_data_dir, dialog['gpp_grounded_persona'][turn_idx]['file_name'])

                if mode == 'train':
                    self.examples.append((dialog_fpath, context, response, persona_sentences, persona_fpaths, final_persona_sentence, final_persona_fpath, mode))
                else:
                    # add response candidates
                    assert dialog['gpp_candidate_authors_candidate_personas'][turn_idx][0]['id'] == dialog['gpp_grounded_persona'][turn_idx]['id']
                    cand_authors_cand_persona_sentences = []
                    cand_authors_cand_persona_fpaths = []
                    for i in range(self.args.max_num_candidate_persona_elements):
                        cand_authors_cand_persona_fpaths.append(os.path.join(args.persona_image_data_dir, dialog['gpp_candidate_authors_candidate_personas'][turn_idx][i]['file_name']))
                        cand_authors_cand_persona_sentences.append(dialog['gpp_candidate_authors_candidate_personas'][turn_idx][i]['title'])
                    self.examples.append((dialog_fpath, context, response, persona_sentences, persona_fpaths, cand_authors_cand_persona_sentences, cand_authors_cand_persona_fpaths, 0, mode))

        print(f'num. of {mode} dataset: {len(self.examples)}')

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        mode = self.examples[index][-1]

        if mode == 'train':
            dialog_fpath, context, response, persona_sentences, persona_fpaths, final_persona_sentence, final_persona_fpath, mode = self.examples[index]

            context_inputs = self.clip_processor(text=context)
            context_input_ids = context_inputs['input_ids']
            context_attention_mask = context_inputs['attention_mask']

            if len(context_input_ids) > self.args.max_seq_length:
                context_input_ids = context_input_ids[len(context_input_ids) - self.args.max_seq_length:]
                context_attention_mask = context_attention_mask[len(context_attention_mask) - self.args.max_seq_length:]
            while len(context_input_ids) < self.args.max_seq_length:
                context_input_ids.append(self.clip_processor.tokenizer.pad_token_id)
                context_attention_mask.append(0)

            assert len(context_input_ids) == self.args.max_seq_length
            assert len(context_attention_mask) == self.args.max_seq_length

            response_inputs = self.clip_processor(text=response)
            response_input_ids = response_inputs['input_ids']
            response_attention_mask = response_inputs['attention_mask']

            if len(response_input_ids) > self.args.max_seq_length:
                response_input_ids = response_input_ids[len(response_input_ids) - self.args.max_seq_length:]
                response_attention_mask = response_attention_mask[len(response_attention_mask) - self.args.max_seq_length:]
            while len(response_input_ids) < self.args.max_seq_length:
                response_input_ids.append(self.clip_processor.tokenizer.pad_token_id)
                response_attention_mask.append(0)

            assert len(response_input_ids) == self.args.max_seq_length
            assert len(response_attention_mask) == self.args.max_seq_length

            persona_inputs = self.clip_processor(text=persona_sentences)
            persona_input_ids = persona_inputs['input_ids']
            persona_attention_mask = persona_inputs['attention_mask']

            if len(persona_input_ids) > self.args.max_seq_length:
                persona_input_ids = persona_input_ids[len(persona_input_ids) - self.args.max_seq_length:]
                persona_attention_mask = persona_attention_mask[len(persona_attention_mask) - self.args.max_seq_length:]
            while len(persona_input_ids) < self.args.max_seq_length:
                persona_input_ids.append(self.clip_processor.tokenizer.pad_token_id)
                persona_attention_mask.append(0)

            assert len(persona_input_ids) == self.args.max_seq_length
            assert len(persona_attention_mask) == self.args.max_seq_length

            final_persona_inputs = self.clip_processor(text=final_persona_sentence)
            final_persona_input_ids = final_persona_inputs['input_ids']
            final_persona_attention_mask = final_persona_inputs['attention_mask']

            if len(final_persona_input_ids) > self.args.max_seq_length:
                final_persona_input_ids = final_persona_input_ids[len(final_persona_input_ids) - self.args.max_seq_length:]
                final_persona_attention_mask = final_persona_attention_mask[len(final_persona_attention_mask) - self.args.max_seq_length:]
            while len(final_persona_input_ids) < self.args.max_seq_length:
                final_persona_input_ids.append(self.clip_processor.tokenizer.pad_token_id)
                final_persona_attention_mask.append(0)

            assert len(final_persona_input_ids) == self.args.max_seq_length
            assert len(final_persona_attention_mask) == self.args.max_seq_length

            if dialog_fpath == '':
                dialog_img_feat = torch.rand(3, self.args.img_size, self.args.img_size)
                dialog_img_mask = 0
            else:
                dialog_img = pil_loader(dialog_fpath)
                dialog_img_feat = self.clip_processor(images=dialog_img, return_tensors='pt')['pixel_values'].squeeze(0)
                dialog_img_mask = 1

            final_persona_img = pil_loader(final_persona_fpath)
            final_persona_img_feat = self.clip_processor(images=final_persona_img, return_tensors='pt')['pixel_values'].squeeze(0)
            assert final_persona_img_feat.shape == torch.Size([3, self.args.img_size, self.args.img_size])

            persona_imgs = [pil_loader(x) for x in persona_fpaths]
            persona_img_feats = self.clip_processor(images=persona_imgs, return_tensors='pt')['pixel_values']
            persona_img_mask = [1 for _ in persona_fpaths]
            if len(persona_fpaths) < self.args.max_num_imgs:
                empty_img_feats = torch.stack([torch.rand(3, self.args.img_size, self.args.img_size) for _ in range(self.args.max_num_imgs - len(persona_fpaths))])
                persona_img_feats = torch.cat([persona_img_feats, empty_img_feats], dim=0)
                persona_img_mask += [0 for _ in range(self.args.max_num_imgs - len(persona_fpaths))]
            assert persona_img_feats.shape == torch.Size([self.args.max_num_imgs, 3, self.args.img_size, self.args.img_size])

            feature = [
                torch.as_tensor(context_input_ids, dtype=torch.long),
                torch.as_tensor(context_attention_mask, dtype=torch.long),
                torch.as_tensor(response_input_ids, dtype=torch.long),
                torch.as_tensor(response_attention_mask, dtype=torch.long),
                torch.as_tensor(persona_input_ids, dtype=torch.long),
                torch.as_tensor(persona_attention_mask, dtype=torch.long),
                torch.as_tensor(final_persona_input_ids, dtype=torch.long),
                torch.as_tensor(final_persona_attention_mask, dtype=torch.long),
                dialog_img_feat,
                persona_img_feats,
                final_persona_img_feat,
                torch.as_tensor(dialog_img_mask, dtype=torch.long),
                torch.as_tensor(persona_img_mask, dtype=torch.long),
            ]
        else:
            dialog_fpath, context, response, persona_sentences, persona_fpaths, cand_final_persona_sentences, cand_final_persona_fpaths, label_idx, mode = self.examples[index]

            context_inputs = self.clip_processor(text=context)
            context_input_ids = context_inputs['input_ids']
            context_attention_mask = context_inputs['attention_mask']

            if len(context_input_ids) > self.args.max_seq_length:
                context_input_ids = context_input_ids[len(context_input_ids) - self.args.max_seq_length:]
                context_attention_mask = context_attention_mask[len(context_attention_mask) - self.args.max_seq_length:]
            while len(context_input_ids) < self.args.max_seq_length:
                context_input_ids.append(self.clip_processor.tokenizer.pad_token_id)
                context_attention_mask.append(0)

            assert len(context_input_ids) == self.args.max_seq_length
            assert len(context_attention_mask) == self.args.max_seq_length

            response_inputs = self.clip_processor(text=response)
            response_input_ids = response_inputs['input_ids']
            response_attention_mask = response_inputs['attention_mask']

            if len(response_input_ids) > self.args.max_seq_length:
                response_input_ids = response_input_ids[len(response_input_ids) - self.args.max_seq_length:]
                response_attention_mask = response_attention_mask[len(response_attention_mask) - self.args.max_seq_length:]
            while len(response_input_ids) < self.args.max_seq_length:
                response_input_ids.append(self.clip_processor.tokenizer.pad_token_id)
                response_attention_mask.append(0)

            assert len(response_input_ids) == self.args.max_seq_length
            assert len(response_attention_mask) == self.args.max_seq_length

            persona_inputs = self.clip_processor(text=persona_sentences)
            persona_input_ids = persona_inputs['input_ids']
            persona_attention_mask = persona_inputs['attention_mask']

            if len(persona_input_ids) > self.args.max_seq_length:
                persona_input_ids = persona_input_ids[len(persona_input_ids) - self.args.max_seq_length:]
                persona_attention_mask = persona_attention_mask[len(persona_attention_mask) - self.args.max_seq_length:]
            while len(persona_input_ids) < self.args.max_seq_length:
                persona_input_ids.append(self.clip_processor.tokenizer.pad_token_id)
                persona_attention_mask.append(0)

            assert len(persona_input_ids) == self.args.max_seq_length
            assert len(persona_attention_mask) == self.args.max_seq_length

            final_persona_inputs = self.clip_processor.tokenizer(cand_final_persona_sentences,
                                                            truncation=True,
                                                            padding='max_length',
                                                            max_length=self.args.max_seq_length)
            final_persona_input_ids = torch.as_tensor(final_persona_inputs['input_ids'], dtype=torch.long)
            final_persona_attention_mask = torch.as_tensor(final_persona_inputs['attention_mask'], dtype=torch.long)

            assert final_persona_input_ids.shape == torch.Size([self.args.max_num_candidate_persona_elements, self.args.max_seq_length])
            assert final_persona_attention_mask.shape == torch.Size([self.args.max_num_candidate_persona_elements, self.args.max_seq_length])

            if dialog_fpath == '':
                dialog_img_feat = torch.rand(3, self.args.img_size, self.args.img_size)
                dialog_img_mask = 0
            else:
                dialog_img = pil_loader(dialog_fpath)
                dialog_img_feat = self.clip_processor(images=dialog_img, return_tensors='pt')['pixel_values'].squeeze(0)
                dialog_img_mask = 1

            final_persona_imgs = [pil_loader(x) for x in cand_final_persona_fpaths]
            final_persona_img_feats = self.clip_processor(images=final_persona_imgs, return_tensors='pt')['pixel_values']
            assert final_persona_img_feats.shape == torch.Size([self.args.max_num_candidate_persona_elements, 3, self.args.img_size, self.args.img_size])

            persona_imgs = [pil_loader(x) for x in persona_fpaths]
            persona_img_feats = self.clip_processor(images=persona_imgs, return_tensors='pt')['pixel_values']
            persona_img_mask = [1 for _ in persona_fpaths]
            if len(persona_fpaths) < self.args.max_num_imgs:
                empty_img_feats = torch.stack([torch.rand(3, self.args.img_size, self.args.img_size) for _ in range(self.args.max_num_imgs - len(persona_fpaths))])
                persona_img_feats = torch.cat([persona_img_feats, empty_img_feats], dim=0)
                persona_img_mask += [0 for _ in range(self.args.max_num_imgs - len(persona_fpaths))]
            assert persona_img_feats.shape == torch.Size([self.args.max_num_imgs, 3, self.args.img_size, self.args.img_size])

            feature = [
                torch.as_tensor(context_input_ids, dtype=torch.long),
                torch.as_tensor(context_attention_mask, dtype=torch.long),
                torch.as_tensor(response_input_ids, dtype=torch.long),
                torch.as_tensor(response_attention_mask, dtype=torch.long),
                torch.as_tensor(persona_input_ids, dtype=torch.long),
                torch.as_tensor(persona_attention_mask, dtype=torch.long),
                torch.as_tensor(final_persona_input_ids, dtype=torch.long),
                torch.as_tensor(final_persona_attention_mask, dtype=torch.long),
                dialog_img_feat,
                persona_img_feats,
                final_persona_img_feats,
                torch.as_tensor(label_idx, dtype=torch.long),
                torch.as_tensor(dialog_img_mask, dtype=torch.long),
                torch.as_tensor(persona_img_mask, dtype=torch.long),
            ]
        return feature

class MpchatClipSbertGppDataset(Dataset):
    def __init__(self,
                 args,
                 tokenizer,
                 clip_processor,
                 mode):
        super(MpchatClipSbertGppDataset, self).__init__()
        assert mode in ['train', 'val', 'test']

        self.args = args
        self.tokenizer = tokenizer
        self.clip_processor = clip_processor
        self.mode = mode
        self.examples = []

        with open(os.path.join(args.dialog_data_dir, 'mpchat_gpp.json'), 'r') as fp:
            data = json.load(fp)[f'{mode}']

        for dialog_idx, dialog in enumerate(data):
            main_author = dialog['main_author']
            turn_indices = []
            for turn_idx, author in enumerate(dialog['authors']):
                if main_author == author and dialog['gpp_grounded_persona'][turn_idx] is not None:
                    turn_indices.append(turn_idx)

            for turn_idx in turn_indices:
                context = ' '.join(dialog['messages'][:turn_idx])
                response = dialog['messages'][turn_idx]
                final_persona_sentence = f"{dialog['gpp_grounded_persona'][turn_idx]['title']}"
                persona_sentences = ' '.join([f"{x['title']}" for x in dialog['gpp_candidate_personas'][turn_idx]])
                persona_fpaths = [os.path.join(args.persona_image_data_dir, x['file_name']) for x in dialog['gpp_candidate_personas'][turn_idx]]
                if dialog['has_image']:
                    fname_context = dialog['file_name']
                    dialog_fpath = os.path.join(args.dialog_image_data_dir, fname_context)
                else:
                    dialog_fpath = ''
                final_persona_fpath = os.path.join(args.persona_image_data_dir, dialog['gpp_grounded_persona'][turn_idx]['file_name'])

                if mode == 'train':
                    self.examples.append((dialog_fpath, context, response, persona_sentences, persona_fpaths, final_persona_sentence, final_persona_fpath, mode))
                else:
                    # add response candidates
                    assert dialog['gpp_candidate_authors_candidate_personas'][turn_idx][0]['id'] == dialog['gpp_grounded_persona'][turn_idx]['id']
                    cand_authors_cand_persona_sentences = []
                    cand_authors_cand_persona_fpaths = []
                    for i in range(self.args.max_num_candidate_persona_elements):
                        cand_authors_cand_persona_fpaths.append(os.path.join(args.persona_image_data_dir, dialog['gpp_candidate_authors_candidate_personas'][turn_idx][i]['file_name']))
                        cand_authors_cand_persona_sentences.append(dialog['gpp_candidate_authors_candidate_personas'][turn_idx][i]['title'])
                    self.examples.append((dialog_fpath, context, response, persona_sentences, persona_fpaths, cand_authors_cand_persona_sentences, cand_authors_cand_persona_fpaths, 0, mode))

        print(f'num. of {mode} dataset: {len(self.examples)}')

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        mode = self.examples[index][-1]

        if mode == 'train':
            dialog_fpath, context, response, persona_sentences, persona_fpaths, final_persona_sentence, final_persona_fpath, mode = self.examples[index]

            context_inputs = self.tokenizer(context,
                                            truncation=True,
                                            padding='max_length',
                                            max_length=self.args.max_seq_length)
            context_input_ids = context_inputs['input_ids']
            context_attention_mask = context_inputs['attention_mask']

            assert len(context_input_ids) == self.args.max_seq_length
            assert len(context_attention_mask) == self.args.max_seq_length

            response_inputs = self.tokenizer(response,
                                            truncation=True,
                                            padding='max_length',
                                            max_length=self.args.max_seq_length)
            response_input_ids = response_inputs['input_ids']
            response_attention_mask = response_inputs['attention_mask']

            assert len(response_input_ids) == self.args.max_seq_length
            assert len(response_attention_mask) == self.args.max_seq_length

            persona_inputs = self.tokenizer(persona_sentences,
                                            truncation=True,
                                            padding='max_length',
                                            max_length=self.args.max_seq_length)
            persona_input_ids = persona_inputs['input_ids']
            persona_attention_mask = persona_inputs['attention_mask']

            assert len(persona_input_ids) == self.args.max_seq_length
            assert len(persona_attention_mask) == self.args.max_seq_length

            final_persona_inputs = self.tokenizer(final_persona_sentence,
                                            truncation=True,
                                            padding='max_length',
                                            max_length=self.args.max_seq_length)
            final_persona_input_ids = final_persona_inputs['input_ids']
            final_persona_attention_mask = final_persona_inputs['attention_mask']

            assert len(final_persona_input_ids) == self.args.max_seq_length
            assert len(final_persona_attention_mask) == self.args.max_seq_length

            if dialog_fpath == '':
                dialog_img_feat = torch.rand(3, self.args.img_size, self.args.img_size)
                dialog_img_mask = 0
            else:
                dialog_img = pil_loader(dialog_fpath)
                dialog_img_feat = self.clip_processor(images=dialog_img, return_tensors='pt')['pixel_values'].squeeze(0)
                dialog_img_mask = 1

            final_persona_img = pil_loader(final_persona_fpath)
            final_persona_img_feat = self.clip_processor(images=final_persona_img, return_tensors='pt')['pixel_values'].squeeze(0)
            assert final_persona_img_feat.shape == torch.Size([3, self.args.img_size, self.args.img_size])

            persona_imgs = [pil_loader(x) for x in persona_fpaths]
            persona_img_feats = self.clip_processor(images=persona_imgs, return_tensors='pt')['pixel_values']
            persona_img_mask = [1 for _ in persona_fpaths]
            if len(persona_fpaths) < self.args.max_num_imgs:
                empty_img_feats = torch.stack([torch.rand(3, self.args.img_size, self.args.img_size) for _ in range(self.args.max_num_imgs - len(persona_fpaths))])
                persona_img_feats = torch.cat([persona_img_feats, empty_img_feats], dim=0)
                persona_img_mask += [0 for _ in range(self.args.max_num_imgs - len(persona_fpaths))]
            assert persona_img_feats.shape == torch.Size([self.args.max_num_imgs, 3, self.args.img_size, self.args.img_size])

            feature = [
                torch.as_tensor(context_input_ids, dtype=torch.long),
                torch.as_tensor(context_attention_mask, dtype=torch.long),
                torch.as_tensor(response_input_ids, dtype=torch.long),
                torch.as_tensor(response_attention_mask, dtype=torch.long),
                torch.as_tensor(persona_input_ids, dtype=torch.long),
                torch.as_tensor(persona_attention_mask, dtype=torch.long),
                torch.as_tensor(final_persona_input_ids, dtype=torch.long),
                torch.as_tensor(final_persona_attention_mask, dtype=torch.long),
                dialog_img_feat,
                persona_img_feats,
                final_persona_img_feat,
                torch.as_tensor(dialog_img_mask, dtype=torch.long),
                torch.as_tensor(persona_img_mask, dtype=torch.long),
            ]
        else:
            dialog_fpath, context, response, persona_sentences, persona_fpaths, cand_final_persona_sentences, cand_final_persona_fpaths, label_idx, mode = self.examples[index]

            context_inputs = self.tokenizer(context,
                                            truncation=True,
                                            padding='max_length',
                                            max_length=self.args.max_seq_length)
            context_input_ids = context_inputs['input_ids']
            context_attention_mask = context_inputs['attention_mask']

            assert len(context_input_ids) == self.args.max_seq_length
            assert len(context_attention_mask) == self.args.max_seq_length

            response_inputs = self.tokenizer(response,
                                            truncation=True,
                                            padding='max_length',
                                            max_length=self.args.max_seq_length)
            response_input_ids = response_inputs['input_ids']
            response_attention_mask = response_inputs['attention_mask']

            assert len(response_input_ids) == self.args.max_seq_length
            assert len(response_attention_mask) == self.args.max_seq_length

            persona_inputs = self.tokenizer(persona_sentences,
                                            truncation=True,
                                            padding='max_length',
                                            max_length=self.args.max_seq_length)
            persona_input_ids = persona_inputs['input_ids']
            persona_attention_mask = persona_inputs['attention_mask']

            assert len(persona_input_ids) == self.args.max_seq_length
            assert len(persona_attention_mask) == self.args.max_seq_length

            final_persona_inputs = self.tokenizer(cand_final_persona_sentences,
                                                  truncation=True,
                                                  padding='max_length',
                                                  max_length=self.args.max_seq_length)
            final_persona_input_ids = torch.as_tensor(final_persona_inputs['input_ids'], dtype=torch.long)
            final_persona_attention_mask = torch.as_tensor(final_persona_inputs['attention_mask'], dtype=torch.long)

            assert final_persona_input_ids.shape == torch.Size([self.args.max_num_candidate_persona_elements, self.args.max_seq_length])
            assert final_persona_attention_mask.shape == torch.Size([self.args.max_num_candidate_persona_elements, self.args.max_seq_length])

            if dialog_fpath == '':
                dialog_img_feat = torch.rand(3, self.args.img_size, self.args.img_size)
                dialog_img_mask = 0
            else:
                dialog_img = pil_loader(dialog_fpath)
                dialog_img_feat = self.clip_processor(images=dialog_img, return_tensors='pt')['pixel_values'].squeeze(0)
                dialog_img_mask = 1

            final_persona_imgs = [pil_loader(x) for x in cand_final_persona_fpaths]
            final_persona_img_feats = self.clip_processor(images=final_persona_imgs, return_tensors='pt')['pixel_values']
            assert final_persona_img_feats.shape == torch.Size([self.args.max_num_candidate_persona_elements, 3, self.args.img_size, self.args.img_size])

            persona_imgs = [pil_loader(x) for x in persona_fpaths]
            persona_img_feats = self.clip_processor(images=persona_imgs, return_tensors='pt')['pixel_values']
            persona_img_mask = [1 for _ in persona_fpaths]
            if len(persona_fpaths) < self.args.max_num_imgs:
                empty_img_feats = torch.stack([torch.rand(3, self.args.img_size, self.args.img_size) for _ in range(self.args.max_num_imgs - len(persona_fpaths))])
                persona_img_feats = torch.cat([persona_img_feats, empty_img_feats], dim=0)
                persona_img_mask += [0 for _ in range(self.args.max_num_imgs - len(persona_fpaths))]
            assert persona_img_feats.shape == torch.Size([self.args.max_num_imgs, 3, self.args.img_size, self.args.img_size])

            feature = [
                torch.as_tensor(context_input_ids, dtype=torch.long),
                torch.as_tensor(context_attention_mask, dtype=torch.long),
                torch.as_tensor(response_input_ids, dtype=torch.long),
                torch.as_tensor(response_attention_mask, dtype=torch.long),
                torch.as_tensor(persona_input_ids, dtype=torch.long),
                torch.as_tensor(persona_attention_mask, dtype=torch.long),
                torch.as_tensor(final_persona_input_ids, dtype=torch.long),
                torch.as_tensor(final_persona_attention_mask, dtype=torch.long),
                dialog_img_feat,
                persona_img_feats,
                final_persona_img_feats,
                torch.as_tensor(label_idx, dtype=torch.long),
                torch.as_tensor(dialog_img_mask, dtype=torch.long),
                torch.as_tensor(persona_img_mask, dtype=torch.long),
            ]
        return feature
