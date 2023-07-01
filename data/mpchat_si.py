import os
import json
import pickle

import torch
from torch.utils.data import (
    Dataset,
)

from utils.misc import pil_loader

class MpchatClipClipSiDataset(Dataset):
    def __init__(self,
                 args,
                 tokenizer,
                 clip_processor,
                 mode):
        super(MpchatClipClipSiDataset, self).__init__()
        assert mode in ['train', 'val', 'test']

        self.args = args
        self.clip_processor = clip_processor
        self.mode = mode
        self.examples = []

        with open(os.path.join(args.dialog_data_dir, 'mpchat_si.json'), 'r') as fp:
            data = json.load(fp)[f'{mode}']

        for dialog_idx, dialog in enumerate(data):
            main_author = dialog['main_author']
            context = ' '.join(dialog['messages'][:-1])
            response = dialog['messages'][-1]
            persona_sentences = ' '.join([f"{x['title']}" for x in dialog['si_main_author_candidate_personas']])
            persona_fpaths = [os.path.join(args.persona_image_data_dir, x['file_name']) for x in dialog['si_main_author_candidate_personas']]
            if dialog['has_image']:
                fname_context = dialog['file_name']
                dialog_fpath = os.path.join(args.dialog_image_data_dir, fname_context)
            else:
                dialog_fpath = ''

            if mode == 'train':
                self.examples.append((context, response, dialog_fpath, persona_sentences, persona_fpaths, mode))
            else:
                # add other authors persona candidates
                assert persona_sentences == ' '.join([x['title'] for x in dialog['si_candidate_authors_candidate_personas'][0]])
                cand_authors_cand_persona_sentences_list = []
                cand_authors_cand_persona_images_list = []
                for i in range(self.args.max_num_candidate_authors):
                    cand_authors_cand_persona_sentences_list.append(' '.join([x['title'] for x in dialog['si_candidate_authors_candidate_personas'][i]]))
                    cand_authors_cand_persona_images_list.append([os.path.join(args.persona_image_data_dir, x['file_name']) for x in dialog['si_candidate_authors_candidate_personas'][i]])
                self.examples.append((context, response, dialog_fpath, cand_authors_cand_persona_sentences_list, cand_authors_cand_persona_images_list, 0, mode))

        print(f'num. of {mode} dataset: {len(self.examples)}')

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        mode = self.examples[index][-1]

        if mode == 'train':
            context_text, response, dialog_fpath, persona_sentences, persona_fpaths, mode = self.examples[index]

            context_inputs = self.clip_processor(text=context_text)
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

            if dialog_fpath == '':
                dialog_img_feat = torch.rand(3, self.args.img_size, self.args.img_size)
                dialog_img_mask = 0
            else:
                dialog_img = pil_loader(dialog_fpath)
                dialog_img_feat = self.clip_processor(images=dialog_img, return_tensors='pt')['pixel_values'].squeeze(0)
                dialog_img_mask = 1
            assert dialog_img_feat.shape == torch.Size([3, self.args.img_size, self.args.img_size])

            persona_imgs = [pil_loader(x) for x in persona_fpaths]
            persona_img_feats = self.clip_processor(images=persona_imgs, return_tensors='pt')['pixel_values']
            persona_img_mask = [1 for _ in persona_fpaths]
            if len(persona_fpaths) < self.args.max_num_imgs:
                empty_img_feats = torch.stack([torch.rand(3, self.args.img_size, self.args.img_size) for _ in range(self.args.max_num_imgs - len(persona_fpaths))])
                persona_img_feats = torch.cat([persona_img_feats, empty_img_feats], dim=0)
                persona_img_mask += [0 for _ in range(self.args.max_num_imgs - len(persona_fpaths))]
            assert persona_img_feats.shape == torch.Size([self.args.max_num_imgs, 3, self.args.img_size, self.args.img_size])
            assert len(persona_img_mask) == self.args.max_num_imgs

            feature = [
                torch.as_tensor(context_input_ids, dtype=torch.long),
                torch.as_tensor(context_attention_mask, dtype=torch.long),
                torch.as_tensor(response_input_ids, dtype=torch.long),
                torch.as_tensor(response_attention_mask, dtype=torch.long),
                torch.as_tensor(persona_input_ids, dtype=torch.long),
                torch.as_tensor(persona_attention_mask, dtype=torch.long),
                dialog_img_feat,
                persona_img_feats,
                torch.as_tensor(dialog_img_mask, dtype=torch.long),
                torch.as_tensor(persona_img_mask, dtype=torch.long),
            ]
        else:
            context_text, response, dialog_fpath, cand_persona_sentences_list, cand_persona_images_list, label_idx, mode = self.examples[index]

            context_inputs = self.clip_processor(text=context_text)
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

            persona_inputs = self.clip_processor(text=cand_persona_sentences_list)
            persona_input_ids = persona_inputs['input_ids']
            persona_attention_mask = persona_inputs['attention_mask']

            for i in range(len(cand_persona_sentences_list)):
                if len(persona_input_ids[i]) > self.args.max_seq_length:
                    persona_input_ids[i] = persona_input_ids[i][len(persona_input_ids[i]) - self.args.max_seq_length:]
                    persona_attention_mask[i] = persona_attention_mask[i][len(persona_attention_mask[i]) - self.args.max_seq_length:]
                while len(persona_input_ids[i]) < self.args.max_seq_length:
                    persona_input_ids[i].append(self.clip_processor.tokenizer.pad_token_id)
                    persona_attention_mask[i].append(0)
            persona_input_ids = torch.as_tensor(persona_input_ids, dtype=torch.long)
            persona_attention_mask = torch.as_tensor(persona_attention_mask, dtype=torch.long)

            assert persona_input_ids.shape == torch.Size([self.args.max_num_candidate_authors, self.args.max_seq_length])

            if dialog_fpath == '':
                dialog_img_feat = torch.rand(3, self.args.img_size, self.args.img_size)
                dialog_img_mask = 0
            else:
                dialog_img = pil_loader(dialog_fpath)
                dialog_img_feat = self.clip_processor(images=dialog_img, return_tensors='pt')['pixel_values'].squeeze(0)
                dialog_img_mask = 1
            assert dialog_img_feat.shape == torch.Size([3, self.args.img_size, self.args.img_size])

            persona_img_feats = []
            persona_img_masks = []
            for i in range(len(cand_persona_images_list)):
                persona_imgs = [pil_loader(x) for x in cand_persona_images_list[i]]
                persona_img_feat = self.clip_processor(images=persona_imgs, return_tensors='pt')['pixel_values']
                if len(persona_imgs) < self.args.max_num_imgs:
                    empty_img_feat = torch.stack([torch.rand(3, self.args.img_size, self.args.img_size) for _ in range(self.args.max_num_imgs - len(persona_imgs))])
                    persona_img_feat = torch.cat([persona_img_feat, empty_img_feat], dim=0)
                persona_img_feats.append(persona_img_feat)
                persona_img_masks.append([1 for _ in range(len(persona_imgs))] + [0 for _ in range(self.args.max_num_imgs - len(persona_imgs))])
            persona_img_feats = torch.stack(persona_img_feats)
            assert persona_img_feats.shape == torch.Size([self.args.max_num_candidate_authors, self.args.max_num_imgs, 3, self.args.img_size, self.args.img_size])
            persona_img_masks = torch.as_tensor(persona_img_masks, dtype=torch.long)
            assert persona_img_masks.shape == torch.Size([self.args.max_num_candidate_authors, self.args.max_num_imgs])

            feature = [
                torch.as_tensor(context_input_ids, dtype=torch.long),
                torch.as_tensor(context_attention_mask, dtype=torch.long),
                torch.as_tensor(response_input_ids, dtype=torch.long),
                torch.as_tensor(response_attention_mask, dtype=torch.long),
                torch.as_tensor(persona_input_ids, dtype=torch.long),
                torch.as_tensor(persona_attention_mask, dtype=torch.long),
                dialog_img_feat,
                persona_img_feats,
                label_idx,
                torch.as_tensor(dialog_img_mask, dtype=torch.long),
                persona_img_masks,
            ]

        return feature

class MpchatClipSbertSiDataset(Dataset):
    def __init__(self,
                 args,
                 tokenizer,
                 clip_processor,
                 mode):
        super(MpchatClipSbertSiDataset, self).__init__()
        assert mode in ['train', 'val', 'test']

        self.args = args
        self.tokenizer = tokenizer
        self.clip_processor = clip_processor
        self.mode = mode
        self.examples = []

        with open(os.path.join(args.dialog_data_dir, 'mpchat_si.json'), 'r') as fp:
            data = json.load(fp)[f'{mode}']

        for dialog_idx, dialog in enumerate(data):
            main_author = dialog['main_author']
            context = ' '.join(dialog['messages'][:-1])
            response = dialog['messages'][-1]
            persona_sentences = ' '.join([f"{x['title']}" for x in dialog['si_main_author_candidate_personas']])
            persona_fpaths = [os.path.join(args.persona_image_data_dir, x['file_name']) for x in dialog['si_main_author_candidate_personas']]
            if dialog['has_image']:
                fname_context = dialog['file_name']
                dialog_fpath = os.path.join(args.dialog_image_data_dir, fname_context)
            else:
                dialog_fpath = ''

            if mode == 'train':
                self.examples.append((context, response, dialog_fpath, persona_sentences, persona_fpaths, mode))
            else:
                # add other authors persona candidates
                assert persona_sentences == ' '.join([x['title'] for x in dialog['si_candidate_authors_candidate_personas'][0]])
                cand_authors_cand_persona_sentences_list = []
                cand_authors_cand_persona_images_list = []
                for i in range(self.args.max_num_candidate_authors):
                    cand_authors_cand_persona_sentences_list.append(' '.join([x['title'] for x in dialog['si_candidate_authors_candidate_personas'][i]]))
                    cand_authors_cand_persona_images_list.append([os.path.join(args.persona_image_data_dir, x['file_name']) for x in dialog['si_candidate_authors_candidate_personas'][i]])
                self.examples.append((context, response, dialog_fpath, cand_authors_cand_persona_sentences_list, cand_authors_cand_persona_images_list, 0, mode))

        print(f'num. of {mode} dataset: {len(self.examples)}')

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        mode = self.examples[index][-1]

        if mode == 'train':
            context_text, response, dialog_fpath, persona_sentences, persona_fpaths, mode = self.examples[index]

            context_inputs = self.tokenizer(context_text,
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

            if dialog_fpath == '':
                dialog_img_feat = torch.rand(3, self.args.img_size, self.args.img_size)
                dialog_img_mask = 0
            else:
                dialog_img = pil_loader(dialog_fpath)
                dialog_img_feat = self.clip_processor(images=dialog_img, return_tensors='pt')['pixel_values'].squeeze(0)
                dialog_img_mask = 1
            assert dialog_img_feat.shape == torch.Size([3, self.args.img_size, self.args.img_size])

            persona_imgs = [pil_loader(x) for x in persona_fpaths]
            persona_img_feats = self.clip_processor(images=persona_imgs, return_tensors='pt')['pixel_values']
            persona_img_mask = [1 for _ in persona_fpaths]
            if len(persona_fpaths) < self.args.max_num_imgs:
                empty_img_feats = torch.stack([torch.rand(3, self.args.img_size, self.args.img_size) for _ in range(self.args.max_num_imgs - len(persona_fpaths))])
                persona_img_feats = torch.cat([persona_img_feats, empty_img_feats], dim=0)
                persona_img_mask += [0 for _ in range(self.args.max_num_imgs - len(persona_fpaths))]
            assert persona_img_feats.shape == torch.Size([self.args.max_num_imgs, 3, self.args.img_size, self.args.img_size])
            assert len(persona_img_mask) == self.args.max_num_imgs

            feature = [
                torch.as_tensor(context_input_ids, dtype=torch.long),
                torch.as_tensor(context_attention_mask, dtype=torch.long),
                torch.as_tensor(response_input_ids, dtype=torch.long),
                torch.as_tensor(response_attention_mask, dtype=torch.long),
                torch.as_tensor(persona_input_ids, dtype=torch.long),
                torch.as_tensor(persona_attention_mask, dtype=torch.long),
                dialog_img_feat,
                persona_img_feats,
                torch.as_tensor(dialog_img_mask, dtype=torch.long),
                torch.as_tensor(persona_img_mask, dtype=torch.long),
            ]
        else:
            context_text, response, dialog_fpath, cand_persona_sentences_list, cand_persona_images_list, label_idx, mode = self.examples[index]

            context_inputs = self.tokenizer(context_text,
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

            persona_inputs = self.tokenizer(cand_persona_sentences_list,
                                            truncation=True,
                                            padding='max_length',
                                            max_length=self.args.max_seq_length)
            persona_input_ids = torch.as_tensor(persona_inputs['input_ids'], dtype=torch.long)
            persona_attention_mask = torch.as_tensor(persona_inputs['attention_mask'], dtype=torch.long)

            assert persona_input_ids.shape == torch.Size([self.args.max_num_candidate_authors, self.args.max_seq_length])

            if dialog_fpath == '':
                dialog_img_feat = torch.rand(3, self.args.img_size, self.args.img_size)
                dialog_img_mask = 0
            else:
                dialog_img = pil_loader(dialog_fpath)
                dialog_img_feat = self.clip_processor(images=dialog_img, return_tensors='pt')['pixel_values'].squeeze(0)
                dialog_img_mask = 1
            assert dialog_img_feat.shape == torch.Size([3, self.args.img_size, self.args.img_size])

            persona_img_feats = []
            persona_img_masks = []
            for i in range(len(cand_persona_images_list)):
                persona_imgs = [pil_loader(x) for x in cand_persona_images_list[i]]
                persona_img_feat = self.clip_processor(images=persona_imgs, return_tensors='pt')['pixel_values']
                if len(persona_imgs) < self.args.max_num_imgs:
                    empty_img_feat = torch.stack([torch.rand(3, self.args.img_size, self.args.img_size) for _ in range(self.args.max_num_imgs - len(persona_imgs))])
                    persona_img_feat = torch.cat([persona_img_feat, empty_img_feat], dim=0)
                persona_img_feats.append(persona_img_feat)
                persona_img_masks.append([1 for _ in range(len(persona_imgs))] + [0 for _ in range(self.args.max_num_imgs - len(persona_imgs))])
            persona_img_feats = torch.stack(persona_img_feats)
            assert persona_img_feats.shape == torch.Size([self.args.max_num_candidate_authors, self.args.max_num_imgs, 3, self.args.img_size, self.args.img_size])
            persona_img_masks = torch.as_tensor(persona_img_masks, dtype=torch.long)
            assert persona_img_masks.shape == torch.Size([self.args.max_num_candidate_authors, self.args.max_num_imgs])

            feature = [
                torch.as_tensor(context_input_ids, dtype=torch.long),
                torch.as_tensor(context_attention_mask, dtype=torch.long),
                torch.as_tensor(response_input_ids, dtype=torch.long),
                torch.as_tensor(response_attention_mask, dtype=torch.long),
                torch.as_tensor(persona_input_ids, dtype=torch.long),
                torch.as_tensor(persona_attention_mask, dtype=torch.long),
                dialog_img_feat,
                persona_img_feats,
                label_idx,
                torch.as_tensor(dialog_img_mask, dtype=torch.long),
                persona_img_masks,
            ]

        return feature
