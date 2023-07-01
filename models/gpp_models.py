import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from typing import Any, Optional, Tuple, Union

from .model import (
    ClipClipModel,
    ClipSbertModel,
    clip_loss,
    mean_pooling,
)

from transformers import (
    CLIPProcessor,
    AdamW,
    get_linear_schedule_with_warmup,
    WEIGHTS_NAME,
)

class ClipClipGpp(ClipClipModel):
    def forward(
        self,
        context_input_ids: Optional[torch.LongTensor] = None,
        context_attention_mask: Optional[torch.LongTensor] = None,
        response_input_ids: Optional[torch.LongTensor] = None,
        response_attention_mask: Optional[torch.LongTensor] = None,
        persona_input_ids: Optional[torch.LongTensor] = None,
        persona_attention_mask: Optional[torch.LongTensor] = None,
        final_persona_input_ids: Optional[torch.LongTensor] = None,
        final_persona_attention_mask: Optional[torch.LongTensor] = None,
        dialog_img_feat: Optional[torch.Tensor] = None,
        persona_img_feats: Optional[torch.Tensor] = None,
        final_persona_img_feats: Optional[torch.Tensor] = None,
        dialog_img_mask: Optional[torch.LongTensor] = None,
        persona_img_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        mode: str = None,
    ):
        if mode == 'train':
            context_output = self.context_text_encoder(
                input_ids=context_input_ids,
                attention_mask=context_attention_mask
            )[1]
            context_output = self.context_text_projection(context_output)
            context_output = F.normalize(context_output, p=2, dim=1)

            persona_output = self.persona_text_encoder(
                input_ids=persona_input_ids,
                attention_mask=persona_attention_mask
            )[1]
            persona_output = self.persona_text_projection(persona_output)
            persona_output = F.normalize(persona_output, p=2, dim=1)

            if self.args.use_response:
                response_output = self.response_encoder(
                    input_ids=response_input_ids,
                    attention_mask=response_attention_mask
                )[1]
                response_output = self.response_projection(response_output)
                response_output = F.normalize(response_output, p=2, dim=1)

            dialog_image_output = self.context_image_encoder(pixel_values=dialog_img_feat)[1]
            dialog_image_output = self.context_image_projection(dialog_image_output)
            dialog_image_output = F.normalize(dialog_image_output, p=2, dim=1)

            persona_image_output = self.persona_image_encoder(pixel_values=persona_img_feats.view(-1, 3, self.args.img_size, self.args.img_size))[1]
            persona_image_output = self.persona_image_projection(persona_image_output)
            persona_image_output = F.normalize(persona_image_output, p=2, dim=1)
            persona_image_output = persona_image_output.view(persona_img_feats.size(0), self.args.max_num_imgs, persona_image_output.size(-1))

            if self.args.sum_persona_images:
                if self.args.remove_empty_images:
                    persona_image_output = torch.sum(persona_img_mask.unsqueeze(-1).repeat(1,1,dialog_image_output.size(-1)) * persona_image_output, dim=1)
                    persona_image_output = persona_image_output / torch.sum(persona_img_mask, dim=1).unsqueeze(-1).repeat(1,dialog_image_output.size(-1))
                    multimodal_persona_output = (persona_output + persona_image_output) / 2
                    if self.args.use_response:
                        multimodal_context_output = context_output + response_output + multimodal_persona_output
                    else:
                        multimodal_context_output = context_output + multimodal_persona_output
                    multimodal_context_output += (dialog_img_mask.unsqueeze(-1).repeat(1,dialog_image_output.size(-1)) * dialog_image_output)
                    if self.args.use_response:
                        multimodal_context_output /= (dialog_img_mask + 3).unsqueeze(-1).repeat(1,dialog_image_output.size(-1))
                    else:
                        multimodal_context_output /= (dialog_img_mask + 2).unsqueeze(-1).repeat(1, dialog_image_output.size(-1))
                else:
                    persona_image_output = torch.mean(persona_image_output, dim=1)
                    multimodal_persona_output = (persona_output + persona_image_output) / 2
                    if self.args.use_response:
                        multimodal_context_output = (context_output + response_output + dialog_image_output + multimodal_persona_output) / 4
                    else:
                        multimodal_context_output = (context_output + dialog_image_output + multimodal_persona_output) / 3
            else:
                raise NotImplementedError

            final_persona_output = self.persona_text_encoder(
                input_ids=final_persona_input_ids,
                attention_mask=final_persona_attention_mask
            )[1]
            final_persona_output = self.persona_text_projection(final_persona_output)
            final_persona_output = F.normalize(final_persona_output, p=2, dim=1)

            final_persona_image_output = self.persona_image_encoder(pixel_values=final_persona_img_feats)[1]
            final_persona_image_output = self.persona_image_projection(final_persona_image_output)
            final_persona_image_output = F.normalize(final_persona_image_output, p=2, dim=1)

            final_multimodal_persona_output = (final_persona_output + final_persona_image_output) / 2

            logit_scale = self.logit_scale.exp()
            dot_products = multimodal_context_output.mm(final_multimodal_persona_output.t()) * logit_scale
            loss = clip_loss(dot_products)

            outputs = (loss,)
        else:
            context_output = self.context_text_encoder(
                input_ids=context_input_ids,
                attention_mask=context_attention_mask
            )[1]
            context_output = self.context_text_projection(context_output)
            context_output = F.normalize(context_output, p=2, dim=1)

            persona_output = self.persona_text_encoder(
                input_ids=persona_input_ids,
                attention_mask=persona_attention_mask
            )[1]
            persona_output = self.persona_text_projection(persona_output)
            persona_output = F.normalize(persona_output, p=2, dim=1)

            response_output = self.response_encoder(
                input_ids=response_input_ids,
                attention_mask=response_attention_mask
            )[1]
            response_output = self.response_projection(response_output)
            response_output = F.normalize(response_output, p=2, dim=1)

            dialog_image_output = self.context_image_encoder(pixel_values=dialog_img_feat)[1]
            dialog_image_output = self.context_image_projection(dialog_image_output)
            dialog_image_output = F.normalize(dialog_image_output, p=2, dim=1)

            persona_image_output = self.persona_image_encoder(pixel_values=persona_img_feats.view(-1, 3, self.args.img_size, self.args.img_size))[1]
            persona_image_output = self.persona_image_projection(persona_image_output)
            persona_image_output = F.normalize(persona_image_output, p=2, dim=1)
            persona_image_output = persona_image_output.view(persona_img_feats.size(0), self.args.max_num_imgs, persona_image_output.size(-1))

            if self.args.sum_persona_images:
                if self.args.remove_empty_images:
                    persona_image_output = torch.sum(persona_img_mask.unsqueeze(-1).repeat(1,1,dialog_image_output.size(-1)) * persona_image_output, dim=1)
                    persona_image_output = persona_image_output / torch.sum(persona_img_mask, dim=1).unsqueeze(-1).repeat(1,dialog_image_output.size(-1))
                    multimodal_persona_output = (persona_output + persona_image_output) / 2
                    if self.args.use_response:
                        multimodal_context_output = context_output + response_output + multimodal_persona_output
                    else:
                        multimodal_context_output = context_output + multimodal_persona_output
                    multimodal_context_output += (dialog_img_mask.unsqueeze(-1).repeat(1,dialog_image_output.size(-1)) * dialog_image_output)
                    if self.args.use_response:
                        multimodal_context_output /= (dialog_img_mask + 3).unsqueeze(-1).repeat(1,dialog_image_output.size(-1))
                    else:
                        multimodal_context_output /= (dialog_img_mask + 2).unsqueeze(-1).repeat(1, dialog_image_output.size(-1))
                else:
                    persona_image_output = torch.mean(persona_image_output, dim=1)
                    multimodal_persona_output = (persona_output + persona_image_output) / 2
                    if self.args.use_response:
                        multimodal_context_output = (context_output + response_output + dialog_image_output + multimodal_persona_output) / 4
                    else:
                        multimodal_context_output = (context_output + dialog_image_output + multimodal_persona_output) / 3
            else:
                raise NotImplementedError

            cand_final_persona_input_ids = final_persona_input_ids.view(-1, final_persona_input_ids.size(-1))
            cand_final_persona_attention_mask = final_persona_attention_mask.view(-1, final_persona_attention_mask.size(-1))
            cand_final_persona_output = self.persona_text_encoder(
                input_ids=cand_final_persona_input_ids,
                attention_mask=cand_final_persona_attention_mask
            )[1]
            cand_final_persona_output = self.persona_text_projection(cand_final_persona_output)
            cand_final_persona_output = F.normalize(cand_final_persona_output, p=2, dim=1)

            cand_final_persona_img_feats = final_persona_img_feats.view(-1, 3, self.args.img_size, self.args.img_size)

            cand_final_persona_image_output = self.persona_image_encoder(pixel_values=cand_final_persona_img_feats)[1]
            cand_final_persona_image_output = self.persona_image_projection(cand_final_persona_image_output)
            cand_final_persona_image_output = F.normalize(cand_final_persona_image_output, p=2, dim=1)

            cand_final_multimodal_persona_output = (cand_final_persona_output + cand_final_persona_image_output) / 2
            logits = torch.bmm(multimodal_context_output.unsqueeze(1),
                               cand_final_multimodal_persona_output.view(
                                   context_input_ids.size(0),
                                   self.args.max_num_candidate_persona_elements,
                                   -1).transpose(1,2)).squeeze(1)

            loss = CrossEntropyLoss(reduction='none')(logits, labels)
            outputs = (loss, logits,)
        return outputs

class ClipSbertGpp(ClipSbertModel):
    def forward(
        self,
        context_input_ids: Optional[torch.LongTensor] = None,
        context_attention_mask: Optional[torch.LongTensor] = None,
        response_input_ids: Optional[torch.LongTensor] = None,
        response_attention_mask: Optional[torch.LongTensor] = None,
        persona_input_ids: Optional[torch.LongTensor] = None,
        persona_attention_mask: Optional[torch.LongTensor] = None,
        final_persona_input_ids: Optional[torch.LongTensor] = None,
        final_persona_attention_mask: Optional[torch.LongTensor] = None,
        dialog_img_feat: Optional[torch.Tensor] = None,
        persona_img_feats: Optional[torch.Tensor] = None,
        final_persona_img_feats: Optional[torch.Tensor] = None,
        dialog_img_mask: Optional[torch.LongTensor] = None,
        persona_img_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        mode: str = None,
    ):
        if mode == 'train':
            context_output = self.context_text_encoder(
                input_ids=context_input_ids,
                attention_mask=context_attention_mask
            )
            context_output = mean_pooling(context_output, context_attention_mask)

            persona_output = self.persona_text_encoder(
                input_ids=persona_input_ids,
                attention_mask=persona_attention_mask
            )
            persona_output = mean_pooling(persona_output, persona_attention_mask)

            response_output = self.response_encoder(
                input_ids=response_input_ids,
                attention_mask=response_attention_mask
            )
            response_output = mean_pooling(response_output, response_attention_mask)

            dialog_image_output = self.context_image_encoder(pixel_values=dialog_img_feat)[1]
            dialog_image_output = self.context_image_projection(dialog_image_output)
            dialog_image_output = F.normalize(dialog_image_output, p=2, dim=1)

            persona_image_output = self.persona_image_encoder(pixel_values=persona_img_feats.view(-1, 3, self.args.img_size, self.args.img_size))[1]
            persona_image_output = self.persona_image_projection(persona_image_output)
            persona_image_output = F.normalize(persona_image_output, p=2, dim=1)
            persona_image_output = persona_image_output.view(persona_img_feats.size(0), self.args.max_num_imgs, persona_image_output.size(-1))

            if self.args.sum_persona_images:
                if self.args.remove_empty_images:
                    persona_image_output = torch.sum(persona_img_mask.unsqueeze(-1).repeat(1,1,dialog_image_output.size(-1)) * persona_image_output, dim=1)
                    persona_image_output = persona_image_output / torch.sum(persona_img_mask, dim=1).unsqueeze(-1).repeat(1,dialog_image_output.size(-1))
                    multimodal_persona_output = (persona_output + persona_image_output) / 2
                    if self.args.use_response:
                        multimodal_context_output = context_output + response_output + multimodal_persona_output
                    else:
                        multimodal_context_output = context_output + multimodal_persona_output
                    multimodal_context_output += (dialog_img_mask.unsqueeze(-1).repeat(1,dialog_image_output.size(-1)) * dialog_image_output)
                    if self.args.use_response:
                        multimodal_context_output /= (dialog_img_mask + 3).unsqueeze(-1).repeat(1,dialog_image_output.size(-1))
                    else:
                        multimodal_context_output /= (dialog_img_mask + 2).unsqueeze(-1).repeat(1, dialog_image_output.size(-1))
                else:
                    persona_image_output = torch.mean(persona_image_output, dim=1)
                    multimodal_persona_output = (persona_output + persona_image_output) / 2
                    if self.args.use_response:
                        multimodal_context_output = (context_output + response_output + dialog_image_output + multimodal_persona_output) / 4
                    else:
                        multimodal_context_output = (context_output + dialog_image_output + multimodal_persona_output) / 3
            else:
                raise NotImplementedError

            final_persona_output = self.persona_text_encoder(
                input_ids=final_persona_input_ids,
                attention_mask=final_persona_attention_mask
            )
            final_persona_output = mean_pooling(final_persona_output, final_persona_attention_mask)

            final_persona_image_output = self.persona_image_encoder(pixel_values=final_persona_img_feats)[1]
            final_persona_image_output = self.persona_image_projection(final_persona_image_output)
            final_persona_image_output = F.normalize(final_persona_image_output, p=2, dim=1)

            final_multimodal_persona_output = (final_persona_output + final_persona_image_output) / 2

            targets = torch.arange(context_output.shape[0], device=context_output.device)
            # dot_products: [batch, batch]
            dot_products = multimodal_context_output.mm(final_multimodal_persona_output.t())
            log_prob = F.log_softmax(dot_products, dim=1)
            loss = F.nll_loss(log_prob, targets)

            outputs = (loss,)
        else:
            context_output = self.context_text_encoder(
                input_ids=context_input_ids,
                attention_mask=context_attention_mask
            )
            context_output = mean_pooling(context_output, context_attention_mask)

            persona_output = self.persona_text_encoder(
                input_ids=persona_input_ids,
                attention_mask=persona_attention_mask
            )
            persona_output = mean_pooling(persona_output, persona_attention_mask)

            response_output = self.response_encoder(
                input_ids=response_input_ids,
                attention_mask=response_attention_mask
            )
            response_output = mean_pooling(response_output, response_attention_mask)

            dialog_image_output = self.context_image_encoder(pixel_values=dialog_img_feat)[1]
            dialog_image_output = self.context_image_projection(dialog_image_output)
            dialog_image_output = F.normalize(dialog_image_output, p=2, dim=1)

            persona_image_output = self.persona_image_encoder(pixel_values=persona_img_feats.view(-1, 3, self.args.img_size, self.args.img_size))[1]
            persona_image_output = self.persona_image_projection(persona_image_output)
            persona_image_output = F.normalize(persona_image_output, p=2, dim=1)
            persona_image_output = persona_image_output.view(persona_img_feats.size(0), self.args.max_num_imgs, persona_image_output.size(-1))

            if self.args.sum_persona_images:
                if self.args.remove_empty_images:
                    persona_image_output = torch.sum(persona_img_mask.unsqueeze(-1).repeat(1,1,dialog_image_output.size(-1)) * persona_image_output, dim=1)
                    persona_image_output = persona_image_output / torch.sum(persona_img_mask, dim=1).unsqueeze(-1).repeat(1,dialog_image_output.size(-1))
                    multimodal_persona_output = (persona_output + persona_image_output) / 2
                    if self.args.use_response:
                        multimodal_context_output = context_output + response_output + multimodal_persona_output
                    else:
                        multimodal_context_output = context_output + multimodal_persona_output
                    multimodal_context_output += (dialog_img_mask.unsqueeze(-1).repeat(1,dialog_image_output.size(-1)) * dialog_image_output)
                    if self.args.use_response:
                        multimodal_context_output /= (dialog_img_mask + 3).unsqueeze(-1).repeat(1,dialog_image_output.size(-1))
                    else:
                        multimodal_context_output /= (dialog_img_mask + 2).unsqueeze(-1).repeat(1, dialog_image_output.size(-1))
                else:
                    persona_image_output = torch.mean(persona_image_output, dim=1)
                    multimodal_persona_output = (persona_output + persona_image_output) / 2
                    if self.args.use_response:
                        multimodal_context_output = (context_output + response_output + dialog_image_output + multimodal_persona_output) / 4
                    else:
                        multimodal_context_output = (context_output + dialog_image_output + multimodal_persona_output) / 3
            else:
                raise NotImplementedError

            cand_final_persona_input_ids = final_persona_input_ids.view(-1, final_persona_input_ids.size(-1))
            cand_final_persona_attention_mask = final_persona_attention_mask.view(-1, final_persona_attention_mask.size(-1))
            cand_final_persona_output = self.persona_text_encoder(
                input_ids=cand_final_persona_input_ids,
                attention_mask=cand_final_persona_attention_mask
            )
            cand_final_persona_output = mean_pooling(cand_final_persona_output, cand_final_persona_attention_mask)

            cand_final_persona_img_feats = final_persona_img_feats.view(-1, 3, self.args.img_size, self.args.img_size)

            cand_final_persona_image_output = self.persona_image_encoder(pixel_values=cand_final_persona_img_feats)[1]
            cand_final_persona_image_output = self.persona_image_projection(cand_final_persona_image_output)
            cand_final_persona_image_output = F.normalize(cand_final_persona_image_output, p=2, dim=1)

            cand_final_multimodal_persona_output = (cand_final_persona_output + cand_final_persona_image_output) / 2
            logits = torch.bmm(multimodal_context_output.unsqueeze(1),
                               cand_final_multimodal_persona_output.view(
                                   context_input_ids.size(0),
                                   self.args.max_num_candidate_persona_elements,
                                   -1).transpose(1,2)).squeeze(1)

            loss = CrossEntropyLoss(reduction='none')(logits, labels)
            outputs = (loss, logits,)
        return outputs
