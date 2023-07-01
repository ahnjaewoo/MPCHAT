import torch

from transformers import (
    CLIPModel,
    AutoModel,
)

# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class BaseModel(torch.nn.Module):
    def __init__(self, args, clip_processor):
        super().__init__()
        self.args = args
        self.clip_processor = clip_processor

    def forward(self):
        pass

class ClipClipModel(BaseModel):
    def __init__(self, args, clip_processor):
        super().__init__(args, clip_processor)

        clip_models = [CLIPModel.from_pretrained(args.clip_model_name) for x in range(3)]
        self.context_image_encoder = clip_models[0].vision_model
        self.context_image_projection = clip_models[0].visual_projection
        self.context_text_encoder = clip_models[0].text_model
        self.context_text_projection = clip_models[0].text_projection
        self.response_encoder = clip_models[1].text_model
        self.response_projection = clip_models[1].text_projection
        self.persona_text_encoder = clip_models[2].text_model
        self.persona_text_projection = clip_models[2].text_projection
        self.persona_image_encoder = clip_models[1].vision_model
        self.persona_image_projection = clip_models[1].visual_projection

        self.logit_scale = clip_models[0].logit_scale

class ClipSbertModel(BaseModel):
    def __init__(self, args, clip_processor):
        super().__init__(args, clip_processor)

        sbert_models = [AutoModel.from_pretrained(args.sbert_model_name) for x in range(3)]
        self.context_text_encoder = sbert_models[0]
        self.persona_text_encoder = sbert_models[1]
        self.response_encoder = sbert_models[2]
        clip_models = [CLIPModel.from_pretrained(args.clip_model_name) for x in range(2)]
        self.context_image_encoder = clip_models[0].vision_model
        self.context_image_projection = torch.nn.Linear(self.context_image_encoder.config.hidden_size, self.context_text_encoder.config.hidden_size)
        self.persona_image_encoder = clip_models[1].vision_model
        self.persona_image_projection = torch.nn.Linear(self.context_image_encoder.config.hidden_size, self.context_text_encoder.config.hidden_size)
