# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/20 - clip.ipynb (unless otherwise specified).

__all__ = ['ClipTokenizer', 'vitb32_config', 'vitl14_config', 'Bottleneck', 'AttentionPool2d', 'ModifiedResNet',
           'LayerNorm', 'QuickGELU', 'ResidualAttentionBlock', 'Transformer', 'VisualTransformer', 'CLIP',
           'RetrievalAtK', 'CLIPTrainer', 'DistributedCLIPTrainer']

# Cell
from fastai.vision.all import *
from ..augmentations import *
from ..layers import *

# Cell
try:
    from clip.simple_tokenizer import SimpleTokenizer
except:
    raise ImportError("""
CLIP package is not installed/importable, please visit https://github.com/openai/CLIP or install following:
$ pip install ftfy regex tqdm
$ pip install git+https://github.com/openai/CLIP.git
    """)

# Cell
class ClipTokenizer(DisplayedTransform):
    "Tokenizer from https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py"
    def __init__(self, context_length=77):
        self._tokenizer = SimpleTokenizer()
        self.context_length = context_length
        self.vocab_size = len(self._tokenizer.encoder)

    def encodes(self:str, text):
        sot_token = self._tokenizer.encoder["<|startoftext|>"]
        eot_token = self._tokenizer.encoder["<|endoftext|>"]
        tokens = [sot_token] + self._tokenizer.encode(text) + [eot_token]
        result = torch.zeros(self.context_length, dtype=torch.long)
        if len(tokens) > self.context_length: raise Exception(f"Token length exceeds {self.context_length} for {text}")
        result[:len(tokens)] = torch.tensor(tokens)
        return TensorBase(result)

# Cell
def vitb32_config(input_res, context_length, vocab_size):
    "ViT-B/32 configuration, uses 32x32 patches"
    return dict(embed_dim=512,
                image_resolution=input_res,
                vision_layers=12,
                vision_width=768,
                vision_patch_size=32,
                context_length=context_length,
                vocab_size=vocab_size,
                transformer_width=512,
                transformer_heads=8,
                transformer_layers=12)

# Cell
def vitl14_config(input_res, context_length, vocab_size):
    "ViT-L/14 configuration, uses 14x14 patches"
    return dict(embed_dim=512,
                image_resolution=input_res,
                vision_layers=24,
                vision_width=1024,
                vision_patch_size=14,
                context_length=context_length,
                vocab_size=vocab_size,
                transformer_width=512,
                transformer_heads=8,
                transformer_layers=12)

# Cell
from collections import OrderedDict
from typing import Tuple, Union


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, checkpoint=False, checkpoint_nchunks=2):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
        self.checkpoint = checkpoint
        self.checkpoint_nchunks = checkpoint_nchunks

    def forward(self, x: torch.Tensor):
        if self.checkpoint: return torch.utils.checkpoint.checkpoint_sequential(self.resblocks, self.checkpoint_nchunks, x)
        else:               return self.resblocks(x)


class VisualTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, **kwargs):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, **kwargs)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 **kwargs
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisualTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                **kwargs
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            **kwargs
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/0.07))) # Same initialization as paper

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        # visual model
        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        # text model
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return image_features, text_features

# Cell
class RetrievalAtK(AccumMetric):

    def __init__(self, k=20, **kwargs):
        super().__init__(func=None, flatten=False, **kwargs)
        self.k = k

    @property
    def value(self):
        "For monitoring retrieval at k during training for sanity checking, should be used on < ~10000 samples"
        if len(self.preds) == 0: return

        image_features = torch.cat(list(L(self.preds).itemgot(0)))
        text_features = torch.cat(list(L(self.preds).itemgot(1)))
        ranking = torch.argsort(to_detach(image_features.to(default_device()) @ text_features.T.to(default_device()), gather=False),
                                descending=True)
        preds = array(torch.where(ranking == torch.arange(len(image_features)).view(-1,1))[1])

        if self.k == "mean":     return preds.mean() + 1
        elif self.k == "median": return np.floor(np.median(preds)) + 1
        else:                    return np.mean(preds < self.k)


    @property
    def name(self):
        if self.k == "mean":     return "mean_retrieval_ranking"
        elif self.k == "median": return "median_retrieval_ranking"
        else:                    return f"retrieval_at_{self.k}"

# Cell
class CLIPTrainer(Callback):
    "Can be used with or without DistributedDataParallel"
    order,run_valid = 9,True

    def before_fit(self):
        self.learn.loss_func = self.lf

    def lf(self, pred, *yb):
        image_features, text_features = pred
        if num_distrib()==0: logit_scale = self.model.logit_scale.exp()
        else:                logit_scale = self.model.module.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        labels = torch.arange(len(logits_per_image)).to(logits_per_image.device)
        image_loss = F.cross_entropy(logits_per_image, labels)
        text_loss  = F.cross_entropy(logits_per_text, labels)
        return (image_loss+text_loss)/2

    def after_step(self):
        # logit scaling set as max 100
        if num_distrib()==0: self.model.logit_scale.data = torch.clamp(self.model.logit_scale.data, 0, 4.6052)
        else:                self.model.module.logit_scale.data = torch.clamp(self.model.module.logit_scale.data, 0, 4.6052)

# Cell
class DistributedCLIPTrainer(Callback):
    "Distributed implementation of InfoNCE loss, should be used with DistributedDataParallel"
    order,run_valid = 9,True

    def before_fit(self):
        self.learn.loss_func = self.lf

    def lf(self, pred, *yb):

        image_features, text_features = pred
        logit_scale = self.model.module.logit_scale.exp()

        bs, rank = image_features.size(0), rank_distrib()

        # image loss
        all_text_features = [torch.zeros_like(text_features) for _ in range(dist.get_world_size())]
        dist.all_gather(all_text_features, text_features)
        all_text_features = torch.cat(all_text_features)
        logits_per_image = logit_scale * image_features @ all_text_features.t()
        labels = torch.arange(rank*bs, (rank+1)*bs).to(logits_per_image.device)
        image_loss = F.cross_entropy(logits_per_image, labels)

        # text loss
        all_image_features = [torch.zeros_like(image_features) for _ in range(dist.get_world_size())]
        dist.all_gather(all_image_features, image_features)
        all_image_features = torch.cat(all_image_features)
        logits_per_text = logit_scale * text_features @ all_image_features.t()
        labels = torch.arange(rank*bs, (rank+1)*bs).to(logits_per_text.device)
        text_loss  = F.cross_entropy(logits_per_text, labels)

        return (image_loss+text_loss)/2

    def after_step(self):
        # logit scaling set as max 100
        self.model.module.logit_scale.data = torch.clamp(self.model.module.logit_scale.data, 0, 4.6052)