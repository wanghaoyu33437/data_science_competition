import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

from category_id_map import CATEGORY_ID_LIST


class MultiModal(nn.Module):
    def __init__(self, args):
        super().__init__()
        # self.bert = BertModel.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)
        self.bert = BertModel.from_pretrained(args.bert_dir, cache_dir=args.bert_cache, add_pooling_layer=False)
        # self.pooler = MeanPooling()
        self.nextvlad = NeXtVLAD(args.frame_embedding_size, args.vlad_cluster_size,
                                 output_size=args.vlad_hidden_size, dropout=args.dropout)
        bert_output_size = args.bert_output_size
        self.enhance = SENet(channels=args.vlad_hidden_size, ratio=args.se_ratio)
        self.fusion = ConcatDenseSE(args.vlad_hidden_size + bert_output_size, args.fc_size, args.se_ratio, args.dropout)
        # self.classifier =nn.Linear(args.fc_size,len(CATEGORY_ID_LIST))
        self.classifier = nn.Sequential(
            nn.Linear(args.fc_size,1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512,len(CATEGORY_ID_LIST))
        )


    def forward(self, inputs, inference=False):
        bert_embedding = self.bert(inputs['title_input'], inputs['title_mask'])['last_hidden_state']
        bert_embedding = torch.einsum("bsh,bs,b->bh", bert_embedding, inputs['title_mask'].float(), 1 / inputs['title_mask'].float().sum(dim=1) + 1e-9)
#         bert_embedding = self.pooler(bert_embedding,inputs['title_mask'])
        vision_embedding = self.nextvlad(inputs['frame_input'], inputs['frame_mask'])
#         vision_embedding = self.bert(inputs_embeds=inputs['frame_input'], attention_mask=inputs['frame_mask'])['last_hidden_state']
#         vision_embedding = torch.einsum("bsh,bs,b->bh", vision_embedding, inputs['frame_mask'].float(), 1 / inputs['frame_mask'].float().sum(dim=1) + 1e-9)
        vision_embedding = self.enhance(vision_embedding)
        # print(bert_embedding.shape)
        # print(vision_embedding.shape)
        final_embedding = self.fusion([vision_embedding, bert_embedding])
        prediction = self.classifier(final_embedding)

        if inference:
            return prediction
        else:
            return self.cal_loss(prediction, inputs['label'])

    @staticmethod
    def cal_loss(prediction, label):
        label = label.squeeze(dim=1)
        loss = F.cross_entropy(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label


class NeXtVLAD(nn.Module):
    def __init__(self, feature_size, cluster_size, output_size=1024, expansion=2, groups=8, dropout=0.2,max_frames=32):
        super().__init__()
        self.feature_size = feature_size
        self.output_size = output_size
        self.expansion_size = expansion
        self.cluster_size = cluster_size
        self.groups = groups
        self.drop_rate = dropout

        self.new_feature_size = self.expansion_size * self.feature_size // self.groups

        self.dropout = torch.nn.Dropout(self.drop_rate)
        self.expansion_linear = torch.nn.Linear(self.feature_size, self.expansion_size * self.feature_size)
        self.group_attention = torch.nn.Linear(self.expansion_size * self.feature_size, self.groups)
        self.cluster_linear = torch.nn.Linear(self.expansion_size * self.feature_size, self.groups * self.cluster_size,
                                              bias=False)
        self.cluster_weight = torch.nn.Parameter(
            torch.nn.init.normal_(torch.rand(1, self.new_feature_size, self.cluster_size), std=0.01))
        self.fc = torch.nn.Linear(self.new_feature_size * self.cluster_size, self.output_size)

        self.bn0 = nn.BatchNorm1d(max_frames)
        self.bn1 = nn.BatchNorm1d(1)
    def forward(self, inputs, mask=None):
        # todo mask
        _,M,N = inputs.shape
        inputs = self.expansion_linear(inputs)

        attention = self.group_attention(inputs)
        attention = torch.sigmoid(attention)
        if mask is not None:
            attention = torch.mul(attention, mask.unsqueeze(2))

        attention = attention.reshape([-1, M * self.groups, 1])
        # print(inputs.shape)
        # reshaped_input = inputs.reshape([-1, self.expansion_size * self.feature_size])
        # reshaped_input = inputs.reshape([-1,M,self.groups,])
        activation = self.cluster_linear(inputs)
        activation = self.bn0(activation)

        activation = activation.reshape([-1, M * self.groups, self.cluster_size])
        activation = torch.softmax(activation, dim=-1)

        activation = activation * attention

        a_sum = activation.sum(-2, keepdim=True)
        a = a_sum * self.cluster_weight
        activation = activation.permute(0, 2, 1).contiguous()

        reshaped_input = inputs.reshape([-1, inputs.shape[1] * self.groups, self.new_feature_size])

        vlad = torch.matmul(activation, reshaped_input)
        vlad = vlad.permute(0, 2, 1).contiguous()
        vlad = F.normalize(vlad - a, p=2, dim=1)
        vlad = vlad.reshape(-1, 1, self.cluster_size * self.new_feature_size)
        vlad = self.bn1(vlad)
        vlad = vlad.reshape([-1, self.cluster_size * self.new_feature_size])
        vlad = self.dropout(vlad)
        vlad = self.fc(vlad)
        return vlad

class SENet(nn.Module):
    def __init__(self, channels, ratio=8):
        super().__init__()
        self.sequeeze = nn.Linear(in_features=channels, out_features=channels // ratio, bias=False)
        self.relu = nn.ReLU()
        self.excitation = nn.Linear(in_features=channels // ratio, out_features=channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gates = self.sequeeze(x)
        gates = self.relu(gates)
        gates = self.excitation(gates)
        gates = self.sigmoid(gates)
        x = torch.mul(x, gates)
        return x


class ConcatDenseSE(nn.Module):
    def __init__(self, multimodal_hidden_size, hidden_size, se_ratio, dropout):
        super().__init__()
        self.fusion = nn.Linear(multimodal_hidden_size, hidden_size)
        self.fusion_dropout = nn.Dropout(dropout)
        self.enhance = SENet(channels=hidden_size, ratio=se_ratio)

    def forward(self, inputs):
        embeddings = torch.cat(inputs, dim=1)
        embeddings = self.fusion_dropout(embeddings)
        embedding = self.fusion(embeddings)
        embedding = self.enhance(embedding)
        return embedding


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

from transformers.models.bert.modeling_bert import *

class MyBert(BertPreTrainedModel):
    def __init__(self,config,args):
        super().__init__(config)
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.video_embeddings = BertEmbeddings(config)
        # self.nextvlad = NeXtVLAD(args.frame_embedding_size, args.vlad_cluster_size,
        #                          output_size=args.vlad_hidden_size, dropout=args.dropout)
        self.video_fc = nn.Sequential(
                nn.Linear(768, config.hidden_size),
				SENet(channels=config.hidden_size, ratio=args.se_ratio)
                # nn.ReLU()
				)
        self.encoder = BertEncoder(config)
        self.drop = nn.Dropout(p = args.dropout)
        # Initialize weights and apply final processing
        # self.enhance =
        self.cls = nn.Sequential(
            nn.Linear(args.vlad_hidden_size,1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512, len(CATEGORY_ID_LIST))
        )
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)
    def _forward_feature(self,text_input,text_mask,frame_input,fram_mask):
        text_embedding= self.embeddings(text_input)
        frame_embedding = self.video_fc(frame_input)
        frame_embedding=self.video_embeddings(inputs_embeds=frame_embedding)
        # frame_embedding = self.drop(frame_embedding)
        embedding_output = torch.cat([frame_embedding, text_embedding], 1)
        mask = torch.cat([fram_mask, text_mask], 1)
        attention_mask = mask[:, None, None, :]
        attention_mask = (1.0 - attention_mask) * -10000.0
        encoder_outputs = self.encoder(embedding_output, attention_mask=attention_mask)['last_hidden_state']
        return encoder_outputs,mask
    def forward(self,inputs,inference=False):
        encoder_outputs,mask = self._forward_feature( inputs['title_input'],inputs['title_mask'],inputs['frame_input'],inputs['frame_mask'])
        emb = torch.einsum("bsh,bs,b->bh", encoder_outputs, mask.float(), 1 / mask.float().sum(dim=1) + 1e-9)
        # text_emb = self.enhance(text_emb)
        # emb = self.drop(emb)
        prediction = self.cls(emb)
        if inference:
            return prediction
        else:
            return self.cal_loss(prediction, inputs['label'])
    @staticmethod
    def cal_loss(prediction, label):
        label = label.squeeze(dim=1)
        loss = F.cross_entropy(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label


