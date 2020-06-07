from typing import Dict, Optional, List, Any

import numpy
from overrides import overrides
import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, TextFieldEmbedder,InputVariationalDropout
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy,Auc

from library.modules.conv_layers import ConvLayers
from library.modules.dynamic_pool import DynamicPool
from library.modules.matching_layer import MatchingLayer


def cross_entropy(out, label):
    tmp=torch.mul(torch.log(out.squeeze(1)),label.squeeze(1))+torch.mul(torch.log(1-out.squeeze(1)),1-label.squeeze(1))
    return -1*torch.sum(tmp)

@Model.register( "match_pyramid" )
class MatchPyramid( Model ):

    def __init__( self, vocab: Vocabulary,
                  text_field_embedder: TextFieldEmbedder,
                  dropout: float,
                  matching_layer: MatchingLayer,
                  inference_encoder: ConvLayers,
                  pool_layer: DynamicPool,
                  output_feedforward: FeedForward,
                  initializer: InitializerApplicator = InitializerApplicator(),
                  regularizer: Optional[RegularizerApplicator] = None ) -> None:
        super( MatchPyramid, self ).__init__( vocab, regularizer )
        self._text_field_embedder = text_field_embedder
        self._seq_dropout = InputVariationalDropout( dropout )
        self._matching_layer = matching_layer
        self._inference_encoder = inference_encoder
        self._pool_layer = pool_layer
        self._output_feedforward = output_feedforward
        self.metrics = {
               "Auc":Auc()
        }
        self._loss = cross_entropy

        initializer( self )

    @overrides
    def forward( self,
                 Orgquestion: Dict[str, torch.LongTensor],
                 Relquestion: Dict[str, torch.LongTensor],
                 label: torch.IntTensor = None,
                 metadata: List[Dict[str, Any]] = None ) -> Dict[str, torch.Tensor]:

        embedded_premise = self._seq_dropout( self._text_field_embedder( Orgquestion ) )
        embedded_hypo = self._seq_dropout( self._text_field_embedder(Relquestion ) )
        
        # 计算词向量的dot-product相似度
        similarity_matrix = torch.unsqueeze( 
            self._matching_layer( embedded_premise, embedded_hypo ), dim = 1 
        )
        conv_out = self._inference_encoder( similarity_matrix ) # 使用堆叠卷积层处理interaction
        pool_out = self._pool_layer( conv_out ) # dynamic pooling到固定维度

        label_logits = self._output_feedforward( pool_out )
        label_logits=F.sigmoid(label_logits)
        output_dict = {"label_logits": label_logits}

        if label is not None:
            loss = self._loss( label_logits, label)
            for metric in self.metrics.values():
                metric( label_logits.squeeze(1), label.squeeze(1) )
            output_dict["loss"] = loss

        return output_dict

    @overrides
    def get_metrics( self, reset: bool = False ) -> Dict[str, float]:
        return {metric_name: metric.get_metric( reset )
                    for metric_name, metric in self.metrics.items() }
    
