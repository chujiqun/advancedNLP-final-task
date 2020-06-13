from typing import Dict, Optional, List, Any

import numpy
from overrides import overrides
import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, TextFieldEmbedder, Seq2SeqEncoder, InputVariationalDropout
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy,Auc

from library.modules.neural_tensor_network import NeuralTensorNetwork
from library.modules.k_max_pool import kMaxPool

def cross_entropy(out, label):
    x1=out.squeeze(1)[out.squeeze(1)>1e-5]
    y1=label.squeeze(1)[out.squeeze(1)>1e-5]
    x2=1-out.squeeze(1)[out.squeeze(1)<1-1e-5]
    y2=1-label.squeeze(1)[out.squeeze(1)<1-1e-5]
    tmp=torch.sum(torch.mul(torch.log(x1),y1))+torch.sum(torch.mul(torch.log(x2),y2))
    return -1*tmp

@Model.register( "mv_lstm" )
class MV_LSTM( Model ):

    def __init__( self, vocab: Vocabulary,
                  text_field_embedder: TextFieldEmbedder,
                  dropout: float,
                  encoder: Seq2SeqEncoder,
                  matching_layer: NeuralTensorNetwork,
                  pool_layer: kMaxPool,
                  output_feedforward: FeedForward,
                  initializer: InitializerApplicator = InitializerApplicator(),
                  regularizer: Optional[RegularizerApplicator] = None ) -> None:
        super( MV_LSTM, self ).__init__( vocab, regularizer )

        self._text_field_embedder = text_field_embedder
        self._seq_dropout = InputVariationalDropout( dropout )
        self._matching_layer = matching_layer
        self._encoder = encoder
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
        premise_mask = util.get_text_field_mask( Orgquestion )
        encoded_premise = self._encoder( embedded_premise, premise_mask )

        embedded_hypo = self._seq_dropout( self._text_field_embedder( Relquestion ) )
        hypo_mask = util.get_text_field_mask( Relquestion )
        encoded_hypo = self._encoder( embedded_hypo, hypo_mask )

        # 使用Neural Tensor Network提取interaction
        similarity_matrix = self._matching_layer( encoded_premise, encoded_hypo )
        pool_out = self._pool_layer( similarity_matrix ) # k-max pooling到固定维度
        label_out = self._output_feedforward( pool_out )
        label_logits=torch.sigmoid(label_out)
        output_dict = {"label_logits": label_logits}

        if label is not None:
            loss = self._loss(label_logits, label)
            for metric in self.metrics.values():
                metric( label_logits.squeeze(1), label.squeeze(1) )
            output_dict["loss"] = loss

        return output_dict

    @overrides
    def get_metrics( self, reset: bool = False ) -> Dict[str, float]:
        return {metric_name: metric.get_metric( reset )
                    for metric_name, metric in self.metrics.items() }