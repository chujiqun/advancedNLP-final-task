"""
Global attention of my own version for the one in AllenNLP is hard to use.
"""

import typing

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError

class GlobalAttention( nn.Module ):
    """
    Compute the attention weight.

    Parameters:
    -----------
    """
    def __init__( self,
                  input_dim: int,
                  weight_dim: int,
                  need_mlp: bool = None,
                  normalize: bool = True ):
        super( GlobalAttention, self ).__init__()

        self._input_dim = input_dim
        self._weight_dim = weight_dim
        self._need_mlp = need_mlp or ( input_dim != weight_dim )
        self._normalize = normalize

        if input_dim != weight_dim and not need_mlp:
            raise ConfigurationError( """Contradiction: input_dim != weight_dim
                but need_mlp is set to False""" )

        if self._need_mlp:
            self._mlp = nn.Linear( input_dim, weight_dim )
            self._activation = nn.Tanh()

        # weight definition
        self._weight = Parameter( torch.Tensor( weight_dim ) )
        # weight init
        self.reset_parameters()

    def reset_parameters( self ):
        bound = 1 / math.sqrt( self._weight_dim )
        nn.init.uniform_( self._weight, -bound, bound )
    
    def forward( self, x: torch.Tensor ) -> torch.Tensor:
        """
        x: shape [bs, nums, ebd], where ebd must equals to self._dims
        out: shape [bs, nums], attention weights
        """
        if x.shape[-1] != self._input_dim:
            raise ConfigurationError( """The last dim of input must equals to
                the construction parameter 'input_dim'. Please check.""" )
        
        if self._need_mlp:
            bs, nums, ebd = x.shape
            x_combine = x.reshape( bs * nums, ebd )
            x_combine = self._activation( self._mlp( x_combine ) )
            x = x_combine.reshape( bs, nums, -1 )
        
        bs = x.shape[0]
        w = self._weight.unsqueeze( 0 ).unsqueeze( 2 ) # shape [1, ebd, 1]
        w = w.expand( bs, -1, -1 ) # shape [bs, ebd, 1]

        logits = x.bmm( w ).squeeze( -1 ) # shape [bs, nums]

        if self._normalize:
            return F.softmax( logits, dim = 1 )
        else:
            return logits

    @classmethod
    def from_params( cls, params: Params ):
        input_dim = params.pop_int( "input_dim" )
        weight_dim = params.pop_int( "weight_dim" )
        need_mlp = params.pop( "need_mlp" )
        params.assert_empty( cls.__name__ )
        return cls( input_dim = input_dim,
                    weight_dim = weight_dim,
                    need_mlp = need_mlp )
