"""
Neural Tensor Network from "Reasoning with Neural Tensor Networks from Knowledge Base Completion"
"""

import typing

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError

class NeuralTensorNetwork( nn.Module ):
    """
    Compute similarities of two entities.

    Parameters
    ----------
    k: ``int``
        The output channel
    d: ``int``
        The input channel
    """
    def __init__( self, k: int,
                  d: int,
                  activation: str = "relu" ):
        super( NeuralTensorNetwork, self ).__init__()

        self._k = k
        self._d = d
        if activation == "relu":
            self._activation = F.relu
        elif activation == "tanh":
            self._activation = F.tanh
        else:
            raise ConfigurationError( "Activation type %s is not supported yet !" 
                % ( activation ) )

        # weights definition
        self._W = Parameter( torch.Tensor( k, d, d ) )
        self._V = Parameter( torch.Tensor( k, 2*d ) )
        self._b = Parameter( torch.Tensor( k ) )

        # weights init
        self.reset_parameters()

    def reset_parameters( self ):
        # imitate torch.nn.Linear
        nn.init.kaiming_uniform_( self._W, a = math.sqrt( 5 ) )
        nn.init.kaiming_uniform_( self._V, a = math.sqrt( 5 ) )

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out( self._V )
        bound = 1 / math.sqrt( fan_in )
        nn.init.uniform_( self._b, -bound, bound )

    def forward( self, x1: torch.Tensor, x2: torch.Tensor ) -> torch.Tensor:
        """
        x1: shape [B, L1, d], x2: shape [B, L2, d]
        out: shape [B, L1, L2, k]
        define L = L1 * L2
        """
        if not ( x1.shape[-1] == x2.shape[-1] == self._d ):
            raise ConfigurationError( """The channels of two inputs and the 
                construction parameter are not same! Please check.""" )
        if not ( x1.shape[0] == x2.shape[0] ):
            raise ConfigurationError( """The batch sizes of two inputs are 
                not same! Please check.""" )

        B, L1, d = x1.shape; L2 = x2.shape[1]; k = self._k

        x1_expand = x1.unsqueeze( 1 ) # [B, 1, L1, d]
        # [B, 1, L1, d] x [k, d, d] -> [B, k, L1, d]
        temp = x1_expand.matmul( self._W )
        temp_combine = temp.reshape( B * k, L1, d )
        x2_expand = x2.transpose( 1, 2 ).unsqueeze( 1 ) # [B, 1, d, L2]
        x2_expand = x2_expand.expand( -1, k, -1, -1 ).reshape( B * k, d, L2 )
        bilinear_out = temp_combine.bmm( x2_expand ).reshape( B, k, L1, L2 )

        x1_expand = x1.unsqueeze( 2 ).expand( -1, -1, L2, -1 ) # [B, L1, L2, d]
        x2_expand = x2.unsqueeze( 1 ).expand( -1, L1, -1, -1 ) # [B, L1, L2, d]
        x_combine = torch.cat( [x1_expand, x2_expand], -1 ) # [B, L1, L2, 2d]
        linear_out = ( x_combine.matmul( self._V.t() ) + self._b ).permute( 0, 3, 1, 2 )

        out = self._activation( bilinear_out + linear_out ) # [B, k, L1, L2]

        return out

    @classmethod
    def from_params( cls, params: Params ):
        k = params.pop_int( "k" )
        d = params.pop_int( "d" )
        activation = params.pop( "activation" )
        params.assert_empty( cls.__name__ )
        return cls( k = k, d = d, activation = activation )




        