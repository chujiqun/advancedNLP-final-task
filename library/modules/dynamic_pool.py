"""
Dynamic pooling of "Dynamic Pooling and Unfolding RAEs for Paraphrase Detection"
"""
from typing import Union, Tuple, Sequence

import torch
import torch.nn as nn

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError

class DynamicPool( nn.Module ):
    """
    Do dynamic pooling to map to a fix-length representation.

    Parameters
    ----------
    pool_type: ``str``
        Do average pooling or max pooling
    output_shape : ``Union[int, Tuple[int, int]]`` 
    """
    def __init__( self, pool_type: str, 
                  output_shape: Union[int, Tuple[int, int]] ):

        super( DynamicPool, self ).__init__()
        
        if pool_type == "average":
            self._pool = nn.AdaptiveAvgPool2d( output_shape )
        elif pool_type == "max":
            self._pool = nn.AdaptiveMaxPool2d( output_shape )
        else:
            raise ConfigurationError( "Pooling type %s not supported!" % pool_type )


    def forward( self, inputs: torch.Tensor ) -> torch.Tensor:
        """
        inputs: 4D tensor, shape: [B, C, H, W], channel first
        output: 2D tensor, shape: [B, C*H^'*W^']
        """
        return torch.flatten( self._pool( inputs ), start_dim = 1  )

    @classmethod
    def from_params( cls, params: Params ):
        pool_type = params.pop( "pool_type" )
        output_shape = params.pop( "output_shape" )
        params.assert_empty( cls.__name__ )
        return cls( pool_type = pool_type,
                    output_shape = output_shape )