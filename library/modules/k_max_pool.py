"""
k-max pooling: return k max values from one channel of feature maps 
    and concatenate them. 
"""

import torch
import torch.nn as nn

from allennlp.common import Params

class kMaxPool( nn.Module ):
    """
    Do k-max pooling to map to a fix-length representation.

    Parameters
    ----------
    k: ``int``
    """

    def __init__( self, k: int ):

        super( kMaxPool, self ).__init__()

        self._k = k

    def forward( self, inputs: torch.Tensor ) -> torch.Tensor:
        """
        inputs: shape [B, C, H, W]
        output: shape [B, C * k]
        """
        B, C, H, W = inputs.shape
        inputs_combine = inputs.view( B, C, H*W )
        output = inputs_combine.topk( self._k )[0].view( B, C * self._k )

        return output 

    @classmethod
    def from_params( cls, params: Params ):
        k = params.pop_int( "k" )
        params.assert_empty( cls.__name__ )
        return cls( k = k )
