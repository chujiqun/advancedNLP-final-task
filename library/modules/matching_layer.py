"""
Matching layer specific for text matching task.
"""
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError

class MatchingLayer( nn.Module ):
    """
    Layer that computes a matching matrix between samples in two tensors.

    Parameters
    ----------
    matching_type: ``str``
        Different methods to calculate tensor similarities.
    """
    
    def __init__( self,
                  matching_type: str ) -> None:
        super( MatchingLayer, self ).__init__()
        self._matching_type = matching_type
        self._valid_type = ["dot", "cos"]
        if matching_type not in self._valid_type:
            raise ConfigurationError( "Matching type of %s is not supported yet !" % matching_type )
    
    def forward( self, x1: torch.Tensor, x2: torch.Tensor ) -> torch.Tensor:
        """
        x1: shape [B, L1, C], x2: shape [B, L2, C] 
        """
        if self._matching_type == "cos":
            x1 = F.normalize( x1, dim = 2 )
            x2 = F.normalize( x2, dim = 2 )
        return torch.einsum( "abd,acd->abc", x1, x2 ) # shappe [B, L1, L2]

    @classmethod
    def from_params( cls, params: Params ):
        matching_type = params.pop( "matching_type" )
        params.assert_empty( cls.__name__ )
        return cls( matching_type = matching_type )
                
