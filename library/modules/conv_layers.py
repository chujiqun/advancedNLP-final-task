"""
Hierarchical convolution layers.
"""
from typing import Union, Tuple, Sequence

import torch
import torch.nn as nn

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.nn import Activation

from ipdb import set_trace

class ConvLayers( nn.Module ):
    """
    Multiple convolution layers applied in image-like tensors.

    Parameters
    ----------
    input_channel: ``int``
        Channel of input tensor.
    kernel_nums: ``List[int]``
        number of hierarchical convolution kernels
    kernel_sizes: ``List[Tuple[int, int]]``
        Multiple convolution kernels
    """
    def __init__(
        self,
        input_channel: int,
        num_layers: int,
        kernel_nums: Union[int, Sequence[int]],
        kernel_sizes: Union[Tuple[int, int], Sequence[Tuple[int, int]]],
        activations: Union[Activation, Sequence[Activation]] ) -> None:
    
        super( ConvLayers, self ).__init__()
        if not isinstance( kernel_nums, list ):
            kernel_nums = [kernel_nums] * num_layers
        if not isinstance( kernel_sizes, list ):
            kernel_sizes = [kernel_sizes] * num_layers
        if not isinstance(activations, list):
            activations = [activations] * num_layers
        
        if len( kernel_nums ) != num_layers:
            raise ConfigurationError( "len(kernel_nums) (%d) != num_layers (%d)" %
                                     ( len( kernel_nums ), num_layers ) )
        if len( kernel_sizes ) != num_layers:
            raise ConfigurationError( "len(kernel_sizes) (%d) != num_layers (%d)" %
                                     ( len( kernel_sizes ), num_layers ) )
        if len( activations ) != num_layers:
            raise ConfigurationError( "len(activations) (%d) != num_layers (%d)" %
                                     ( len( activations ), num_layers ) )

        self._activations = activations
        input_channels = [input_channel] + kernel_nums[:-1]
        conv_layers = []
        bn_layers = []
        for in_channel, out_channel, kernel_size in zip( input_channels, kernel_nums, kernel_sizes ):
            padding = [( ks - 1 ) // 2 for ks in kernel_size] # 为了尽量保持卷积前的大小
            conv_layers.append( nn.Conv2d( in_channel, out_channel,
                kernel_size, padding = padding ) )
            bn_layers.append( nn.BatchNorm2d( out_channel ) )
        self._conv_layers = nn.ModuleList( conv_layers )
        self._bn_layers = nn.ModuleList( bn_layers )
        
    def forward( self, inputs: torch.Tensor ) -> torch.Tensor:
        """
        inputs: 4D tensor, shape: [B, C, H, W], channel first
        """
        output = inputs
        for conv, bn, activation in zip( self._conv_layers, self._bn_layers, self._activations ):
            output = activation( bn( conv( output ) ) )
        return output

    @classmethod
    def from_params( cls, params: Params ):
        input_channel = params.pop_int( "input_channel" )
        num_layers = params.pop_int( 'num_layers' )
        kernel_nums = params.pop( "kernel_nums" )
        kernel_sizes = params.pop( "kernel_sizes" )
        activations = params.pop( "activations" )
        if isinstance( activations, list ):
            activations = [Activation.by_name( name )() for name in activations]
        else:
            activations = Activation.by_name( activations )()
        params.assert_empty( cls.__name__ )
        return cls( input_channel = input_channel,
                    num_layers = num_layers,
                    kernel_nums = kernel_nums,
                    kernel_sizes = kernel_sizes,
                    activations = activations )


        

