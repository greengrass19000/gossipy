from collections import OrderedDict
import torch
from torch.nn import Module, Linear, Sequential
from torch.nn.init import xavier_uniform_
from torch.nn.modules.activation import ReLU, Sigmoid
from typing import Tuple
from . import TorchModel

# AUTHORSHIP
__version__ = "0.0.1"
__author__ = "Mirko Polato"
__copyright__ = "Copyright 2022, gossipy"
__license__ = "MIT"
__maintainer__ = "Mirko Polato, PhD"
__email__ = "mak1788@gmail.com"
__status__ = "Development"
#


__all__ = ["TorchPerceptron",
           "TorchMLP",
           "AdaLine",
           "Pegasos",
           "LogisticRegression"]

class TorchPerceptron(TorchModel):
    def __init__(self,
                 dim: int,
                 activation: torch.nn.modules.activation=Sigmoid,
                 bias: bool=True):
        """Perceptron model.

        Implementation of the perceptron model by Rosenblatt [1]_.

        Parameters
        ----------
        dim : int
            The number of input features.
        activation : torch.nn.modules.activation, default=Sigmoid()
            The activation function of the output neuron.
        bias : bool, optional
            Whether to add a bias term to the output neuron.
        
        References
        ----------
        .. [1] Rosenblatt, Frank. "The perceptron: a probabilistic model for information storage and 
           organization in the brain". Psychological review 65 6 (1958): 386-408.
        """

        super(TorchPerceptron, self).__init__()
        self.input_dim = dim
        self.model = Sequential(OrderedDict({
            "linear" : Linear(self.input_dim, 1, bias=bias), 
            "sigmoid" : activation()
        }))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def init_weights(self) -> None:
        xavier_uniform_(self.model._modules['linear'].weight)
    
    def __repr__(self) -> str:
        return str(self)
    
    def __str__(self) -> str:
        return "TorchPerceptron(size=%d)\n%s" %(self.get_size(), str(self.model))


class TorchMLP(TorchModel):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dims: Tuple[int]=(100,),
                 activation: Module=ReLU):
        """Multi-layer perceptron model.

        Implementation of the multi-layer perceptron model. The model is composed of a sequence of
        linear layers with the specified activation function (same activation for all layers but the
        last one).

        Parameters
        ----------
        input_dim : int
            The number of input features.
        output_dim : int
            The number of output neurons.
        hidden_dims : Tuple[int], default=(100,)
            The number of hidden neurons in each hidden layer.
        activation : torch.nn.modules.activation, default=ReLU
            The activation function of the hidden layers.
        """

        super(TorchMLP, self).__init__()
        dims = [input_dim] + list(hidden_dims)
        layers = OrderedDict()
        for i in range(len(dims)-1):
            layers["linear_%d" %(i+1)] = Linear(dims[i], dims[i+1])
            layers["activ_%d" %(i+1)] = activation()
        layers["linear_%d" %len(dims)] = Linear(dims[len(dims)-1], output_dim)
        #layers["softmax"] = Softmax(1)
        self.model = Sequential(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def init_weights(self) -> None:
        def _init_weights(m: Module):
            if type(m) == Linear:
                xavier_uniform_(m.weight)
        self.model.apply(_init_weights)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(size=%d)\n%s" %(self.get_size(), str(self.model))


class AdaLine(TorchModel):
    def __init__(self, dim: int):
        """The Adaline perceptron model.

        Implementation of the AdaLine perceptron model [1]_ [2]_.
        The model is a simple perceptron with a linear activation function.

        Parameters
        ----------
        dim : int
            The number of input features.
        
        References
        ----------
        .. [1] Bernard Widrow and Marcian E. Hoff. 1988. Adaptive switching circuits.
           Neurocomputing: foundations of research. MIT Press, Cambridge, MA, USA, 123–134.
        .. [2] Ormándi, R., Hegedűs, I. and Jelasity, M. (2013), Gossip learning with linear models 
           on fully distributed data. Concurrency Computat.: Pract. Exper., 25: 556-571.
           https://doi.org/10.1002/cpe.2858
        """
        
        super(AdaLine, self).__init__()
        self.input_dim = dim
        self.model = torch.nn.Parameter(torch.zeros(self.input_dim), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model @ x.T

    def get_size(self) -> int:
        return self.input_dim

    def init_weights(self) -> None:
        pass



class LogisticRegression(TorchModel):
    def __init__(self, input_dim: int, output_dim: int):
        """Logistic regression model.
        
        Implementation of the logistic regression model.
        
        Parameters
        ----------
        input_dim : int
            The number of input features.
        output_dim : int
            The number of output neurons.
        """

        super(LogisticRegression, self).__init__()
        self.model = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def init_weights(self) -> None:
        pass
    
    def __str__(self) -> str:
        return "LogisticRegression(in_size=%d, out_size=%d)" %(self.model.in_features,
                                                               self.model.out_features)
