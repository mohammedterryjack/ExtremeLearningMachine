############   NATIVE IMPORTS  ###########################
############ INSTALLED IMPORTS ###########################
from numpy import ndarray, exp, save, load
from numpy.linalg import pinv
from numpy.random import seed, uniform
############   LOCAL IMPORTS   ###########################
##########################################################
class ELMClassifier:
    """
    a shallow neural network with only one (wide) hidden layer of random weights
    the input features are nonlinearly projected into a larger space via the random hidden layer
    and the final layer has an easier time to fit the features in the larger space to the desired output behaviour
    The final layer uses pseudoinverse to learn in one go (extreme learning) as opposed to iteratively (i.e. backprop)
    """
    def __init__(self, input_layer_width: int, hidden_layer_width:int = 1000) -> None:
        seed(0) 
        self.input_layer_to_hidden_layer_weights = uniform(
            low=-.1, high=.1, size =[
                input_layer_width, 
                hidden_layer_width
            ]
        )
    
    def fit(self, inputs: ndarray, outputs: ndarray) -> None:
        hidden = self._hidden_layer(inputs)
        self.hidden_layer_to_output_layer_weights = pinv(hidden) @ outputs
        self.save_trained_weights(weights=self.hidden_layer_to_output_layer_weights)
    
    def load_trained_weights(self) -> None:
        self.hidden_layer_to_output_layer_weights = load(
            'src/hidden_layer_to_output_layer_weights.npy', 
            allow_pickle = False
        )

    def infer(self, inputs: ndarray) -> ndarray:
        return self._output_layer(
            hidden = self._hidden_layer(inputs)
        )

    def _hidden_layer(self, inputs: ndarray) -> ndarray: 
        return self.activation_function(
            inputs @ self.input_layer_to_hidden_layer_weights
        )
  
    def _output_layer(self, hidden: ndarray) -> ndarray: 
        return hidden @ self.hidden_layer_to_output_layer_weights
    
    @staticmethod
    def save_trained_weights(weights:ndarray) -> None:
        save("src/hidden_layer_to_output_layer_weights", weights, allow_pickle = False)

    @staticmethod
    def activation_function(x: ndarray) -> ndarray: 
        return ELMClassifier.sigmoid(x)
    
    @staticmethod
    def sigmoid(x: ndarray) -> ndarray:
        return 1. / (1. + exp(-x))
    