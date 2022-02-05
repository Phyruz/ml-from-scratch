"""
This module contains Linear Regression Model class.

Author: Phyruz
"""
import numpy as np
import logging
from numpy import ndarray
from typing import Dict,Tuple

class LinearRegression():
    """A Linear Regression class"""
    
    batch: Dict[str, ndarray] = {}
    weights: Dict[str, ndarray] = {}
    forward_info: Dict[str, ndarray] = {}
    loss_gradients: Dict[str, ndarray] = {}

    def initialise_weights(self, nins:int) -> Tuple[ndarray]:
        return np.random.rand(nins,1), np.random.rand(1,1)

    def forward_pass(self) -> float:
        """Start from """
        # Matrix multiplication between input and weights
        self.forward_info['N'] = np.matmul(
            self.batch['X'], self.weights['W']
            )
        # Adding bias
        self.forward_info['P'] = self.forward_info['N'] + self.weights['b']

        # Calculate loss 
        self.forward_info['loss'] = np.mean((self.batch['y'] - self.forward_info['P'])**2)

    def backward_pass(self) -> ndarray:
        
        # Calculate derivative with respect to weights (dLdW)
        # Last derivative for the loss function (dLdP)
        dLdP = -2*(self.batch['y'] - self.forward_info['P'])

        # Adding bias function (dPdN)
        dPdN = np.ones_like(self.forward_info['N'])

        # Matrix multiplication operation (dNdW)
        dNdW = np.transpose(self.batch['X'])
    
        dLdN = dLdP*dPdN
        dLdW = np.matmul(dNdW,dLdN)
        
        # Derivative with respect to bias
        dPdB = np.ones_like(self.batch['y'])
        dLdB = (dLdP*dPdB).sum(axis=0)

        self.loss_gradients['W'] = dLdW
        self.loss_gradients['b'] = dLdB

    def train(self, 
            X_batch: ndarray,
            y_batch: ndarray,
            n_epochs: int,
            learning_rate: float = 0.001,
             ) -> None:
        
        self.batch['X'] = X_batch
        self.batch['y'] = y_batch
        self.weights['W'], self.weights['b'] = self.initialise_weights(X_batch.shape[1])
        self.weights['W'] = np.transpose(self.weights['W'], (0,1))
        for _ in range(n_epochs):
            self.forward_pass()
            print(self.forward_info['loss'])
            self.backward_pass()
            for key in self.weights.keys():
                self.weights[key] -= learning_rate * self.loss_gradients[key]

if __name__=='__main__':
    X_batch = np.array([[0,], [1,], [2,], [3,], [4,], [5,], [6,], [7,], [8,], [9,]])
    y_batch = np.array([[10], [12], [14], [16], [18], [20], [22], [24], [26], [28]])
    model = LinearRegression()
    model.train(X_batch, y_batch, n_epochs=3000)
    print(model.weights)
