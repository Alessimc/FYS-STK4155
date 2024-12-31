"""
Class to perform logistic regression
Based on Mortens notes, then expanded with regularization and SGD
"""

import numpy as np
from scheduler import *


class HomeMadeLogisticRegression:
    """
    A binary logistic regression classifier with mini-batch stochastic gradient descent,
    L2 regularization, and learning rate schedulers.

    Parameters
    ----------
    scheduler : Scheduler, optional (default=Constant(0.01))
        Learning rate scheduler to use for optimization.
    n_epochs : int, default=100
        Number of passes over the training data.
    n_batches : int, default=10
        Number of batches to divide the training data into per epoch.
    lambda_reg : float, default=0.01
        L2 regularization strength. Larger values specify stronger regularization.
    """
    def __init__(self, scheduler=None, n_epochs=100, n_batches=10, lambda_reg=0.01):
        # If no scheduler is provided, use default Constant scheduler
        self.scheduler = scheduler if scheduler is not None else Constant(0.01)
        self.n_epochs = n_epochs
        self.n_batches = n_batches  
        self.lambda_reg = lambda_reg
        self.beta_logreg = None
        
    def sigmoid(self, z):
        """
        Compute the sigmoid function.

        Parameters
        ----------
        z : ndarray
            Input values to transform.

        Returns
        -------
        ndarray
            Sigmoid transformation of input values, element-wise.
        """
        return 1 / (1 + np.exp(-z))
    
    def SGDfit(self, X, y):
        """
        Fit the logistic regression model using mini-batch stochastic gradient descent 
        with the specified learning rate scheduler.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target values (0 or 1).

        Returns
        -------
        list
            Training losses for each epoch.
        """
        n_data, num_features = X.shape
        self.beta_logreg = np.zeros(num_features)
        
        # Calculate batch size based on number of batches
        self.batch_size = max(n_data // self.n_batches, 1)
        # Adjust n_batches if dataset is too small
        self.actual_n_batches = max(n_data // self.batch_size, 1)
        
        if self.actual_n_batches != self.n_batches:
            print(f"Warning: Adjusted number of batches from {self.n_batches} to {self.actual_n_batches} due to dataset size")
        
        losses = []
        
        # Reset scheduler at the start of training
        self.scheduler.reset()
        
        for epoch in range(self.n_epochs):
            # Shuffle the data at the start of each epoch
            shuffle_idx = np.random.permutation(n_data)
            X_shuffled = X[shuffle_idx]
            y_shuffled = y[shuffle_idx]
            
            epoch_loss = 0
            # Mini-batch training within each epoch
            for batch in range(self.actual_n_batches):
                start_idx = batch * self.batch_size
                end_idx = min(start_idx + self.batch_size, n_data)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward pass
                linear_model = X_batch @ self.beta_logreg
                y_predicted = self.sigmoid(linear_model)
                
                # Gradient calculation with L2 regularization
                batch_size = end_idx - start_idx
                gradient = (X_batch.T @ (y_predicted - y_batch))/batch_size
                regularization_term = self.lambda_reg * self.beta_logreg
                
                # Use scheduler to update gradient
                update = self.scheduler.update_change(gradient + regularization_term)
                
                # Update beta_logreg
                self.beta_logreg -= update
                
                # Calculate loss for monitoring
                batch_loss = -np.mean(
                    y_batch * np.log(y_predicted + 1e-15) + 
                    (1 - y_batch) * np.log(1 - y_predicted + 1e-15)
                )
                epoch_loss += batch_loss
            
            # Average loss for the epoch
            epoch_loss /= self.actual_n_batches
            losses.append(epoch_loss)
            
            # Print progress every few epochs
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.n_epochs}, Loss: {epoch_loss:.4f}")
        
        return losses
    
    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        list
            Predicted class labels (0 or 1) for each sample.
        """
        linear_model = X @ self.beta_logreg
        y_predicted = self.sigmoid(linear_model)
        return [1 if i >= 0.5 else 0 for i in y_predicted]
