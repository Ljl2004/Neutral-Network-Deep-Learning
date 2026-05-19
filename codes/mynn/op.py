from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        return X @ self.W + self.b

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        self.grads['W'] = self.input.T @ grad
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)
        return grad @ self.W.T
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Vectorized implementation using im2col.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.W = initialize_method(size=(out_channels, in_channels, kernel_size, kernel_size))
        self.b = np.zeros((1, out_channels))
        self.grads = {'W': None, 'b': None}
        self.input = None
        self.col = None
        self.params = {'W': self.W, 'b': self.b}

        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def _im2col(self, X):
        batch, C, H, W = X.shape
        k = self.kernel_size
        s = self.stride
        H_out = (H - k) // s + 1
        W_out = (W - k) // s + 1

        col = np.zeros((batch, C, k * k, H_out, W_out))
        idx = 0
        for ki in range(k):
            for kj in range(k):
                col[:, :, idx, :, :] = X[:, :, ki:ki + H_out * s:s, kj:kj + W_out * s:s]
                idx += 1

        col = col.transpose(0, 3, 4, 1, 2).reshape(batch * H_out * W_out, C * k * k)
        return col, H_out, W_out

    def forward(self, X):
        self.input = X
        batch_size = X.shape[0]
        k = self.kernel_size

        self.col, H_out, W_out = self._im2col(X)

        W_col = self.W.reshape(self.out_channels, -1)  # [outC, inC * k * k]
        out = self.col @ W_col.T + self.b  # [N, outC]
        out = out.reshape(batch_size, self.out_channels, H_out, W_out)
        return out

    def backward(self, grads):
        batch_size, _, H_out, W_out = grads.shape
        k = self.kernel_size
        s = self.stride

        grad_flat = grads.reshape(batch_size * H_out * W_out, self.out_channels)  # [N, outC]
        W_col = self.W.reshape(self.out_channels, -1)  # [outC, inC * k * k]

        self.grads['W'] = (grad_flat.T @ self.col).reshape(self.out_channels, self.in_channels, k, k)
        self.grads['b'] = np.sum(grad_flat, axis=0, keepdims=True)

        grad_col = grad_flat @ W_col  # [N, inC * k * k]
        grad_input = self._col2im(grad_col, self.input.shape)
        return grad_input

    def _col2im(self, col, X_shape):
        batch, C, H, W = X_shape
        k = self.kernel_size
        s = self.stride
        H_out = (H - k) // s + 1
        W_out = (W - k) // s + 1

        col = col.reshape(batch, H_out, W_out, C, k * k).transpose(0, 3, 4, 1, 2)

        grad_X = np.zeros((batch, C, H, W))
        idx = 0
        for ki in range(k):
            for kj in range(k):
                grad_X[:, :, ki:ki + H_out * s:s, kj:kj + W_out * s:s] += col[:, :, idx, :, :]
                idx += 1

        return grad_X

    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}
        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        super().__init__()
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True
        self.optimizable = False
        self.probs = None
        self.labels = None

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)

    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        self.labels = labels
        batch_size = predicts.shape[0]

        if self.has_softmax:
            self.probs = softmax(predicts)
        else:
            self.probs = predicts

        loss = -np.log(self.probs[np.arange(batch_size), labels] + 1e-8).mean()
        return loss

    def backward(self):
        batch_size = self.probs.shape[0]
        self.grads = self.probs.copy()
        self.grads[np.arange(batch_size), self.labels] -= 1
        self.grads /= batch_size

        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class Flatten(Layer):
    """Flattens input from [batch, C, H, W] to [batch, C*H*W]"""
    def __init__(self) -> None:
        super().__init__()
        self.input_shape = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, grads):
        return grads.reshape(self.input_shape)


class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition