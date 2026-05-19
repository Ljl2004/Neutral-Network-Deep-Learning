from .op import *
import pickle

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        self.size_list = size_list
        self.act_func = act_func

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        for i in range(len(self.size_list) - 1):
            self.layers = []
            for i in range(len(self.size_list) - 1):
                layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
                layer.W = param_list[i + 2]['W']
                layer.b = param_list[i + 2]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_list[i + 2]['weight_decay']
                layer.weight_decay_lambda = param_list[i+2]['lambda']
                if self.act_func == 'Logistic':
                    raise NotImplemented
                elif self.act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(self.size_list) - 2:
                    self.layers.append(layer_f)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
        

class Model_CNN(Layer):
    """
    A model with conv2D layers. Implement it using the operators you have written in op.py
    """
    def __init__(self, conv_config=None, linear_config=None, input_shape=(1, 28, 28)):
        self.conv_config = conv_config
        self.linear_config = linear_config
        self.input_shape = input_shape
        self.layers = []

        if conv_config is not None and linear_config is not None:
            in_channels = input_shape[0]

            for cfg in conv_config:
                k = cfg['kernel_size']
                he_std = np.sqrt(2.0 / (in_channels * k * k))
                layer = conv2D(in_channels=in_channels, out_channels=cfg['out_channels'],
                               kernel_size=k, stride=cfg.get('stride', 1),
                               initialize_method=lambda size, s=he_std: np.random.normal(size=size) * s)
                self.layers.append(layer)
                self.layers.append(ReLU())
                in_channels = cfg['out_channels']

            self.layers.append(Flatten())

            flat_dim = self._calc_flat_dim(input_shape)
            in_dim = flat_dim
            for i, out_dim in enumerate(linear_config):
                he_std = np.sqrt(2.0 / in_dim)
                layer = Linear(in_dim=in_dim, out_dim=out_dim,
                               initialize_method=lambda size, s=he_std: np.random.normal(size=size) * s)
                self.layers.append(layer)
                if i < len(linear_config) - 1:
                    self.layers.append(ReLU())
                in_dim = out_dim

    def _calc_flat_dim(self, shape):
        C, H, W = shape
        for cfg in self.conv_config:
            k = cfg['kernel_size']
            s = cfg.get('stride', 1)
            H = (H - k) // s + 1
            W = (W - k) // s + 1
            C = cfg['out_channels']
        return C * H * W

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], *self.input_shape)

        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            saved = pickle.load(f)
        self.conv_config = saved['conv_config']
        self.linear_config = saved['linear_config']
        self.input_shape = saved['input_shape']

        self.layers = []
        in_channels = self.input_shape[0]
        for cfg in self.conv_config:
            layer = conv2D(in_channels=in_channels, out_channels=cfg['out_channels'],
                           kernel_size=cfg['kernel_size'], stride=cfg.get('stride', 1))
            self.layers.append(layer)
            self.layers.append(ReLU())
            in_channels = cfg['out_channels']

        self.layers.append(Flatten())

        flat_dim = self._calc_flat_dim(self.input_shape)
        in_dim = flat_dim
        for i, out_dim in enumerate(self.linear_config):
            layer = Linear(in_dim=in_dim, out_dim=out_dim)
            self.layers.append(layer)
            if i < len(self.linear_config) - 1:
                self.layers.append(ReLU())
            in_dim = out_dim

        param_idx = 0
        for layer in self.layers:
            if layer.optimizable:
                saved_layer = saved['params'][param_idx]
                layer.W = saved_layer['W']
                layer.b = saved_layer['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = saved_layer['weight_decay']
                layer.weight_decay_lambda = saved_layer['lambda']
                param_idx += 1

    def save_model(self, save_path):
        saved = {
            'conv_config': self.conv_config,
            'linear_config': self.linear_config,
            'input_shape': self.input_shape,
            'params': []
        }
        for layer in self.layers:
            if layer.optimizable:
                saved['params'].append({
                    'W': layer.params['W'],
                    'b': layer.params['b'],
                    'weight_decay': layer.weight_decay,
                    'lambda': layer.weight_decay_lambda
                })

        with open(save_path, 'wb') as f:
            pickle.dump(saved, f)