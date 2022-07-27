"""
Implements fully connected networks in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
from re import I
import torch
from a3_helper import softmax_loss
from eecs598 import Solver


def hello_fully_connected_networks():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print('Hello from fully_connected_networks.py!')


class Linear(object):

    @staticmethod
    def forward(x, w, b):
        """
        Computes the forward pass for an linear (fully-connected) layer.
        The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
        examples, where each example x[i] has shape (d_1, ..., d_k). We will
        reshape each input into a vector of dimension D = d_1 * ... * d_k, and
        then transform it to an output vector of dimension M.
        Inputs:
        - x: A tensor containing input data, of shape (N, d_1, ..., d_k)
        - w: A tensor of weights, of shape (D, M)
        - b: A tensor of biases, of shape (M,)
        Returns a tuple of:
        - out: output, of shape (N, M)
        - cache: (x, w, b)
        """
        out = None
        ######################################################################
        # TODO: Implement the linear forward pass. Store the result in out.  #
        # You will need to reshape the input into rows.                      #
        ######################################################################
        # print(x.shape)
        # print(w.shape)
        # print(b)
        # print(x.shape)
        out = x.reshape(x.shape[0], -1) @ w + b

        ######################################################################
        #                        END OF YOUR CODE                            #
        ######################################################################
        cache = (x, w, b)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Computes the backward pass for an linear layer.
        Inputs:
        - dout: Upstream derivative, of shape (N, M)
        - cache: Tuple of:
          - x: Input data, of shape (N, d_1, ... d_k)
          - w: Weights, of shape (D, M)
          - b: Biases, of shape (M,)
        Returns a tuple of:
        - dx: Gradient with respect to x, of shape
          (N, d1, ..., d_k)
        - dw: Gradient with respect to w, of shape (D, M)
        - db: Gradient with respect to b, of shape (M,)
        """
        x, w, b = cache
        dx, dw, db = None, None, None
        ##################################################
        # TODO: Implement the linear backward pass.      #
        ##################################################
        N, M = dout.shape
        # out = x @ w + b
        
        # dout/dx
        # Key: should race back to x's dimension
        dx = dout @ w.T                #  (N,M) @ (M,D)

        # dx/dw
        dw = x.reshape(N,-1).T @ dout                # (D,N) @ (N,M) => (D, M)

        # dx/db
        db = torch.sum(dout,dim=0)     # (M, )

        listofints = list(x.shape)
        # listofints = [int(x) for x in x.shape]
        dx = dx.reshape(listofints)
        # print("dx.shape ", dx.shape)
        # print("dw.shape ", dw.shape)
        # print("db.shape ", db.shape)

        ##################################################
        #                END OF YOUR CODE                #
        ##################################################
        return dx, dw, db


class ReLU(object):

    @staticmethod
    def forward(x):
        """
        Computes the forward pass for a layer of rectified
        linear units (ReLUs).
        Input:
        - x: Input; a tensor of any shape
        Returns a tuple of:
        - out: Output, a tensor of the same shape as x
        - cache: x
        """
        out = None
        ###################################################
        # TODO: Implement the ReLU forward pass.          #
        # You should not change the input tensor with an  #
        # in-place operation.                             #
        ###################################################
        out = x.clone()
        out[x<0] = 0
        ###################################################
        #                 END OF YOUR CODE                #
        ###################################################
        cache = x
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Computes the backward pass for a layer of rectified
        linear units (ReLUs).
        Input:
        - dout: Upstream derivatives, of any shape
        - cache: Input x, of same shape as dout
        Returns:
        - dx: Gradient with respect to x
        """
        dx, x = None, cache
        #####################################################
        # TODO: Implement the ReLU backward pass.           #
        # You should not change the input tensor with an    #
        # in-place operation.                               #
        #####################################################

        # print(dout)
        # print(x)

        # dout/dx : should be dimension: (N,D)
        dx = dout.clone() 
        dx[x<=0] = 0
        # print(dout.shape)
        # print(x.shape)
        # print(dx.shape)

        #####################################################
        #                  END OF YOUR CODE                 #
        #####################################################
        return dx


class Linear_ReLU(object):

    @staticmethod
    def forward(x, w, b):
        """
        Convenience layer that performs an linear transform
        followed by a ReLU.

        Inputs:
        - x: Input to the linear layer
        - w, b: Weights for the linear layer
        Returns a tuple of:
        - out: Output from the ReLU
        - cache: Object to give to the backward pass
        """
        a, fc_cache = Linear.forward(x, w, b)
        out, relu_cache = ReLU.forward(a)
        cache = (fc_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the linear-relu convenience layer
        """
        fc_cache, relu_cache = cache
        da = ReLU.backward(dout, relu_cache)
        dx, dw, db = Linear.backward(da, fc_cache)
        return dx, dw, db


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.
    The architecure should be linear - relu - linear - softmax.
    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to PyTorch tensors.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0,
                 dtype=torch.float32, device='cpu'):
        """
        Initialize a new network.
        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        - dtype: A torch data type object; all computations will be
          performed using this datatype. float is faster but less accurate,
          so you should use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.params = {}
        self.reg = reg

        ###################################################################
        # TODO: Initialize the weights and biases of the two-layer net.   #
        # Weights should be initialized from a Gaussian centered at       #
        # 0.0 with standard deviation equal to weight_scale, and biases   #
        # should be initialized to zero. All weights and biases should    #
        # be stored in the dictionary self.params, with first layer       #
        # weights and biases using the keys 'W1' and 'b1' and second layer#
        # weights and biases using the keys 'W2' and 'b2'.                #
        ###################################################################
        # print(hidden_dims)
        
        # zeroH1 = torch.zeros(H1)
        zero = torch.zeros(hidden_dim)

        self.params["W1"] = torch.normal(0, weight_scale, size=(input_dim, hidden_dim)).to(dtype).to(device)
        self.params["b1"] = torch.zeros_like(zero).to(dtype).to(device)
        self.params["W2"] = torch.normal(0, weight_scale, size=(hidden_dim, num_classes)).to(dtype).to(device)
        self.params["b2"] = torch.zeros(num_classes).to(dtype).to(device)
        ###############################################################
        #                            END OF YOUR CODE                 #
        ###############################################################

    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'params': self.params,
        }

        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.reg = checkpoint['reg']
        for p in self.params:
            self.params[p] = self.params[p].type(dtype).to(device)
        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Tensor of input data of shape (N, d_1, ..., d_k)
        - y: int64 Tensor of labels, of shape (N,). y[i] gives the
          label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model
        and return:
        - scores: Tensor of shape (N, C) giving classification scores,
          where scores[i, c] is the classification score for X[i]
          and class c.
        If y is not None, then run a training-time forward and backward
        pass and return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping
          parameter names to gradients of the loss with respect to
          those parameters.
        """
        scores = None
        #############################################################
        # TODO: Implement the forward pass for the two-layer net,   #
        # computing the class scores for X and storing them in the  #
        # scores variable.                                          #
        #############################################################


        # Unpack variables from the params dictionary
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        N, D = X.shape

        hidden = X.mm(W1) + b1        # shape: (N, H)
        hidden[hidden <0 ] = 0 # ReLU 
        scores = hidden.mm(W2) + b2  # shape: (N, C)        
        h1 = hidden
        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = None
       

        ##############################################################
        #                     END OF YOUR CODE                       #
        ##############################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ###################################################################
        # TODO: Implement the backward pass for the two-layer net.        #
        # Store the loss in the loss variable and gradients in the grads  #
        # dictionary. Compute data loss using softmax, and make sure that #
        # grads[k] holds the gradients for self.params[k]. Don't forget   #
        # to add L2 regularization!                                       #
        #                                                                 #
        # NOTE: To ensure that your implementation matches ours and       #
        # you pass the automated tests, make sure that your L2            #
        # regularization does not include a factor of 0.5.                #
        ###################################################################
        # compute the loss and the gradient
        num_classes = b2.shape[0]
        num_train = X.shape[0]
      
        loss = 0.0
        hidden = X.mm(W1) + b1        # shape: (N, H)
        hidden[hidden <0 ] = 0 # ReLU 

        scores = hidden.mm(W2) + b2  # shape: (N, C)
        val, indices = scores.max(dim=1)
        # For stability issue
        scores -= val.reshape(-1,1)
        each_row = list(range(0, num_train))
        denom = torch.sum(torch.exp(scores), dim=1)
        losses = torch.exp(scores[each_row, y]) / denom   # get (N,)
        loss = torch.sum(-torch.log(losses))

        # dL/ds = -1 + exp(sj) / denom
        grad_matrix = torch.exp(scores) / denom.reshape(-1,1)  # (N, C) matrix
        grad_matrix[each_row, y] += -1

        ds = grad_matrix
        db2 = torch.sum(ds, dim=0) # (C,)
        dW2 = hidden.t() @ grad_matrix # (H, C)
        dh1 = grad_matrix @ W2.t()        # (N, H)
        dh1[hidden<=0] = 0 # (N,H)
        dW1 = X.t() @ dh1  # (D,H)
        db1 = torch.sum(dh1, dim=0)       # (H, )
        
        grads['W1'] = dW1
        grads['b1'] = db1
        grads['W2'] = dW2
        grads['b2'] = db2

        # Right now the loss is a sum over all training examples, but we want it
        # to be an average instead so we divide by num_train.
        loss /= num_train
        grads['W1'] /= num_train
        grads['b1'] /= num_train
        grads['W2'] /= num_train
        grads['b2'] /= num_train


        # Add regularization to the loss.
        loss += self.reg * torch.sum(W1 * W1) + self.reg * torch.sum(W2*W2)

        #############################################################################
        # TODO:                                                                     #
        # Compute the gradient of the loss function w.r.t. the regularization term  #
        # and add it to dW. (part 2)                                                #
        #############################################################################
        # Replace "pass" statement with your code
        # dloss/dw = 2 * r * W
        # dW += 2 * reg * W 
        grads['W1'] += 2 * self.reg * W1
        grads['W2'] += 2 * self.reg * W2

        ###################################################################
        #                     END OF YOUR CODE                            #
        ###################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function.
    For a network with L layers, the architecture will be:

    {linear - relu - [dropout]} x (L - 1) - linear - softmax

    where dropout is optional, and the {...} block is repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0.0, reg=0.0, weight_scale=1e-2, seed=None,
                 dtype=torch.float, device='cpu'):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each
          hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving the drop probability
          for networks with dropout. If dropout=0 then the network
          should not use dropout.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - seed: If not None, then pass this random seed to the dropout
          layers. This will make the dropout layers deteriminstic so we
          can gradient check the model.
        - dtype: A torch data type object; all computations will be
          performed using this datatype. float is faster but less accurate,
          so you should use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.use_dropout = dropout != 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        #######################################################################
        # TODO: Initialize the parameters of the network, storing all         #
        # values in the self.params dictionary. Store weights and biases      #
        # for the first layer in W1 and b1; for the second layer use W2 and   #
        # b2, etc. Weights should be initialized from a normal distribution   #
        # centered at 0 with standard deviation equal to weight_scale. Biases #
        # should be initialized to zero.                                      #
        #######################################################################

        # 15, 20  ;  20,30 ;  30, 10
        # for i from 1 to 2
        # W2 => (20, 30) 

        zeroH1 = torch.zeros(hidden_dims[0])
        self.params["W1"] = torch.normal(0, weight_scale, size=(input_dim, hidden_dims[0])).to(dtype).to(device)
        self.params["b1"] = torch.zeros_like(zeroH1).to(dtype).to(device)

        for i in range(1, len(hidden_dims)+1):
          param_w = "W" + str(i+1)
          param_b = "b" + str(i+1)
          # print(param_w)
          # print(param_b)
          dim_in = hidden_dims[i-1]
          if i == len(hidden_dims):
            dim_out = num_classes
          else:
            dim_out = hidden_dims[i]
          self.params[param_w] = torch.normal(0, weight_scale, size=(dim_in, dim_out)).to(dtype).to(device)
          zero = torch.zeros(dim_out)
          self.params[param_b] = torch.zeros_like(zero).to(dtype).to(device)


        print("hidden_dims[-1]: ", hidden_dims[-1])
        print("num_classes: ", num_classes)
        #######################################################################
        #                         END OF YOUR CODE                            #
        #######################################################################

        # When using dropout we need to pass a dropout_param dictionary
        # to each dropout layer so that the layer knows the dropout
        # probability and the mode (train / test). You can pass the same
        # dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'dtype': self.dtype,
          'params': self.params,
          'num_layers': self.num_layers,
          'use_dropout': self.use_dropout,
          'dropout_param': self.dropout_param,
        }

        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = dtype
        self.reg = checkpoint['reg']
        self.num_layers = checkpoint['num_layers']
        self.use_dropout = checkpoint['use_dropout']
        self.dropout_param = checkpoint['dropout_param']

        for p in self.params:
            self.params[p] = self.params[p].type(dtype).to(device)

        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.
        Input / output: Same as TwoLayerNet above.
        """
        X = X.to(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param
        # since they behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        scores = None
        ##################################################################
        # TODO: Implement the forward pass for the fully-connected net,  #
        # computing the class scores for X and storing them in the       #
        # scores variable.                                               #
        #                                                                #
        # When using dropout, you'll need to pass self.dropout_param     #
        # to each dropout forward pass.                                  #
        ##################################################################

        ################## Forward Pass ##################################
        num_train = X.shape[0]
        # print("num_layers: ",self.num_layers)

        scores = 0
        hiddens = []
        # Input:
        out = X.mm(self.params["W1"]) + self.params["b1"]
        out[out<0] = 0
        hiddens.append(out)
        for layer in range(1, self.num_layers-1): # Layer 2
          param_w = "W"+str(layer+1)
          param_b = "b"+str(layer+1)
          # print("\nout.shape ", out.shape)
          # print("W.shape ", self.params[param_w].shape)
          # print("b.shape ", self.params[param_b].shape)
          scores = out.mm(self.params[param_w]) + self.params[param_b]
          scores[scores<0] = 0 # ReLU      
          out = scores # update outputs
          hiddens.append(out)
        
        param_w = "W"+str(self.num_layers)
        param_b = "b"+str(self.num_layers)
        # Last layer: don't need ReLU
        scores = out.mm(self.params[param_w]) + self.params[param_b]
        hiddens.append(scores)
        ####################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        #####################################################################
        # TODO: Implement the backward pass for the fully-connected net.    #
        # Store the loss in the loss variable and gradients in the grads    #
        # dictionary. Compute data loss using softmax, and make sure that   #
        # grads[k] holds the gradients for self.params[k]. Don't forget to  #
        # add L2 regularization!                                            #
        # NOTE: To ensure that your implementation matches ours and you     #
        # pass the automated tests, make sure that your L2 regularization   #
        # includes a factor of 0.5 to simplify the expression for           #
        # the gradient.                                                     #
        #####################################################################  

        ####################### Backward Pass ###############################
        loss, dx = softmax_loss(scores, y) 
        loss /= num_train
        # print(len(hiddens))
        # for i in range(len(hiddens)):
        #   print("hiddens[",i,"] ", hiddens[i].shape)

        upstream_grad = dx
        dblast = torch.sum(upstream_grad, dim=0) # (C,)
        dWlast = hiddens[-2].t() @ upstream_grad # (H, C)

        Wlast = "W"+str(self.num_layers)
        blast = "b"+str(self.num_layers)
        grads[Wlast] = dWlast
        grads[blast] =dblast
        grads[Wlast] /= num_train
        grads[blast] /= num_train  
        grads[Wlast] += 2 * self.reg * self.params[Wlast] 
        loss += self.reg * torch.sum( self.params[Wlast]**2 )


        upstream_grad = upstream_grad @ self.params[Wlast].t()
    
        # print("Last W: upstream.shape ", upstream_grad.shape)
        upstream_grad[hiddens[-2]<=0] = 0
        # print("\nBefore for-loop: upstream_grad.shape ", upstream_grad.shape)

        for layer in range(self.num_layers-1, 1, -1):  # 2
          # print("\n\nlayer: ", layer)
          param_w = "W"+str(layer)
          param_b = "b"+str(layer)
          local_grad = hiddens[layer-2].t()

          # print("local_grad ", local_grad.shape)
          # print("upstream_grad ", upstream_grad.shape)
          dw = local_grad @ upstream_grad
          db = torch.sum(upstream_grad, dim=0)
          grads[param_w] = dw
          grads[param_b] = db
          grads[param_w] /= num_train
          grads[param_b] /= num_train   

          # print("gradsW.shape ", grads[param_w].shape)
          # print("param_w.shape: ", self.params[param_w].shape)
          grads[param_w] += 2 * self.reg * self.params[param_w] 
          
          loss += self.reg * torch.sum( self.params[param_w]**2 )

          upstream_grad = upstream_grad @ self.params[param_w].t()
          # print("upgrad.shape ", upstream_grad.shape)

       
        # Now upstream_grad = (2, 20)  W2: 20,30
        # upstream_grad = upstream_grad @ W2.t()  # (2,) @ (30,20) = (2, 20)
        W2 = self.params["W2"]
        upstream_grad[hiddens[0]<=0] = 0          # Should be (2, 20)
        dW1 = X.t() @ upstream_grad               # (D, N) @ (N,H)  (D,H)  (15,2)@(2,20)         (15, 20)
        db1 = torch.sum(upstream_grad, dim=0)     # (H, )
        grads["W1"] = dW1
        grads["b1"] = db1
        grads["W1"] /= num_train
        grads["b1"] /= num_train   
        grads["W1"] += 2 * self.reg * self.params["W1"] 
        loss += self.reg * torch.sum( self.params["W1"]**2 )

        ###########################################################
        #                   END OF YOUR CODE                      #
        ###########################################################

        return loss, grads

  



def create_solver_instance(data_dict, dtype, device):
    model = TwoLayerNet(hidden_dim=200, dtype=dtype, device=device)
    #############################################################
    # TODO: Use a Solver instance to train a TwoLayerNet that   #
    # achieves at least 50% accuracy on the validation set.     #
    #############################################################
    solver = Solver(model, data_dict, 
                    update_rule=sgd,
                    optim_config={'learning_rate': 1e-1},
                    lr_decay=0.95,
                    num_epochs=10, batch_size=64,
                    print_every=100,
                    device= device)
    ##############################################################
    #                    END OF YOUR CODE                        #
    ##############################################################
    return solver





def get_three_layer_network_params():
    ###############################################################
    # TODO: Change weight_scale and learning_rate so your         #
    # model achieves 100% training accuracy within 20 epochs.     #
    ###############################################################
    weight_scale = 1e-2   # Experiment with this!
    learning_rate = 1e-4  # Experiment with this!
    # Replace "pass" statement with your code
    pass
    ################################################################
    #                             END OF YOUR CODE                 #
    ################################################################
    return weight_scale, learning_rate


def get_five_layer_network_params():
    ################################################################
    # TODO: Change weight_scale and learning_rate so your          #
    # model achieves 100% training accuracy within 20 epochs.      #
    ################################################################
    learning_rate = 2e-3  # Experiment with this!
    weight_scale = 1e-5   # Experiment with this!
    # Replace "pass" statement with your code
    pass
    ################################################################
    #                       END OF YOUR CODE                       #
    ################################################################
    return weight_scale, learning_rate


def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.
    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)

    w -= config['learning_rate'] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    """
    Performs stochastic gradient descent with momentum.
    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to
      store a moving average of the gradients.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    v = config.get('velocity', torch.zeros_like(w))

    next_w = None
    ##################################################################
    # TODO: Implement the momentum update formula. Store the         #
    # updated value in the next_w variable. You should also use and  #
    # update the velocity v.                                         #
    ##################################################################
    v = config['momentum'] * v - config['learning_rate'] * dw # integrate velocity
    next_w = w + v # integrate position
    ###################################################################
    #                           END OF YOUR CODE                      #
    ###################################################################
    config['velocity'] = v

    return next_w, config


def rmsprop(w, dw, config=None):
    """
    Uses the RMSProp update rule, which uses a moving average of squared
    gradient values to set adaptive per-parameter learning rates.
    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('decay_rate', 0.99)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('cache', torch.zeros_like(w))

    next_w = None
    ###########################################################################
    # TODO: Implement the RMSprop update formula, storing the next value of w #
    # in the next_w variable. Don't forget to update cache value stored in    #
    # config['cache'].                                                        #
    ###########################################################################
    config['cache'] = config['decay_rate']* config['cache'] + (1-config['decay_rate'])*dw*dw
    next_w = w - (config['learning_rate']*dw) / ( config['cache']**0.5 + config['epsilon'])
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return next_w, config


def adam(w, dw, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.
    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', torch.zeros_like(w))
    config.setdefault('v', torch.zeros_like(w))
    config.setdefault('t', 0)

    next_w = None
    ##########################################################################
    # TODO: Implement the Adam update formula, storing the next value of w in#
    # the next_w variable. Don't forget to update the m, v, and t variables  #
    # stored in config.                                                      #
    #                                                                        #
    # NOTE: In order to match the reference output, please modify t _before_ #
    # using it in any calculations.                                          #
    ##########################################################################
    config['t'] += 1
    config['m'] = config['beta1']*config['m'] + (1-config['beta1'])*dw
    config['v'] = config['beta2']*config['v'] + (1-config['beta2'])*dw*dw
    m_unbiased = config['m'] / (1- config['beta1']**config['t'])
    v_unbiased = config['v'] / (1- config['beta2']**config['t'])
    next_w = w - (config['learning_rate']*m_unbiased) / (v_unbiased**0.5 + config['epsilon'])
    
    #########################################################################
    #                              END OF YOUR CODE                         #
    #########################################################################

    return next_w, config


class Dropout(object):

    @staticmethod
    def forward(x, dropout_param):
        """
        Performs the forward pass for (inverted) dropout.
        Inputs:
        - x: Input data: tensor of any shape
        - dropout_param: A dictionary with the following keys:
          - p: Dropout parameter. We *drop* each neuron output with
            probability p.
          - mode: 'test' or 'train'. If the mode is train, then
            perform dropout;
          if the mode is test, then just return the input.
          - seed: Seed for the random number generator. Passing seed
            makes this
            function deterministic, which is needed for gradient checking
            but not in real networks.
        Outputs:
        - out: Tensor of the same shape as x.
        - cache: tuple (dropout_param, mask). In training mode, mask
          is the dropout mask that was used to multiply the input; in
          test mode, mask is None.
        NOTE: Please implement **inverted** dropout, not the vanilla
              version of dropout.
        See http://cs231n.github.io/neural-networks-2/#reg for more details.
        NOTE 2: Keep in mind that p is the probability of **dropping**
                a neuron output; this might be contrary to some sources,
                where it is referred to as the probability of keeping a
                neuron output.
        """
        p, mode = dropout_param['p'], dropout_param['mode']
        if 'seed' in dropout_param:
            torch.manual_seed(dropout_param['seed'])

        mask = None
        out = None

        if mode == 'train':
            ##############################################################
            # TODO: Implement training phase forward pass for            #
            # inverted dropout.                                          #
            # Store the dropout mask in the mask variable.               #
            ##############################################################
            # print("x.shape ", x.shape)
            mask = (torch.rand(*x.shape, device=x.device) <(1-p) ) / (1-p)
            # print(mask[:5,:5])
            out = mask * x
            
            ##############################################################
            #                   END OF YOUR CODE                         #
            ##############################################################
        elif mode == 'test':
            ##############################################################
            # TODO: Implement the test phase forward pass for            #
            # inverted dropout.                                          #
            ##############################################################
            out = x
            ##############################################################
            #                      END OF YOUR CODE                      #
            ##############################################################

        cache = (dropout_param, mask)

        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Perform the backward pass for (inverted) dropout.
        Inputs:
        - dout: Upstream derivatives, of any shape
        - cache: (dropout_param, mask) from Dropout.forward.
        """
        dropout_param, mask = cache
        mode = dropout_param['mode']

        dx = None
        if mode == 'train':
            ###########################################################
            # TODO: Implement training phase backward pass for        #
            # inverted dropout                                        #
            ###########################################################
            # print("dout.shape ", dout.shape)
            # print(mask)
            dx = dout * mask

            ###########################################################
            #                     END OF YOUR CODE                    #
            ###########################################################
        elif mode == 'test':
            dx = dout
        return dx
