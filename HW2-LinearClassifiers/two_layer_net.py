"""
Implements a two-layer Neural Network classifier in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
import torch
import random
import statistics
from linear_classifier import sample_batch
from typing import Dict, List, Callable, Optional


def hello_two_layer_net():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Hello from two_layer_net.py!")


# Template class modules that we will use later: Do not edit/modify this class
class TwoLayerNet(object):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dtype: torch.dtype = torch.float32,
        device: str = "cuda",
        std: float = 1e-4,
    ):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        - dtype: Optional, data type of each initial weight params
        - device: Optional, whether the weight params is on GPU or CPU
        - std: Optional, initial weight scaler.
        """
        # reset seed before start
        random.seed(0)
        torch.manual_seed(0)

        self.params = {}
        self.params["W1"] = std * torch.randn(
            input_size, hidden_size, dtype=dtype, device=device
        )
        self.params["b1"] = torch.zeros(hidden_size, dtype=dtype, device=device)
        self.params["W2"] = std * torch.randn(
            hidden_size, output_size, dtype=dtype, device=device
        )
        self.params["b2"] = torch.zeros(output_size, dtype=dtype, device=device)

    def loss(
        self,
        X: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        reg: float = 0.0,
    ):
        return nn_forward_backward(self.params, X, y, reg)

    def train(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        learning_rate: float = 1e-3,
        learning_rate_decay: float = 0.95,
        reg: float = 5e-6,
        num_iters: int = 100,
        batch_size: int = 200,
        verbose: bool = False,
    ):
        # fmt: off
        return nn_train(
            self.params, nn_forward_backward, nn_predict, X, y,
            X_val, y_val, learning_rate, learning_rate_decay,
            reg, num_iters, batch_size, verbose,
        )

        
        # fmt: on

    def predict(self, X: torch.Tensor):
        return nn_predict(self.params, nn_forward_backward, X)

    def save(self, path: str):
        torch.save(self.params, path)
        print("Saved in {}".format(path))

    def load(self, path: str):
        checkpoint = torch.load(path, map_location="cpu")
        self.params = checkpoint
        if len(self.params) != 4:
            raise Exception("Failed to load your checkpoint")

        for param in ["W1", "b1", "W2", "b2"]:
            if param not in self.params:
                raise Exception("Failed to load your checkpoint")
        # print("load checkpoint file: {}".format(path))


def nn_forward_pass(params: Dict[str, torch.Tensor], X: torch.Tensor):
    """
    The first stage of our neural network implementation: Run the forward pass
    of the network to compute the hidden layer features and classification
    scores. The network architecture should be:

    FC layer -> ReLU (hidden) -> FC layer (scores)

    As a practice, we will NOT allow to use torch.relu and torch.nn ops
    just for this time (you can use it from A3).

    Inputs:
    - params: a dictionary of PyTorch Tensor that store the weights of a model.
      It should have following keys with shape
          W1: First layer weights; has shape (D, H)
          b1: First layer biases; has shape (H,)
          W2: Second layer weights; has shape (H, C)
          b2: Second layer biases; has shape (C,)
    - X: Input data of shape (N, D). Each X[i] is a training sample.

    Returns a tuple of:
    - scores: Tensor of shape (N, C) giving the classification scores for X
    - hidden: Tensor of shape (N, H) giving the hidden layer representation
      for each input value (after the ReLU).
    """
    # Unpack variables from the params dictionary
    W1, b1 = params["W1"], params["b1"]
    W2, b2 = params["W2"], params["b2"]
    N, D = X.shape

    # Compute the forward pass
    hidden = None
    scores = None
    ############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input.#
    # Store the result in the scores variable, which should be an tensor of    #
    # shape (N, C).                                                            #
    ############################################################################
    # print(X.shape)
    # print(W1.shape)
    hidden = X.mm(W1) + b1        # shape: (N, H)
    hidden[hidden <0 ] = 0 # ReLU 
    scores = hidden.mm(W2) + b2  # shape: (N, C)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return scores, hidden


def nn_forward_backward(
    params: Dict[str, torch.Tensor],
    X: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    reg: float = 0.0
):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network. When you implement loss and gradient, please don't forget to
    scale the losses/gradients by the batch size. => divided by num_train(batch size)

    Inputs: First two parameters (params, X) are same as nn_forward_pass
    - params: a dictionary of PyTorch Tensor that store the weights of a model.
      It should have following keys with shape
          W1: First layer weights; has shape (D, H)
          b1: First layer biases; has shape (H,)
          W2: Second layer weights; has shape (H, C)
          b2: Second layer biases; has shape (C,)
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a tensor scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = params["W1"], params["b1"]
    W2, b2 = params["W2"], params["b2"]
    N, D = X.shape

    scores, h1 = nn_forward_pass(params, X)
    # If the targets are not given then jump out, we're done
    if y is None:
        return scores

    # Compute the loss
    loss = None
    ############################################################################
    # TODO: Compute the loss, based on the results from nn_forward_pass.       #
    # This should include both the data loss and L2 regularization for W1 and  #
    # W2. Store the result in the variable loss, which should be a scalar. Use #
    # the Softmax classifier loss. When you implment the regularization over W,#
    # please DO NOT multiply the regularization term by 1/2 (no coefficient).  #
    # If you are not careful here, it is easy to run into numeric instability  #
    # (Check Numeric Stability in http://cs231n.github.io/linear-classify/).   #
    ############################################################################
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


    # Backward pass: compute gradients
    grads = {}
    ###########################################################################
    # TODO: Compute the backward pass, computing the derivatives of the       #
    # weights and biases. Store the results in the grads dictionary.          #
    # For example, grads['W1'] should store the gradient on W1, and be a      #
    # tensor of same size                                                     #
    ###########################################################################

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

    # # ds/dW2
    # dsdw2 = hidden # (N,H)
    
    # # ds/db2
    # # dsdb2 = 1
    
    # # ds/dh
    # dsdh = W2 # (H,C)
    
    # # dh/dh0 # (N,H)
    # dhdh0 = torch.ones_like(hidden)
    # dhdh0[hidden<0] = 0

    # # dh0/dW1
    # dh0dw1 = X     # (N,D) (5,4) => If W1.t().mm(X) + b1 <0 => set it to 0
    # # dh0/db1
    # # dhdb1 = torch.ones((1,N))
    # # h = ReLU(W1.t()@X + b1)
    # # z = W1.t()@X + b1
    # # Should make z<0 entries' gradient to be 0
    # grads['W1'] =  dh0dw1.t() @ dhdh0 @ dsdh @ grad_matrix.t()                # (D, H)
    # print(grads['W1'].shape)
    # grads['b1'] =  torch.sum( (grad_matrix @ dsdh), dim=0)  # (H,)
    # grads['W2'] =  dsdw2.t() @ grad_matrix                        # (H, C)
    # grads['b2'] =  torch.sum( grad_matrix, dim=0)# (C)

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    grads['W1'] /= num_train
    grads['b1'] /= num_train
    grads['W2'] /= num_train
    grads['b2'] /= num_train


    # Add regularization to the loss.
    loss += reg * torch.sum(W1 * W1) + reg * torch.sum(W2*W2)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function w.r.t. the regularization term  #
    # and add it to dW. (part 2)                                                #
    #############################################################################
    # Replace "pass" statement with your code
    # dloss/dw = 2 * r * W
    # dW += 2 * reg * W 
    grads['W1'] += 2 * reg * W1
    grads['W2'] += 2 * reg * W2
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return loss, grads


def nn_train(
    params: Dict[str, torch.Tensor],
    loss_func: Callable,
    pred_func: Callable,
    X: torch.Tensor,
    y: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    learning_rate: float = 1e-3,
    learning_rate_decay: float = 0.95,
    reg: float = 5e-6,
    num_iters: int = 100,
    batch_size: int = 200,
    verbose: bool = False,
):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - params: a dictionary of PyTorch Tensor that store the weights of a model.
      It should have following keys with shape
          W1: First layer weights; has shape (D, H)
          b1: First layer biases; has shape (H,)
          W2: Second layer weights; has shape (H, C)
          b2: Second layer biases; has shape (C,)
    - loss_func: a loss function that computes the loss and the gradients.
      It takes as input:
      - params: Same as input to nn_train
      - X_batch: A minibatch of inputs of shape (B, D)
      - y_batch: Ground-truth labels for X_batch
      - reg: Same as input to nn_train
      And it returns a tuple of:
        - loss: Scalar giving the loss on the minibatch
        - grads: Dictionary mapping parameter names to gradients of the loss with
          respect to the corresponding parameter.
    - pred_func: prediction function that im
    - X: A PyTorch tensor of shape (N, D) giving training data.
    - y: A PyTorch tensor of shape (N,) giving training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - X_val: A PyTorch tensor of shape (N_val, D) giving validation data.
    - y_val: A PyTorch tensor of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.

    Returns: A dictionary giving statistics about the training process
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train // batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in range(num_iters):
        X_batch, y_batch = sample_batch(X, y, num_train, batch_size)

        # Compute loss and gradients using the current minibatch
        loss, grads = loss_func(params, X_batch, y=y_batch, reg=reg)
        loss_history.append(loss.item())

        #########################################################################
        # TODO: Use the gradients in the grads dictionary to update the         #
        # parameters of the network (stored in the dictionary self.params)      #
        # using stochastic gradient descent. You'll need to use the gradients   #
        # stored in the grads dictionary defined above.                         #
        #########################################################################
        params['W1'] -= learning_rate * grads['W1']
        params['b1'] -= learning_rate * grads['b1']
        params['W2'] -= learning_rate * grads['W2']
        params['b2'] -= learning_rate * grads['b2']

        # save params
        #########################################################################
        #                             END OF YOUR CODE                          #
        #########################################################################

        if verbose and it % 100 == 0:
            print("iteration %d / %d: loss %f" % (it, num_iters, loss.item()))

        # Every epoch, check train and val accuracy and decay learning rate.
        if it % iterations_per_epoch == 0:
            # Check accuracy
            y_train_pred = pred_func(params, loss_func, X_batch)
            train_acc = (y_train_pred == y_batch).float().mean().item()
            y_val_pred = pred_func(params, loss_func, X_val)
            val_acc = (y_val_pred == y_val).float().mean().item()
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)

            # Decay learning rate
            learning_rate *= learning_rate_decay

    return {
        "loss_history": loss_history,
        "train_acc_history": train_acc_history,
        "val_acc_history": val_acc_history,
    }


def nn_predict(
    params: Dict[str, torch.Tensor], loss_func: Callable, X: torch.Tensor
):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - params: a dictionary of PyTorch Tensor that store the weights of a model.
      It should have following keys with shape
          W1: First layer weights; has shape (D, H)
          b1: First layer biases; has shape (H,)
          W2: Second layer weights; has shape (H, C)
          b2: Second layer biases; has shape (C,)
    - loss_func: a loss function that computes the loss and the gradients
    - X: A PyTorch tensor of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A PyTorch tensor of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    scores, hidden = nn_forward_pass(params, X)
    y_pred = torch.argmax(scores, dim=1) # (N, 1)
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred


def nn_get_search_params():
    """
    Return candidate hyperparameters for a TwoLayerNet model.
    You should provide at least two param for each, and total grid search
    combinations should be less than 256. If not, it will take
    too much time to train on such hyperparameter combinations.

    Returns:
    - learning_rates: learning rate candidates, e.g. [1e-3, 1e-2, ...]
    - hidden_sizes: hidden value sizes, e.g. [8, 16, ...]
    - regularization_strengths: regularization strengths candidates
                                e.g. [1e0, 1e1, ...]
    - learning_rate_decays: learning rate decay candidates
                                e.g. [1.0, 0.95, ...]
    """
    learning_rates = []
    hidden_sizes = []
    regularization_strengths = []
    learning_rate_decays = []
    ###########################################################################
    # TODO: Add your own hyper parameter lists. This should be similar to the #
    # hyperparameters that you used for the SVM, but you may need to select   #
    # different hyperparameters to achieve good performance with the softmax  #
    # classifier.                                                             #
    ###########################################################################
    learning_rates = [1e0, 1e-1]
    hidden_sizes = [32,64,128,256]
    regularization_strengths = [0.001, 0.0001, 0.00001]
    learning_rate_decays = [1.0, 0.95]
    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################

    return (
        learning_rates,
        hidden_sizes,
        regularization_strengths,
        learning_rate_decays,
    )


def find_best_net(
    data_dict: Dict[str, torch.Tensor], get_param_set_fn: Callable
):
    """
    Tune hyperparameters using the validation set.
    Store your best trained TwoLayerNet model in best_net, with the return value
    of ".train()" operation in best_stat and the validation accuracy of the
    trained best model in best_val_acc. Your hyperparameters should be received
    from in nn_get_search_params

    Inputs:
    - data_dict (dict): a dictionary that includes
                        ['X_train', 'y_train', 'X_val', 'y_val']
                        as the keys for training a classifier
    - get_param_set_fn (function): A function that provides the hyperparameters
                                   (e.g., nn_get_search_params)
                                   that gives (learning_rates, hidden_sizes,
                                   regularization_strengths, learning_rate_decays)
                                   You should get hyperparameters from
                                   get_param_set_fn.

    Returns:
    - best_net (instance): a trained TwoLayerNet instances with
                           (['X_train', 'y_train'], batch_size, learning_rate,
                           learning_rate_decay, reg)
                           for num_iter times.
    - best_stat (dict): return value of "best_net.train()" operation
    - best_val_acc (float): validation accuracy of the best_net
    """

    best_net = None
    best_stat = None
    best_val_acc = 0.0

    #############################################################################
    # TODO: Tune hyperparameters using the validation set. Store your best      #
    # trained model in best_net.                                                #
    #                                                                           #
    # To help debug your network, it may help to use visualizations similar to  #
    # the ones we used above; these visualizations will have significant        #
    # qualitative differences from the ones we saw above for the poorly tuned   #
    # network.                                                                  #
    #                                                                           #
    # Tweaking hyperparameters by hand can be fun, but you might find it useful #
    # to write code to sweep through possible combinations of hyperparameters   #
    # automatically like we did on the previous exercises.                      #
    #############################################################################

    # YOUR_TURN: find the best learning_rates and regularization_strengths combination
    #            in 'svm_get_search_params'
    params = get_param_set_fn()
    learning_rates = params[0]
    hidden_sizes = params[1]
    regularization_strengths = params[2]
    learning_rate_decays = params[3]
    num_models = len(learning_rates) * len(regularization_strengths) * len(hidden_sizes) * len(learning_rate_decays)

    ####
    # It is okay to comment out the following conditions when you are working on svm_get_search_params.
    # But, please do not forget to reset back to the original setting once you are done.
    if num_models > 25:
      raise Exception("Please do not test/submit more than 25 items at once")
    elif num_models < 5:
      raise Exception("Please present at least 5 parameter sets in your final ipynb")
    ####


    i = 0
    # results is dictionary mapping tuples of the form
    # (learning_rate, regularization_strength) to tuples of the form
    # (train_acc, val_acc). 
    results = {}
    num_iters = 3000 # number of iterations

    for lr in learning_rates:
      for reg in regularization_strengths:
        for hs in hidden_sizes:
          for lr_decay in learning_rate_decays:
            i += 1
            print('Training NN %d / %d with learning_rate=%e and reg=%e, hidden_size=%e and lr_decay=%e'
                  % (i, num_models, lr, reg, hs, lr_decay))
            
            # eecs598.reset_seed(0)
            input_size = data_dict['X_train'].shape[1]
            num_classes = data_dict['y_train'].shape[0]
            net = TwoLayerNet(input_size, hs, num_classes, dtype=data_dict['X_train'].dtype, device=data_dict['X_train'].device)

            # Train the network
            stats = net.train(data_dict['X_train'], data_dict['y_train'],
                              data_dict['X_val'], data_dict['y_val'],
                              num_iters=num_iters, batch_size=1000,
                              learning_rate=lr, learning_rate_decay=lr_decay,
                              reg=reg, verbose=True)

            # Predict on the validation set
            y_val_pred = net.predict(data_dict['X_val'])
            val_acc = 100.0 * (y_val_pred == data_dict['y_val']).double().mean().item()
            print('Validation accuracy: %.2f%%' % val_acc)
            if val_acc > best_val_acc:
              best_val_acc = val_acc
              best_stat = stats
              best_net = net # save the svm
            results[(lr, reg, hs, lr_decay)] = (val_acc)

    #############################################################################
    #                               END OF YOUR CODE                            #
    #############################################################################

    return best_net, best_stat, best_val_acc
