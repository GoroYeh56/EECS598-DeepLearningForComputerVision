"""
Implements a network visualization in PyTorch.
Make sure to write device-agnostic code. For any function, initialize new tensors
on the same device as input tensors
"""

import torch


def hello():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Hello from network_visualization.py!")


def compute_saliency_maps(X, y, model):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images; Tensor of shape (N, 3, H, W)
    - y: Labels for X; LongTensor of shape (N,)
    - model: A pretrained CNN that will be used to compute the saliency map.

    Returns:
    - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
    images.
    """
    # Make input tensor require gradient
    X.requires_grad_()

    saliency = None
    ##############################################################################
    # TODO: Implement this function. Perform a forward and backward pass through #
    # the model to compute the gradient of the correct class score with respect  #
    # to each input image. You first want to compute the loss over the correct   #
    # scores (we'll combine losses across a batch by summing), and then compute  #
    # the gradients with a backward pass.                                        #
    # Hint: X.grad.data stores the gradients                                     #
    ##############################################################################
    yhat = model(X) # forward => scores (N, C)
    print("X.shape ", X.shape)       # 5, 3, 224, 224
    print("yhat.shape ", yhat.shape) # 5, 1000 => 1000 classes
    # print("yhat[:,y].shape", yhat[:,y].shape)
    loss = torch.sum(yhat[torch.arange(yhat.shape[0]),y])
    loss.backward()
    saliency =  X.grad.abs().max(dim = 1).values
    print("saliency.shape ",saliency.shape)
    # print(type(saliency))
    ##############################################################################
    #               END OF YOUR CODE                                             #
    ##############################################################################
    return saliency


def make_adversarial_attack(X, target_y, model, max_iter=100, verbose=True):
    """
    Generate an adversarial attack that is close to X, but that the model classifies
    as target_y.

    Inputs:
    - X: Input image; Tensor of shape (1, 3, 224, 224)
    - target_y: An integer in the range [0, 1000)
    - model: A pretrained CNN
    - max_iter: Upper bound on number of iteration to perform
    - verbose: If True, it prints the pogress (you can use this flag for debugging)

    Returns:
    - X_adv: An image that is close to X, but that is classifed as target_y
    by the model.
    """
    # Initialize our adversarial attack to the input image, and make it require
    # gradient
    X_adv = X.clone()
    X_adv = X_adv.requires_grad_()

    learning_rate = 1
    ##############################################################################
    # TODO: Generate an adversarial attack X_adv that the model will classify    #
    # as the class target_y. You should perform gradient ascent on the score     #
    # of the target class, stopping when the model is fooled.                    #
    # When computing an update step, first normalize the gradient:               #
    #   dX = learning_rate * g / ||g||_2                                         #
    #                                                                            #
    # You should write a training loop.                                          #
    #                                                                            #
    # HINT: For most examples, you should be able to generate an adversarial     #
    # attack in fewer than 100 iterations of gradient ascent.                    #
    # You can print your progress over iterations to check your algorithm.       #
    ##############################################################################

    model.eval() # To eval mode
    y = torch.tensor([target_y], dtype=torch.long, device=X.device)


    for i in range(max_iter):

        yhat = model(X_adv ) # forward => scores (N, C)
        # print("yhat.shape ", yhat.shape) #[1, 1000])
        predict_class = torch.argmax(yhat, dim=1)
        # print("predicted: ", predict_class)
        if predict_class == target_y:
            print("The model is fooled at iteration ",i)
            break
        
        # Gradient ascent on the "score" of the target class

        loss = yhat[0][target_y]
        # print("iter ", i, " loss ", loss)
        loss.backward()  
        dX = learning_rate * X_adv.grad / torch.sqrt(torch.sum(X_adv.grad**2))     
        with torch.no_grad():
            X_adv.data += dX
        X_adv.requires_grad_()
        
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return X_adv


def class_visualization_step(img, target_y, model, **kwargs):
    """
    Performs gradient step update to generate an image that maximizes the
    score of target_y under a pretrained model.

    Inputs:
    - img: random image with jittering as a PyTorch tensor
    - target_y: Integer in the range [0, 1000) giving the index of the class
    - model: A pretrained CNN that will be used to generate the image

    Keyword arguments:
    - l2_reg: Strength of L2 regularization on the image
    - learning_rate: How big of a step to take
    """

    l2_reg = kwargs.pop("l2_reg", 1e-3)
    learning_rate = kwargs.pop("learning_rate", 25)
    ########################################################################
    # TODO: Use the model to compute the gradient of the score for the     #
    # class target_y with respect to the pixels of the image, and make a   #
    # gradient step on the image using the learning rate. Don't forget the #
    # L2 regularization term!                                              #
    # Be very careful about the signs of elements in your code.            #
    # Hint: You have to perform inplace operations on img.data to update   #
    # the generated image using gradient ascent & reset img.grad to zero   #
    # after each step.                                                     #
    ########################################################################
    # print(img.shape) # 1,3,224,224
    model.eval() # To eval mode
    img.requires_grad_()
    y = torch.tensor([target_y], dtype=torch.long, device=img.device)

    yhat = model(img ) # forward => scores (N, C)    
    loss = yhat[:,target_y] - l2_reg * torch.sum(img**2)
    loss.backward()  

    dX = learning_rate * img.grad #/torch.sqrt(torch.sum(img.grad**2))     
    with torch.no_grad():
        img += dX
    # reset img.grad to zero
    img.grad = torch.zeros(img.shape, dtype=img.dtype, device=img.device)

    # loss = -torch.nn.functional.cross_entropy(yhat, y, reduction = 'sum')
    # loss += l2_reg * torch.sqrt(torch.sum(img**2))
    ########################################################################
    #                             END OF YOUR CODE                         #
    ########################################################################
    return img
