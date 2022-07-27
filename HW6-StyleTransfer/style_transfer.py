"""
Implements a style transfer in PyTorch.
Make sure to write device-agnostic code. For any function, initialize new tensors
on the same device as input tensors
"""

import torch
import torch.nn as nn


def hello():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Hello from style_transfer.py!")


def content_loss(content_weight, content_current, content_original):
    """
    Compute the content loss for style transfer.

    Inputs:
    - content_weight: Scalar giving the weighting for the content loss.
    - content_current: features of the current image; this is a PyTorch Tensor
      of shape (1, C_l, H_l, W_l).
    - content_original: features of the content image, Tensor with shape
      (1, C_l, H_l, W_l).

    Returns:
    - scalar content loss
    """
    ############################################################################
    # TODO: Compute the content loss for style transfer.                       #
    ############################################################################
    # _, Cl, Hl, Wl = content_current.shape
    # print(content_current.shape)

    loss = content_weight * torch.sum((content_current - content_original)**2)
    return loss
    # for c in range(Cl):
    #     for i in range(Hl):
    #         for j in range(Wl):
    #             loss += content_weight * (content_current[:,c,i,j] - content_original[:,c,i,j]) **2
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################


def gram_matrix(features, normalize=True):
    """
    Compute the Gram matrix from features.

    Inputs:
    - features: PyTorch Tensor of shape (N, C, H, W) giving features for
      a batch of N images.
    - normalize: optional, whether to normalize the Gram matrix
        If True, divide the Gram matrix by the number of neurons (H * W * C)

    Returns:
    - gram: PyTorch Tensor of shape (N, C, C) giving the
      (optionally normalized) Gram matrices for the N input images.
    """
    gram = None
    ############################################################################
    # TODO: Compute the Gram matrix from features.                             #
    # Don't forget to implement for both normalized and non-normalized version #
    ############################################################################
    N, C, H, W = features.shape
    
    # for n in range(N):
        # gram[n,:,:] = features[n,:,:] @ features[n,:,:].T
    # gram = torch.zeros((N,C,C), dtype=features.dtype, device=features.device)

    features = features.reshape(N, C, -1) # (N, C, H*W)
    gram = torch.bmm(features,features.permute(0,2,1))

    if normalize:
        gram /= (H*W*C)
    
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################
    return gram


def style_loss(feats, style_layers, style_targets, style_weights):
    """
    Computes the style loss at a set of layers.

    Inputs:
    - feats: list of the features at every layer of the current image, as produced
      by the extract_features function.
    - style_layers: List of layer indices into feats giving the layers to include
      in the style loss.
    - style_targets: List of the same length as style_layers, where
      style_targets[i] is a PyTorch Tensor giving the Gram matrix of the source
      style image computed at layer style_layers[i].
    - style_weights: List of the same length as style_layers, where
      style_weights[i] is a scalar giving the weight for the style loss at layer
      style_layers[i].

    Returns:
    - style_loss: A PyTorch Tensor holding a scalar giving the style loss.
    """
    ############################################################################
    # TODO: Computes the style loss at a set of layers.                        #
    # Hint: you can do this with one for loop over the style layers, and       #
    # should not be very much code (~5 lines).                                 #
    # You will need to use your gram_matrix function.                          #
    ############################################################################

    # feats: 13 layers?
    # feats[0].shape (1, 64, 95, 127)
    # style_layers: [1, 4, 6, 7]
    # style_targets: length of 4 item
    # style_weights: len 4 items


    # KEY: Use len(style_layers) ! instead of len(feat) !
    loss = 0
    for i in range(len(style_layers)):
        loss += style_weights[i] * torch.sum((gram_matrix(feats[style_layers[i]]) -style_targets[i]  )**2 )
    return loss
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################


def tv_loss(img, tv_weight):
    """
    Compute total variation loss.

    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.

    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    ############################################################################
    # TODO: Compute total variation loss.                                      #
    # Your implementation should be vectorized and not require any loops!      #
    ############################################################################
    # print(img.shape) # 1, 3, 192, 256 N, C, H, W
    loss = 0

    # For each image:
        # C, H, W => (xi+1-xi)**2 + (xi, j+1 - xi,j)**2
        # torch.sum( (), dim=2, dim=1, dim=0)
    N, C, H, W = img.shape

    ############# TODO: Vectorization(No FOR-LOOP) ##############
    # for n in range(N):

    # Create a (N, C, H-1, W)
    i_up   = img[:,:,1:,:]
    i_down = img[:,:,:-1,:]
    
    # Create a (N, C, H, W-1)
    j_up   = img[:,:,:, 1:]
    j_down = img[:,:,:,:-1]
    loss = tv_weight * (torch.sum((i_up - i_down)**2) + torch.sum((j_up - j_down)**2))
 

    # for i in range(H-1):

    #     loss += torch.sum((img[:,:,i+1,:] - img[:,:,i,:])**2 )
    # for j in range(W-1):        
    #     loss += torch.sum((img[:,:,:,j+1] - img[:,:,:,j])**2)

    # loss = tv_weight * loss
    return loss    

    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################


def guided_gram_matrix(features, masks, normalize=True):
    """
    Inputs:
      - features: PyTorch Tensor of shape (N, R, C, H, W) giving features for
        a batch of N images.
      - masks: PyTorch Tensor of shape (N, R, H, W)
      - normalize: optional, whether to normalize the Gram matrix
          If True, divide the Gram matrix by the number of neurons (H * W * C)

      Returns:
      - gram: PyTorch Tensor of shape (N, R, C, C) giving the
        (optionally normalized) guided Gram matrices for the N input images.
    """
    guided_gram = None
    ##############################################################################
    # TODO: Compute the guided Gram matrix from features.                        #
    # Apply the regional guidance mask to its corresponding feature and          #
    # calculate the Gram Matrix. You are allowed to use one for-loop in          #
    # this problem.                                                              #
    ##############################################################################
    
    N, R, C, H, W = features.shape
    # print("features.shape ", features.shape)
    # print("masks.shape ", masks.shape)
    features = features.reshape(N, R, C, -1) # H*W    
    guided_gram = torch.zeros((N,R,C,C), dtype=features.dtype, device=features.device)


    for r in range(R):
        # mask is the same for each channel c
        #  (N, H,W) => (N,1,H*W) =>  (N, C, H*W
        m = masks[:,r,:,:].reshape(N,1,-1).repeat(1,C,1)
                                           # (N, C, H*W)              (N, H*W,C)
        guided_gram[:,r,:,:] = torch.bmm(m*features[:,r,:,:],(m*features[:,r,:,:]).permute(0,2,1))

        # print("m.shape ", m.shape)
        # print("features[:,r,:,:].shape ", features[:,r,:,:].shape)


    if normalize:
        guided_gram /= (H*W*C)

    return guided_gram
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################


def guided_style_loss(
    feats, style_layers, style_targets, style_weights, content_masks
):
    """
    Computes the style loss at a set of layers.

    Inputs:
    - feats: list of the features at every layer of the current image, as produced
      by the extract_features function.
    - style_layers: List of layer indices into feats giving the layers to include
      in the style loss.
    - style_targets: List of the same length as style_layers, where
      style_targets[i] is a PyTorch Tensor giving the guided Gram matrix of the
      source style image computed at layer style_layers[i].
    - style_weights: List of the same length as style_layers, where
      style_weights[i] is a scalar giving the weight for the style loss at layer
      style_layers[i].
    - content_masks: List of the same length as style_layers, where
      content_masks[i] is a PyTorch Tensor giving the binary masks of the content
      image.

    Returns:
    - style_loss: A PyTorch Tensor holding a scalar giving the style loss.
    """
    ############################################################################
    # TODO: Computes the guided style loss at a set of layers.                 #
    ############################################################################
    # KEY: Use len(style_layers) ! instead of len(feat) !
    loss = 0
    for i in range(len(style_layers)):
        # print("\n\ni: =======",i )
        guided_gram = guided_gram_matrix(feats[style_layers[i]], content_masks[style_layers[i]])
        loss += style_weights[i] * torch.sum( (guided_gram-style_targets[i])**2 )
        # print("guided_.shape ", guided_gram.shape)
        # print("style_targets[i].shape ", style_targets[i].shape)
    return loss
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################
