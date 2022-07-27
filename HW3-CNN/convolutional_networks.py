"""
Implements convolutional networks in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
from numpy.lib import stride_tricks
import torch
from torch._C import device
from a3_helper import softmax_loss
from fully_connected_networks import Linear_ReLU, Linear, Solver, adam, ReLU


def hello_convolutional_networks():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print('Hello from convolutional_networks.py!')


class Conv(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        """
        A naive implementation of the forward pass for a convolutional layer.
        The input consists of N data points, each with C channels, height H and
        width W. We convolve each input with F different filters, where each
        filter spans all C channels and has height HH and width WW.

        Input:
        - x: Input data of shape (N, C, H, W)
        - w: Filter weights of shape (F, C, HH, WW)
        - b: Biases, of shape (F,)
        - conv_param: A dictionary with the following keys:
          - 'stride': The number of pixels between adjacent receptive fields
            in the horizontal and vertical directions.
          - 'pad': The number of pixels that is used to zero-pad the input.

        During padding, 'pad' zeros should be placed symmetrically (i.e equally
        on both sides) along the height and width axes of the input. Be careful
        not to modfiy the original input x directly.

        Returns a tuple of:
        - out: Output data of shape (N, F, H', W') where H' and W' are given by
          H' = 1 + (H + 2 * pad - HH) / stride
          W' = 1 + (W + 2 * pad - WW) / stride
        - cache: (x, w, b, conv_param)
        """
        out = None
        ####################################################################
        # TODO: Implement the convolutional forward pass.                  #
        # Hint: you can use function torch.nn.functional.pad for padding.  #
        # You are NOT allowed to use anything in torch.nn in other places. #
        ####################################################################

        # Replace "pass" statement with your code
        # stride = conv_param['stride']
        # pad = conv_param['pad']
        # N = x.shape[0]
        # C = x.shape[1]
        # H = x.shape[2]
        # W = x.shape[3]
        # F = w.shape[0]
        # HH = w.shape[2]
        # WW = w.shape[3]

        # H1 = int(1 + (H + 2 * pad - HH) / stride)
        # W1 = int(1 + (W + 2 * pad - WW) / stride)
        
        # #F, C, HH, WW
        # if (H + 2 * pad - HH) % stride != 0:
        #   raise ValueError('conv_forward: (H + 2 * pad - HH) % stride != 0')
        # if (W + 2 * pad - WW) % stride != 0:
        #   raise ValueError('conv_forward: (W + 2 * pad - WW) % stride != 0')

        # out = torch.zeros(N, F, H1, W1, dtype = x.dtype, device = x.device)
        # xp = torch.nn.functional.pad(x, (pad, pad, pad, pad), "constant", 0)
        
        
        # for f in range(F):
        #   # print("filter ",f)
        #   for n in range(N):
        #     xn = xp[n]
        #     for wi in range(W1):
        #       for hi in range(H1):
        #         x_start = wi * stride 
        #         y_start = hi * stride
        #         x2conv = xn[:, y_start:y_start + HH, x_start:x_start + WW]
        #         out[n, f, hi, wi] = torch.sum(x2conv * w[f]) + b[f]


        # stride = conv_param['stride']
        # pad = conv_param['pad']
        # # print("stride: ",stride, " pad: ", pad)
        # N, C, H, W = x.shape
        # F, _, HH, WW = w.shape
        # Hout = int(1 + (H + 2 * pad - HH) / stride)
        # Wout = int(1 + (W + 2 * pad - WW) / stride)
        # print("Hout, Wout", Hout, Wout)

        # #F, C, HH, WW
        # if (H + 2 * pad - HH) % stride != 0:
        #   raise ValueError('conv_forward: (H + 2 * pad - HH) % stride != 0')
        # if (W + 2 * pad - WW) % stride != 0:
        #   raise ValueError('conv_forward: (W + 2 * pad - WW) % stride != 0')

        # out = torch.zeros(N, F, Hout, Wout, dtype = x.dtype, device = x.device)
        # xpad = torch.nn.functional.pad(x, (pad, pad, pad, pad), "constant", 0)
        # for f in range(F):
        #   for n in range(N): # Each image sample
        #     xi = xpad[n]     # Take sample i
        #     for wi in range(Wout):
        #       for hi in range(Hout):
        #         # print("wi, hi ",wi, hi)
        #         # j_start = wi * stride 
        #         # i_start = hi * stride
        #         # x2conv = xi[:, i_start: i_start + HH, j_start: j_start + WW]
        #         # out[n, f, hi, wi] = torch.sum(x2conv * w[f]) + b[f]
        #     # print(" End sample ", n)
        #   # print("End filter ",f)
     
        N, C, H, W = x.shape
        F, C, HH, WW = w.shape # F (C*HH*WW) filters
        stride = conv_param['stride']
        pad = conv_param['pad']
        pad_tuple = (pad, pad, pad, pad)
        # 1. Pad input tensor x
        # print(x.shape)
        input_x = torch.nn.functional.pad(x, pad_tuple)
        # print(input_x.shape)

        # 2. Perform Convolution
        # Number of chuncks: 
        # New Height: H' = 1 + (H + 2 * pad - HH) / stride
        # New Width:　W' = 1 + (W + 2 * pad - WW) / stride
        Hout =  int(1 + (H + 2 * pad - HH) / stride)
        Wout = int(1 + (W + 2 * pad - WW) / stride)
        # print("H', W' ", Hout, Wout)

        a =torch.empty((N, F, Hout, Wout), dtype=torch.float64, device=x.device)
        out =  torch.empty_like(a)

        # F filters => New number of channels(F')
        for index in range(N): # N samples
          # print("\n\nSample: ",index)
          for c in range(F):
            # print(" Kernel: ", c)
            for i in range(Hout):
              # print("   row: ",i)
              for j in range(Wout):
                # print("   column ",j)
                # from i*stride ~ i+HH
                # from j*stride ~ j+WW
                # slice input_x tensor:
                start_i = i*stride
                stop_i = i*stride + HH 
                start_j = j * stride
                stop_j = j * stride + WW 
                slice_input = input_x[index, :, start_i:stop_i , start_j:stop_j ]
                # Both are (3, 4, 4) => Output_pixel should be a single value
                out_pixel = (slice_input * w[c,:,:,:] )
                out_pixel = torch.sum(out_pixel, dim=[0,1,2])            
                out[index, c, i, j] = out_pixel + b[c]
                
        # # 3. Return the output
        # out.cpu()
        #####################################################################
        #                          END OF YOUR CODE                         #
        #####################################################################
        cache = (x, w, b, conv_param)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        A naive implementation of the backward pass for a convolutional layer.
          Inputs:
        - dout: Upstream derivatives.
        - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

        Returns a tuple of:
        - dx: Gradient with respect to x
        - dw: Gradient with respect to w
        - db: Gradient with respect to b
        """
        dx, dw, db = None, None, None
        ###############################################################
        # TODO: Implement the convolutional backward pass.            #
        ###############################################################
        # upstream derivative: dout

        print("dout shape ", dout.shape)
        (x, w, b, conv_param) = cache

        stride = conv_param['stride']
        pad = conv_param['pad']
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        _, _, Hout, Wout = dout.shape

        # Hout = int(1 + ((H+2*pad)-HH)/stride)
        # Wout = int(1 + ((W+2*pad)-WW)/stride)

        dw = torch.zeros(w.shape, device=w.device, dtype=w.dtype)
        x_pad = x.clone()
        x_pad = torch.nn.functional.pad(x_pad, (pad, pad, pad, pad), "constant", 0)
        dx_pad = torch.zeros(x_pad.shape, device=x.device, dtype=x.dtype)
        db = torch.zeros(b.shape, device=b.device, dtype=b.dtype)

        for f in range(F):
          for n in range(N):
            for hi in range(Hout):
              for wi in range(Wout):
                y_start = hi * stride                
                x_start = wi * stride 
                dx_pad[n, :, y_start:y_start + HH, x_start:x_start + WW] += w[f, :, :, :] * dout[n, f, hi, wi]
                dw[f, :, :, :] += x_pad[n, :, y_start:y_start + HH, x_start:x_start + WW] * dout[n, f, hi, wi]

        dx = dx_pad[:,:,pad:pad+H,pad:pad+W]         
        db = torch.sum(dout, [0, 2, 3]) 


        #################################

        # # Initialisations
        # dx = torch.zeros_like(x)
        # dw = torch.zeros_like(w)
        # db = torch.zeros_like(b)
        
        # # Dimensions
        # N, C, H, W = x.shape
        # F, _, HH, WW = w.shape
        # _, _, H_, W_ = dout.shape
        
        # # db - dout (N, F, H', W')
        # # On somme sur tous les éléments sauf les indices des filtres
        # db =torch.sum(dout, axis=(0, 2, 3))
        
        # # dw = xp * dy
        # # 0-padding juste sur les deux dernières dimensions de x
        # xp = torch.nn.functional.pad(x, (pad,pad,pad,pad), 'constant')
        # # 
        # for n in range(N):       # Number of images
        #     for f in range(F):   # Number of filters
        #         for i in range(HH): # Filter height
        #             for j in range(WW):# Filter width
        #                 for k in range(H_): # For each row in dout (k)
        #                     for l in range(W_): # For each column in dout (l)
        #                         for c in range(C): # FOr ecah channel
        #                             #              # padded_x [n, c, ]
        #                             dw[f,c,i,j] += xp[n, c, stride*i+k, stride*j+l] * dout[n, f, k, l]

        # # dx = dy_0 * w'
        # # Valide seulement pour un stride = 1
        # # 0-padding juste sur les deux dernières dimensions de dy = dout (N, F, H', W')
        # # doutp = torch.nn.functional.pad(dout, (WW-stride, HH-stride, WW-stride, HH-stride), 'constant')
        # # Wdiff = int(torch.ceil(torch.Tensor([ (W + WW - W_ )/2]))) 
        # # Hdiff = int(torch.ceil(torch.Tensor([ (H + HH - H_ )/2]))) 
        # # doutp = torch.nn.functional.pad(dout, (Wdiff+1,Wdiff, Hdiff+1, Hdiff), 'constant') 
        # doutp = torch.nn.functional.pad(dout, (WW-1, WW-1, HH-1, HH-1), 'constant')
        # print("doutp.shape ", doutp.shape)

        # dxp = torch.nn.functional.pad(dx, (pad,pad,pad,pad), 'constant')
        # # 0-padding juste sur les deux dernières dimensions de dx
        # # dxp = torch.nn.functional.pad(dx, ((0,), (0,), (pad,), (pad, )), 'constant')

        # # filter inverse 180 degree dimension (F, C, HH, WW)
        # w_ = torch.zeros_like(w)
        # for i in range(HH):
        #     for j in range(WW):
        #         w_[:,:,i,j] = w[:,:,HH-i-1,WW-j-1]
        
        # # Version sans vectorisation
        # for n in range(N):       # On parcourt toutes les images
        #     for f in range(F):   # On parcourt tous les filtres
        #         for i in range(H+2*pad): # For each row in xp (0~32)
        #             for j in range(W+2*pad):# For each column in xp
        #                 for k in range(HH): # For each row in filter: (0,1,2)  (k)
        #                     for l in range(WW):# For each column in the filter (l)
        #                         for c in range(C): # For each channel
        #                             # print(doutp[n, f, i+k, j+l])
        #                             # print(w_[f, c, k, l])
        #                             dxp[n,c,i,j] += doutp[n, f, i+k, j+l] * w_[f, c, k, l]
        # #Remove padding for dx
        # dx = dxp[:,:,pad:-pad,pad:-pad]

        ########### Mine ##########
        # dL/dw : (F, C, HH, WW) = Conv ( dout, input_x) 
        # print("dout.shape ", dout.shape) (4, 2, 5, 5)
        # dw = dout @ input_x ( w: F, C, HH, WW)
        # Nd, Fd, Hd, Wd = dout.shape
        # dw = torch.zeros_like(w)

        # # sum input_x over channels(dim 1)
        # input_x = torch.sum(input_x, dim=1)
        # # sum dout over channels (dim 1)
        # dout = torch.sum(dout, dim=1)

        # for index in range(F): # F filters
        #   # print("\n\nSample: ",index)
        #   for c in range(C): # different channels
        #     for i in range(HH):
        #       # print("   row: ",i)
        #       for j in range(WW):
        #         # print("   column ",j)
        #         # slice input_x tensor:
        #         start_i = i*stride
        #         stop_i = i*stride + Hd 
        #         start_j = j * stride
        #         stop_j = j * stride + Wd 
        #         slice_input = input_x[index, c, start_i:stop_i , start_j:stop_j ]
        #         # print(slice_input.shape)          # (5,5)
        #         # print(dout[index, :, :,:].shape)  # (2, 5,5)
        #         out_pixel = (slice_input * dout[index, :, :,:] )
        #         print("out_pixel.shape", out_pixel.shape)
        #         out_pixel = torch.sum(out_pixel)            
        #         dw[index, c, i, j] = out_pixel
 

        
        # # dL/dx : (N, C, H, W) = Conv (w, dout) 
        # dx = dout @ w
        # # print("dout.shape ", dout.shape) (4, 2, 5, 5)
        # # dx.shape => ( x: N, C, H, W)
        # Nd, Fd, Hd, Wd = dout.shape
        # dx = torch.zeros_like(x)
        # for index in range(N): # N samples
        #   # print("\n\nSample: ",index)
        #   for c in range(C):
        #     # print(" Kernel: ", c)
        #     for i in range(H):
        #       # print("   row: ",i)
        #       for j in range(W):
        #         # print("   column ",j)
        #         # slice input_x tensor:
        #         start_i = i*stride
        #         stop_i = i*stride + Hd 
        #         start_j = j * stride
        #         stop_j = j * stride + Wd 
        #         slice_input = w[index, :, start_i:stop_i , start_j:stop_j ]
        #         # Both are (3, 4, 4) => Output_pixel should be a single value
        #         out_pixel = (slice_input * dout[c,:,:,:] )
        #         out_pixel = torch.sum(out_pixel)            
        #         dx[index, c, i, j] = out_pixel


        # # # dL/db : (F)
        # db = torch.sum(torch.sum(torch.sum(dout,dim=3),dim=2), dim=0)

        print(dx.shape)
        print(dw.shape)
        print(db.shape)
        ###############################################################
        #                       END OF YOUR CODE                      #
        ###############################################################
        return dx, dw, db


class MaxPool(object):

    @staticmethod
    def forward(x, pool_param):
        """
        A naive implementation of the forward pass for a max-pooling layer.

        Inputs:
        - x: Input data, of shape (N, C, H, W)
        - pool_param: dictionary with the following keys:
          - 'pool_height': The height of each pooling region
          - 'pool_width': The width of each pooling region
          - 'stride': The distance between adjacent pooling regions
        No padding is necessary here.

        Returns a tuple of:
        - out: Output of shape (N, C, H', W') where H' and W' are given by
          H' = 1 + (H - pool_height) / stride
          W' = 1 + (W - pool_width) / stride
        - cache: (x, pool_param)
        """
        out = None
        ####################################################################
        # TODO: Implement the max-pooling forward pass                     #
        ####################################################################
        
        N, C, H, W = x.shape

        stride = pool_param["stride"]
        pool_height = pool_param["pool_height"]
        pool_width = pool_param["pool_width"]
        Hp = int(1 +(H-pool_height)/stride)
        Wp = int(1+ (W-pool_width)/stride)
        out = torch.empty((N, C, Hp, Wp), dtype=x.dtype, device=x.device)

        if (H-pool_height)%stride !=0:
          raise ValueError('maxpool_forward: (H - pool_height) % stride != 0')
        if (W-pool_width)%stride !=0:
          raise ValueError('maxpool_forward: ( W- pool_width) % stride != 0')
        for index in range(N): # N samples
          # print("\n\nSample: ",index)
          for c in range(C):
            # print(" Kernel: ", c)
            for i in range(Hp):
              # print("   row: ",i)
              for j in range(Wp):
                # print("   column ",j)
                # from i*stride ~ i+HH
                # from j*stride ~ j+WW
                # slice input_x tensor:
                start_i = i*stride
                stop_i = i*stride + pool_height
                start_j = j * stride
                stop_j = j * stride + pool_width
                slice_input = x[index, c, start_i:stop_i , start_j:stop_j ] #(1, 1, 2,2)
                # print("slice.shape ", slice_input.shape)
                out_pixel = torch.max(slice_input)             
                out[index, c, i, j] = out_pixel

        # print(x[0,0,0:5,0:5])
        # print(out[0,0,:,:])
        ####################################################################
        #                         END OF YOUR CODE                         #
        ####################################################################
        cache = (x, pool_param)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        A naive implementation of the backward pass for a max-pooling layer.
        Inputs:
        - dout: Upstream derivatives
        - cache: A tuple of (x, pool_param) as in the forward pass.
        Returns:
        - dx: Gradient with respect to x
        """
        dx = None
        #####################################################################
        # TODO: Implement the max-pooling backward pass                     #
        #####################################################################
        
        (x, pool_param) = cache

        stride = pool_param["stride"]
        pool_height = pool_param["pool_height"]
        pool_width = pool_param["pool_width"]

        _, _, H, W = x.shape
        N, C, Hp, Wp = dout.shape
        # out = torch.empty((N, C, Hp, Wp), dtype=x.dtype, device=x.device)

        dx = torch.zeros_like(x, dtype=x.dtype, device=x.device)

  
        for index in range(N): # N samples
          # print("\n\nSample: ",index)
          for c in range(C):   # for each channel
            for i in range(Hp):
              for j in range(Wp):
                # from i*stride ~ i+pool_height
                # from j*stride ~ j+pool_width
                # slice input_x tensor:
                start_i = i*stride
                stop_i = i*stride + pool_height
                start_j = j * stride
                stop_j = j * stride + pool_width
                slice_input = x[index, c, start_i:stop_i , start_j:stop_j ] #(1, 1, 2,2)
                dx[index, c, start_i:stop_i , start_j:stop_j] += (slice_input==torch.max(slice_input)) * dout[index, c, i, j]            
                # print(dx[index, c, start_i:stop_i , start_j:stop_j])
                
                # print(dx_slice)
                # dx[index, c, start_i:stop_i , start_j:stop_j] = dx_slice
                # max_index = torch.argmax(slice_input)
                # dx[index, c, start_i:stop_i , start_j:stop_j] = 0
                # dx[index, c, start_i + int(max_index/pool_width),   start_j+max_index%pool_width] = dout[index, c, i, j]
                # print("index: ", torch.argmax(slice_input))

        # Replace "pass" statement with your code
        # x, pool_param = cache
        # N = x.shape[0]
        # C = x.shape[1]
        # H = x.shape[2]
        # W = x.shape[3]

        # pool_height = pool_param['pool_height']
        # pool_width = pool_param['pool_width']
        # stride = pool_param['stride']
        # dx = torch.zeros_like(x)
        
        # H1 = int(1 + (H - pool_height) / stride)
        # W1 = int(1 + (W - pool_width) / stride)

        # for n in range(N):
        #   for c in range(C):
        #     for wi in range(W1):
        #       for hi in range(H1):
        #         x_start = wi * stride 
        #         y_start = hi * stride
        #         i = torch.argmax(x[n, c, y_start:y_start + pool_height, x_start:x_start + pool_width])
        #         i_y = i // pool_width
        #         i_x = i % pool_width
        #         dx[n, c,  y_start + i_y, x_start + i_x] = dout[n, c, hi, wi]
        ####################################################################
        #                          END OF YOUR CODE                        #
        ####################################################################
        return dx


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:
    conv - relu - 2x2 max pool - linear - relu - linear - softmax
    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self,
                 input_dims=(3, 32, 32),
                 num_filters=32,
                 filter_size=7,
                 hidden_dim=100,
                 num_classes=10,
                 weight_scale=1e-3,
                 reg=0.0,
                 dtype=torch.float,
                 device='cpu'):
        """
        Initialize a new network.
        Inputs:
        - input_dims: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in convolutional layer
        - hidden_dim: Number of units to use in fully-connected hidden layer
        - num_classes: Number of scores to produce from the final linear layer.
        - weight_scale: Scalar giving standard deviation for random
          initialization of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: A torch data type object; all computations will be performed
          using this datatype. float is faster but less accurate, so you
          should use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ######################################################################
        # TODO: Initialize weights，biases for the three-layer convolutional #
        # network. Weights should be initialized from a Gaussian             #
        # centered at 0.0 with standard deviation equal to weight_scale;     #
        # biases should be initialized to zero. All weights and biases       #
        # should be stored in the dictionary self.params. Store weights and   #
        # biases for the convolutional layer using the keys 'W1' and 'b1';   #
        # use keys 'W2' and 'b2' for the weights and biases of the hidden    #
        # linear layer, and key 'W3' and 'b3' for the weights and biases of  #
        # the output linear layer                                            #
        #                                                                    #
        # IMPORTANT: For this assignment, you can assume that the padding    #
        # and stride of the first convolutional layer are chosen so that     #
        # **the width and height of the input are preserved**. Take a        #
        # look at the start of the loss() function to see how that happens.  #
        ######################################################################


        # Hout =  int(1 + (H + 2 * pad - HH) / stride)
        # Wout = int(1 + (W + 2 * pad - WW) / stride)
        # H' = 1 + (1 + (H + 2 * pad - HH) / stride - pool_height) / stride
        # W' = 1 + (1 + (W + 2 * pad - WW) / stride - pool_width) / stride
        C, H, W = input_dims
        F = num_filters
        HH = filter_size
        WW = filter_size

        H2, W2 = int(1 + (H - 2)/2), int(1 + (W - 2)/2)
        W2in = (F * H2 * W2) # Note: should flatten out1 to be this shape
        W2out = hidden_dim

        self.params["W1"] = torch.normal(0, weight_scale, size=(F,C,HH,WW)).to(dtype).to(device)
        self.params["b1"] = torch.zeros(F).to(dtype).to(device)
        self.params["W2"] = torch.normal(0, weight_scale, size=(W2in, W2out)).to(dtype).to(device)
        self.params["b2"] = torch.zeros(W2out).to(dtype).to(device)
        self.params["W3"] = torch.normal(0, weight_scale, size=(W2out, num_classes)).to(dtype).to(device)
        self.params["b3"] = torch.zeros(num_classes).to(dtype).to(device)


        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'dtype': self.dtype,
          'params': self.params,
        }
        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = checkpoint['dtype']
        self.reg = checkpoint['reg']
        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.
        Input / output: Same API as TwoLayerNet.
        """
        X = X.to(self.dtype)
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ######################################################################
        # TODO: Implement the forward pass for three-layer convolutional     #
        # net, computing the class scores for X and storing them in the      #
        # scores variable.                                                   #
        #                                                                    #
        # Remember you can use functions defined in your implementation      #
        # above                                                              #
        ######################################################################
        # conv_param = {'stride': 1, 'pad': 1}
        # pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        # out1, cache1 = Conv.forward(X, W1, b1, conv_param)

        # conv - relu - 2x2 max pool - linear - relu - linear - softmax
        out1, cache1 = Conv_ReLU_Pool.forward(X, W1, b1, conv_param, pool_param)
        # print("out1 success")
        # print("out1.shape ", out1.shape)
        # print("W2.shape ", W2.shape)
        out2, cache2 = Linear_ReLU.forward(out1, W2, b2)
        # print("out2.shape ", out2.shape)
        # print("W3.shape ", W3.shape)
        scores, cache3 = Linear.forward(out2, W3, b3)
        # print("scores.shape ", scores.shape)
        # print("scores success")
 
        ######################################################################
        #                             END OF YOUR CODE                       #
        ######################################################################

        if y is None:
            return scores

        loss, grads = 0.0, {}
        ####################################################################
        # TODO: Implement backward pass for three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables.  #
        # Compute data loss using softmax, and make sure that grads[k]     #
        # holds the gradients for self.params[k]. Don't forget to add      #
        # L2 regularization!                                               #
        #                                                                  #
        # NOTE: To ensure that your implementation matches ours and you    #
        # pass the automated tests, make sure that your L2 regularization  #
        # does not include a factor of 0.5                                 #
        ####################################################################
        loss, dout = softmax_loss(scores, y) 
        num_train = X.shape[0]
        loss /= num_train

        dout2, dw3, db3 = Linear.backward(dout, cache3)
        dout1, dw2, db2 = Linear_ReLU.backward(dout2, cache2)
        dx, dw1, db1 = Conv_ReLU_Pool.backward(dout1, cache1)                
        
        grads['W1'] = dw1
        grads['b1'] = db1
        grads['W2'] = dw2
        grads['b2'] = db2
        grads['W3'] = dw3
        grads['b3'] = db3

        grads['W1'] /= num_train
        grads['b1'] /= num_train
        grads['W2'] /= num_train
        grads['b2'] /= num_train
        grads['W3'] /= num_train
        grads['b3'] /= num_train

        grads['W1'] += 2 * self.reg * W1
        grads['W2'] += 2 * self.reg * W2
        grads['W3'] += 2 * self.reg * W3        


        # Add regularization to the loss.
        loss += self.reg * torch.sum(W1 * W1) + self.reg * torch.sum(W2*W2) + self.reg * torch.sum(W3*W3)
        ###################################################################
        #                             END OF YOUR CODE                    #
        ###################################################################

        return loss, grads


class DeepConvNet(object):
    """
    A convolutional neural network with an arbitrary number of convolutional
    layers in VGG-Net style. All convolution layers will use kernel size 3 and
    padding 1 to preserve the feature map size, and all pooling layers will be
    max pooling layers with 2x2 receptive fields and a stride of 2 to halve the
    size of the feature map.

    The network will have the following architecture:

    {conv - [batchnorm?] - relu - [pool?]} x (L - 1) - linear

    Each {...} structure is a "macro layer" consisting of a convolution layer,
    an optional batch normalization layer, a ReLU nonlinearity, and an optional
    pooling layer. After L-1 such macro layers, a single fully-connected layer
    is used to predict the class scores.

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """
    def __init__(self,
                 input_dims=(3, 32, 32),
                 num_filters=[8, 8, 8, 8, 8],
                 max_pools=[0, 1, 2, 3, 4],
                 batchnorm=False,
                 num_classes=10,
                 weight_scale=1e-3,
                 reg=0.0,
                 weight_initializer=None,
                 dtype=torch.float,
                 device='cpu'):
        """
        Initialize a new network.

        Inputs:
        - input_dims: Tuple (C, H, W) giving size of input data
        - num_filters: List of length (L - 1) giving the number of
          convolutional filters to use in each macro layer.
        - max_pools: List of integers giving the indices of the macro
          layers that should have max pooling (zero-indexed).
        - batchnorm: Whether to include batch normalization in each macro layer
        - num_classes: Number of scores to produce from the final linear layer.
        - weight_scale: Scalar giving standard deviation for random
          initialization of weights, or the string "kaiming" to use Kaiming
          initialization instead
        - reg: Scalar giving L2 regularization strength. L2 regularization
          should only be applied to convolutional and fully-connected weight
          matrices; it should not be applied to biases or to batchnorm scale
          and shifts.
        - dtype: A torch data type object; all computations will be performed
          using this datatype. float is faster but less accurate, so you should
          use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.params = {}
        self.num_layers = len(num_filters)+1
        self.max_pools = max_pools
        self.batchnorm = batchnorm
        self.reg = reg
        self.dtype = dtype

        if device == 'cuda':
            device = 'cuda:0'

        #####################################################################
        # TODO: Initialize the parameters for the DeepConvNet. All weights, #
        # biases, and batchnorm scale and shift parameters should be        #
        # stored in the dictionary self.params.                             #
        #                                                                   #
        # Weights for conv and fully-connected layers should be initialized #
        # according to weight_scale. Biases should be initialized to zero.  #
        # Batchnorm scale (gamma) and shift (beta) parameters should be     #
        # initilized to ones and zeros respectively.                        #
        #####################################################################
        
        C, H, W = input_dims    
        kernel_size = 3
        pad = 1
        pool_kernel_size = 2
        pool_stride = 2
        HH, WW = 3, 3
        H_conv_out, W_conv_out = H, W
        num_channels =0

        # For each layers
        # print("Total num_layers: ", self.num_layers)
        for i in range(len(num_filters)):
          F = num_filters[i]
          if i == 0:
            num_channels = C # Input channel
          else:
            num_channels  = num_filters[i - 1]

          Wi, bi = 'W' + str(i + 1), 'b' + str(i + 1)
          
          if weight_scale == 'kaiming':
            Din = num_channels 
            Dout = F
            K = HH
            self.params[Wi] = kaiming_initializer(Din, Dout, K, True, device, self.dtype)
          else:
            self.params[Wi] = weight_scale * torch.randn(F, num_channels , HH, WW, device=device, dtype = self.dtype)
          self.params[bi] = torch.zeros(F, device=device, dtype = self.dtype)

          if self.batchnorm:
            gammai, betai = 'gamma' + str(i + 1), 'beta' + str(i + 1)
            coeff_length = num_filters[i]
            self.params[gammai] = torch.ones(coeff_length, device=device, dtype = self.dtype)
            self.params[betai] = torch.zeros(coeff_length, device=device, dtype = self.dtype)

          if i in max_pools:
            H_conv_out, W_conv_out =  H_conv_out // 2, W_conv_out // 2 # Floor division (largest interger that is <= x)

        # For the last layer
        i = i + 1 # e.g. i = 5 (i from 0-4 are all Conv layers (W1~W5))
        Wi, bi = 'W' + str(i + 1), 'b' + str(i + 1) # W6, b6
        if weight_scale == 'kaiming':
          Din = F * H_conv_out * W_conv_out
          Dout = num_classes
          K = None
          self.params[Wi] = kaiming_initializer(Din, Dout, K, False, device, self.dtype)
        else:
          self.params[Wi] = weight_scale * torch.randn(F * H_conv_out * W_conv_out, num_classes,device=device, dtype = self.dtype)
        self.params[bi] = torch.zeros(num_classes, device=device, dtype = self.dtype)
        # print("Last filter: ", i)
        # print("W",i," : shape ", self.params[Wi].shape)


        ################################################################
        #                      END OF YOUR CODE                        #
        ################################################################

        # With batch normalization we need to keep track of running
        # means and variances, so we need to pass a special bn_param
        # object to each batch normalization layer. You should pass
        # self.bn_params[0] to the forward pass of the first batch
        # normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.batchnorm:
            self.bn_params = [{'mode': 'train'}
                              for _ in range(len(num_filters))]

        # Check that we got the right number of parameters
        if not self.batchnorm:
            params_per_macro_layer = 2  # weight and bias
        else:
            params_per_macro_layer = 4  # weight, bias, scale, shift
        num_params = params_per_macro_layer * len(num_filters) + 2
        msg = 'self.params has the wrong number of ' \
              'elements. Got %d; expected %d'
        msg = msg % (len(self.params), num_params)
        assert len(self.params) == num_params, msg

        # Check that all parameters have the correct device and dtype:
        for k, param in self.params.items():
            msg = 'param "%s" has device %r; should be %r' \
                  % (k, param.device, device)
            assert param.device == torch.device(device), msg
            msg = 'param "%s" has dtype %r; should be %r' \
                  % (k, param.dtype, dtype)
            assert param.dtype == dtype, msg

    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'dtype': self.dtype,
          'params': self.params,
          'num_layers': self.num_layers,
          'max_pools': self.max_pools,
          'batchnorm': self.batchnorm,
          'bn_params': self.bn_params,
        }
        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = dtype
        self.reg = checkpoint['reg']
        self.num_layers = checkpoint['num_layers']
        self.max_pools = checkpoint['max_pools']
        self.batchnorm = checkpoint['batchnorm']
        self.bn_params = checkpoint['bn_params']

        for p in self.params:
            self.params[p] = \
                self.params[p].type(dtype).to(device)

        for i in range(len(self.bn_params)):
            for p in ["running_mean", "running_var"]:
                self.bn_params[i][p] = \
                    self.bn_params[i][p].type(dtype).to(device)

        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the deep convolutional
        network.
        Input / output: Same API as ThreeLayerConvNet.
        """
        X = X.to(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params since they
        # behave differently during training and testing.
        if self.batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None

        # pass conv_param to the forward pass for the
        # convolutional layer
        # Padding and stride chosen to preserve the input
        # spatial size
        filter_size = 3
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        #########################################################
        # TODO: Implement the forward pass for the DeepConvNet, #
        # computing the class scores for X and storing them in  #
        # the scores variable.                                  #
        #                                                       #
        # You should use the fast versions of convolution and   #
        # max pooling layers, or the convolutional sandwich     #
        # layers, to simplify your implementation.              #
        #########################################################
        
        # Conv_BatchNorm_ReLU_Pool * (L-1) - Linear
        # print("bn_params: ",self.bn_params)
        # print("bn_param ", bn_param)
        out = X 
        caches =[]
        for layer in range(self.num_layers -1 ):
          wi = "W"+str(layer+1)
          bi = "b"+str(layer+1)
          gammai = "gamma"+str(layer+1)
          betai = "beta"+str(layer+1)

          if layer in self.max_pools: # Should perform maxpool
            if self.batchnorm:
                out, cache = Conv_BatchNorm_ReLU_Pool.forward(out,self.params[wi], self.params[bi], self.params[gammai], self.params[betai], conv_param, self.bn_params[layer], pool_param)
            else:
                out, cache = Conv_ReLU_Pool.forward(out,self.params[wi], self.params[bi], conv_param, pool_param)
            caches.append(cache)
          else: # No maxpool
            if self.batchnorm:
                out, cache = Conv_BatchNorm_ReLU.forward(out,self.params[wi], self.params[bi], self.params[gammai], self.params[betai], conv_param, self.bn_params[layer])
            else:
                out, cache = Conv_ReLU.forward(out,self.params[wi], self.params[bi], conv_param)
            caches.append(cache)
          # print("caches[",layer,"]: len ", len(cache))
          # print("output.shape ", out.shape)
        # Linear
        wlast = "W"+str(self.num_layers)
        blast = "b"+str(self.num_layers)
        # print("Forward: last layer")
        scores, cachef = Linear.forward(out,self.params[wlast], self.params[blast])
        caches.append(cachef)
        # print("caches[",self.num_layers-1,"]: len ", len(cachef))       
        #####################################################
        #                 END OF YOUR CODE                  #
        #####################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ###################################################################
        # TODO: Implement the backward pass for the DeepConvNet,          #
        # storing the loss and gradients in the loss and grads variables. #
        # Compute data loss using softmax, and make sure that grads[k]    #
        # holds the gradients for self.params[k]. Don't forget to add     #
        # L2 regularization!                                              #
        #                                                                 #
        # NOTE: To ensure that your implementation matches ours and you   #
        # pass the automated tests, make sure that your L2 regularization #
        # does not include a factor of 0.5                                #
        ###################################################################

        loss, dout = softmax_loss(scores, y) 
        num_train = X.shape[0]
        # loss /= num_train

        # Last layer
        # print("\n\nTotal caches length: ", len(caches))
        # print("Last layer backward:")
        doutlast, dwlast, dblast = Linear.backward(dout, caches[-1])
        wi = "W"+str(self.num_layers)
        bi = "b"+str(self.num_layers) 
        # KEY: Don't /= num_Train, why?       
        # grads[wi] = dwlast/num_train
        # grads[bi] = dblast/num_train
        grads[wi] = dwlast
        grads[bi] = dblast
        grads[wi] += 2*self.reg* self.params[wi]
        loss += self.reg * torch.sum(self.params[wi]*self.params[wi])

        dx = doutlast

        # From layer L-1 to layer 1: All conv backward
        for layer in range(self.num_layers-1, 0, -1 ):
          wi = "W"+str(layer)
          bi = "b"+str(layer)
          gammai = "gamma"+str(layer)
          betai = "beta"+str(layer)

          # print("Layer ", layer)
          # print("Using cache: ", layer-1)

          if layer-1 in self.max_pools: # Should perform maxpool
            if self.batchnorm:
              dx, dw, db, dgamma, dbeta= Conv_BatchNorm_ReLU_Pool.backward(dx, caches[layer-1])
            else:
              dx, dw, db = Conv_ReLU_Pool.backward(dx, caches[layer-1])
          else:
            if self.batchnorm:
              dx, dw, db, dgamma, dbeta = Conv_BatchNorm_ReLU.backward(dx, caches[layer-1])
            else:
              # print("items in cache: ", len(caches[layer-1]))
              dx, dw, db = Conv_ReLU.backward(dx, caches[layer-1])            
          # grads[wi] = dw/num_train
          # grads[bi] = db/num_train
          grads[wi] = dw
          grads[bi] = db
          grads[wi] += 2*self.reg* self.params[wi]

          if self.batchnorm:
            grads[gammai] = dgamma
            grads[betai] = dbeta 

          # Add regularization to the loss.
          loss += self.reg * torch.sum(self.params[wi]*self.params[wi])



        #############################################################
        #                       END OF YOUR CODE                    #
        #############################################################

        return loss, grads


def find_overfit_parameters():
    weight_scale = 2e-3   # Experiment with this!
    learning_rate = 1e-5  # Experiment with this!
    ###########################################################
    # TODO: Change weight_scale and learning_rate so your     #
    # model achieves 100% training accuracy within 30 epochs. #
    ###########################################################
    weight_scale = 1e-1   # Experiment with this!
    learning_rate = 1e-3  # Experiment with this!
    ###########################################################
    #                       END OF YOUR CODE                  #
    ###########################################################
    return weight_scale, learning_rate


def create_convolutional_solver_instance(data_dict, dtype, device):
    model = None
    solver = None
    #########################################################
    # TODO: Train the best DeepConvNet that you can on      #
    # CIFAR-10 within 60 seconds.                           #
    #########################################################



    # # weight_scale, learning_rate = find_overfit_parameters()
    # weight_scale = 'kaiming'
    # learning_rate = 1e-1
    # model = DeepConvNet(input_dims=input_dims, num_classes=10,
    #                     num_filters=[64,64,128,128],
    #                     max_pools=[0,2],
    #                     reg=1e5, weight_scale=weight_scale, dtype=torch.float32, device='cuda')
    # solver = Solver(model, data_dict,
    #                 print_every=1000, num_epochs=20, batch_size=64,
    #                 update_rule=adam,
    #                 optim_config={
    #                   'learning_rate': learning_rate,},
    #                 device='cuda',
    #         )
    input_dims = data_dict['X_train'].shape[1:]
    batch_size = 128
    weight_scales = ['kaiming']
    num_epochs = 8
    weight_scale =  'kaiming'
    # learning_rate =  9.00E-04
    learning_rate = 9e-4
    reg =  1.00E-03

    model = DeepConvNet(input_dims=input_dims, num_classes=10,
                          num_filters=([8] * 3) + ([32] * 3) + ([128] * 3),
                          max_pools=[3, 6],
                          weight_scale=weight_scale,
                          reg=reg, dtype=dtype, device=device)

    solver = Solver(model, data_dict,
                      num_epochs=num_epochs, batch_size=batch_size,
                      update_rule=adam,
                      optim_config={
                        'learning_rate': learning_rate,
                      },
                      print_every=100, device='cuda')

    #########################################################
    #                  END OF YOUR CODE                     #
    #########################################################
    return solver


def kaiming_initializer(Din, Dout, K=None, relu=True, device='cpu',
                        dtype=torch.float32):
    """
    Implement Kaiming initialization for linear and convolution layers.

    Inputs:
    - Din, Dout: Integers giving the number of input and output dimensions
      for this layer
    - K: If K is None, then initialize weights for a linear layer with
      Din input dimensions and Dout output dimensions. Otherwise if K is
      a nonnegative integer then initialize the weights for a convolution
      layer with Din input channels, Dout output channels, and a kernel size
      of KxK.
    - relu: If ReLU=True, then initialize weights with a gain of 2 to
      account for a ReLU nonlinearity (Kaiming initializaiton); otherwise
      initialize weights with a gain of 1 (Xavier initialization).
    - device, dtype: The device and datatype for the output tensor.

    Returns:
    - weight: A torch Tensor giving initialized weights for this layer.
      For a linear layer it should have shape (Din, Dout); for a
      convolution layer it should have shape (Dout, Din, K, K).
    """
    gain = 2. if relu else 1.
    weight = None
    if K is None:
        ###################################################################
        # TODO: Implement Kaiming initialization for linear layer.        #
        # The weight scale is sqrt(gain / fan_in),                        #
        # where gain is 2 if ReLU is followed by the layer, or 1 if not,  #
        # and fan_in = num_in_channels (= Din).                           #
        # The output should be a tensor in the designated size, dtype,    #
        # and device.                                                     #
        ###################################################################
        # Linear Layer: (Din, Dout)
        fan_in = Din
        weight_scale = (gain/fan_in)**0.5
        weight = weight_scale * torch.randn((Din, Dout), dtype= dtype, device = device)


        ###################################################################
        #                            END OF YOUR CODE                     #
        ###################################################################
    else:
        ###################################################################
        # TODO: Implement Kaiming initialization for convolutional layer. #
        # The weight scale is sqrt(gain / fan_in),                        #
        # where gain is 2 if ReLU is followed by the layer, or 1 if not,  #
        # and fan_in = num_in_channels (= Din) * K * K                    #
        # The output should be a tensor in the designated size, dtype,    #
        # and device.                                                     #
        ###################################################################
        # Conv Layer: (Dout, Din, K, K)
        fan_in = Din * K * K
        weight_scale =  (gain/fan_in)**0.5
        weight = weight_scale * torch.randn((Dout, Din, K, K), dtype= dtype, device = device)

        ###################################################################
        #                         END OF YOUR CODE                        #
        ###################################################################
    return weight


class BatchNorm(object):

    @staticmethod
    def forward(x, gamma, beta, bn_param):
        """
        Forward pass for batch normalization.

        During training the sample mean and (uncorrected) sample variance
        are computed from minibatch statistics and used to normalize the
        incoming data. During training we also keep an exponentially decaying
        running mean of the mean and variance of each feature, and these
        averages are used to normalize data at test-time.

        At each timestep we update the running averages for mean and
        variance using an exponential decay based on the momentum parameter:

        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

        Note that the batch normalization paper suggests a different
        test-time behavior: they compute sample mean and variance for
        each feature using a large number of training images rather than
        using a running average. For this implementation we have chosen to use
        running averages instead since they do not require an additional
        estimation step; the PyTorch implementation of batch normalization
        also uses running averages.

        Input:
        - x: Data of shape (N, D)
        - gamma: Scale parameter of shape (D,)
        - beta: Shift paremeter of shape (D,)
        - bn_param: Dictionary with the following keys:
          - mode: 'train' or 'test'; required
          - eps: Constant for numeric stability
          - momentum: Constant for running mean / variance.
          - running_mean: Array of shape (D,) giving running mean
            of features
          - running_var Array of shape (D,) giving running variance
            of features

        Returns a tuple of:
        - out: of shape (N, D)
        - cache: A tuple of values needed in the backward pass
        """
        mode = bn_param['mode']
        eps = bn_param.get('eps', 1e-5)
        momentum = bn_param.get('momentum', 0.9)

        N, D = x.shape
        # print("N, D, ", N, D)
        running_mean = bn_param.get('running_mean',
                                    torch.zeros(D,
                                                dtype=x.dtype,
                                                device=x.device))
        running_var = bn_param.get('running_var',
                                   torch.zeros(D,
                                               dtype=x.dtype,
                                               device=x.device))
        # print("Initialized: running_mean.shape", running_mean.shape)
        out, cache = None, None
        if mode == 'train':
            ##################################################################
            # TODO: Implement the training-time forward pass for batch norm. #
            # Use minibatch statistics to compute the mean and variance, use #
            # these statistics to normalize the incoming data, and scale and #
            # shift the normalized data using gamma and beta.                #
            #                                                                #
            # You should store the output in the variable out.               #
            # Any intermediates that you need for the backward pass should   #
            # be stored in the cache variable.                               #
            #                                                                #
            # You should also use your computed sample mean and variance     #
            # together with the momentum variable to update the running mean #
            # and running variance, storing your result in the running_mean  #
            # and running_var variables.                                     #
            #                                                                #
            # Note that though you should be keeping track of the running    #
            # variance, you should normalize the data based on the standard  #
            # deviation (square root of variance) instead!                   #
            # Referencing the original paper                                 #
            # (https://arxiv.org/abs/1502.03167) might prove to be helpful.  #
            ##################################################################
            # print("x.shape ", x.shape)

            sample_mean = torch.sum(x, dim=0) / N
            
            xmu = x - sample_mean # x - mu
            xmusqr = xmu**2 # (x - mu)**2 (squared)
            sample_var =  (1/N) * torch.sum(xmusqr, dim=0) # sum over i (samples)

            var_sqrt = torch.sqrt(sample_var + eps)
            denom = 1 / var_sqrt 
            
            xhat = xmu * denom

            out = gamma * xhat + beta
            # sample_var = (torch.sum((x-sample_mean)**2, dim=0))/(N-1)
            cache = (xhat, gamma, beta, denom, var_sqrt, sample_var, xmusqr, xmu, sample_mean, x, eps)

            # running_mean = momentum * running_mean + (1 - momentum) * sample_mean
            # running_var = momentum * running_var + (1 - momentum) * 
            # print(sample_mean.shape)
            # print(running_mean.shape)
            running_mean = momentum * running_mean + (1 - momentum) * sample_mean
            running_var = momentum * running_var + (1 - momentum) * sample_var
            # xhat = (x - sample_mean) / torch.sqrt(sample_var + eps)
            # out = gamma * xhat + beta
            # cache = (xhat, gamma, beta, sample_mean, sample_var, eps)
            ################################################################
            #                           END OF YOUR CODE                   #
            ################################################################
        elif mode == 'test':
            ################################################################
            # TODO: Implement the test-time forward pass for               #
            # batch normalization. Use the running mean and variance to    #
            # normalize the incoming data, then scale and shift the        #
            # normalized data using gamma and beta. Store the result       #
            # in the out variable.                                         #
            ################################################################

            ######### These are for cache ##########
  
            xmu = x - running_mean # x - mu
            var_sqrt = torch.sqrt(running_var + eps)
            denom = 1 / var_sqrt 
            ########################################

            xhat = xmu / var_sqrt
            out = gamma * xhat + beta
            cache = (xhat, gamma, beta, denom, )
            cache = (xhat, gamma, beta, denom, var_sqrt, running_var, xmu, running_mean, x, eps)
            # cache = (xhat, gamma, beta, x, eps)
            ################################################################
            #                      END OF YOUR CODE                        #
            ################################################################
        else:
            raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

        # Store the updated running means back into bn_param
        bn_param['running_mean'] = running_mean.detach()
        bn_param['running_var'] = running_var.detach()

        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for batch normalization.

        For this implementation, you should write out a
        computation graph for batch normalization on paper and
        propagate gradients backward through intermediate nodes.

        Inputs:
        - dout: Upstream derivatives, of shape (N, D)
        - cache: Variable of intermediates from batchnorm_forward.

        Returns a tuple of:
        - dx: Gradient with respect to inputs x, of shape (N, D)
        - dgamma: Gradient with respect to scale parameter gamma,
          of shape (D,)
        - dbeta: Gradient with respect to shift parameter beta,
          of shape (D,)
        """
        dx, dgamma, dbeta = None, None, None
        #####################################################################
        # TODO: Implement the backward pass for batch normalization.        #
        # Store the results in the dx, dgamma, and dbeta variables.         #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167) #
        # might prove to be helpful.                                        #
        # Don't forget to implement train and test mode separately.         #
        #####################################################################
        
        if len(cache)==11: # training mode
          (xhat, gamma, beta, denom, var_sqrt, sample_var, xmusqr, xmu, sample_mean, x, eps)= cache 
          N, D = x.shape
          # step9
          dbeta = 1* torch.sum(dout, dim=0) # sum over samples (i)
          dgammax = 1 * dout #(N,D)
          # step 8
          dgamma = torch.sum( dgammax*xhat, dim=0 ) # (D, ) sum over samples(i)
          dxhat = dgammax * gamma #(N,D)
          
          # step 7
          dxmu1 = dxhat * denom #(N,D)

          ddenom = torch.sum( dxhat*xmu, dim=0) #(D,)

          # step 6
          dvar_sqrt = -1 * (var_sqrt)**-2 * ddenom

          # step 5
          dvar = 0.5 * (sample_var+eps)**-0.5 * dvar_sqrt

          # step 4
          dxmusqr = (1/N)*torch.ones_like(xmu) * dvar

          # step 3
          dxmu2 = 2 *(xmu) * dxmusqr

          # step 2
          dmu =  -torch.sum(dxmu1 + dxmu2, dim=0) # (D,)
          dx1 = (dxmu1 + dxmu2)                   # (N,D)
          
          # step 1 (from mu)
          dx2 = (1/N) * torch.ones_like(x) * dmu  # (N,D)

          # step 0
          dx = dx1 + dx2  # (N,D)

        # test mode:
        else:
          (xhat, gamma, beta, denom, var_sqrt, running_var, xmu, running_mean, x, eps) = cache
          dbeta = torch.sum(dout, dim=0)
          dgamma = torch.sum(dout*xhat, dim=0)
          dxhat = dout*gamma
          dxmu = dxhat * denom
          dx = dxmu

          # the mean and sigma are constanst in test mode


        #################################################################
        #                      END OF YOUR CODE                         #
        #################################################################

        return dx, dgamma, dbeta

    @staticmethod
    def backward_alt(dout, cache):
        """
        Alternative backward pass for batch normalization.
        For this implementation you should work out the derivatives
        for the batch normalizaton backward pass on paper and simplify
        as much as possible. You should be able to derive a simple expression
        for the backward pass. See the jupyter notebook for more hints.

        Note: This implementation should expect to receive the same
        cache variable as batchnorm_backward, but might not use all of
        the values in the cache.

        Inputs / outputs: Same as batchnorm_backward
        """
        dx, dgamma, dbeta = None, None, None
        ###################################################################
        # TODO: Implement the backward pass for batch normalization.      #
        # Store the results in the dx, dgamma, and dbeta variables.       #
        #                                                                 #
        # After computing the gradient with respect to the centered       #
        # inputs, you should be able to compute gradients with respect to #
        # the inputs in a single statement; our implementation fits on a  #
        # single 80-character line.                                       #
        ###################################################################
        dbeta = torch.sum(dout, dim = 0)
        N, D = dout.shape
        
        if len(cache)==11:
          (xhat, gamma, beta, denom, var_sqrt, sample_var, xmusqr, xmu, sample_mean, x, eps)= cache

          dgamma = torch.sum(dout * xhat, dim = 0)
          dx = gamma * ((dout - torch.sum(dout, dim = 0) / N) / var_sqrt
                        - xmu * torch.sum(dout * xmu, dim = 0) / (var_sqrt**3 * (N)))


        else:
          (xhat, gamma, beta, denom, var_sqrt, running_var, xmu, running_mean, x, eps) = cache
          # dbeta = torch.sum(dout, dim=0)
          # dgamma = torch.sum(dout*xhat, dim=0)
          # dxhat = dout*gamma
          # dxmu = dxhat * denom
          # dx = dxmu          
          dgamma = torch.sum(dout*xhat, dim=0)
          dxhat = dout*gamma
          dxmu = dxhat * denom
          dx = dxmu

        # dgamma = torch.sum(dout * xhat, dim = 0)
        # dx = gamma * ((dout - torch.sum(dout, dim = 0) / N) / var_sqrt
        #               - xmu * torch.sum(dout * xmu, dim = 0) / (var_sqrt**3 * (N)))
        #################################################################
        #                        END OF YOUR CODE                       #
        #################################################################

        return dx, dgamma, dbeta

######### Note: Above is for input X.shape=(N,D), Below is for Conv input X.shape=(N,C,H,W) ##########
class SpatialBatchNorm(object): 

    @staticmethod
    def forward(x, gamma, beta, bn_param):
        """
        Computes the forward pass for spatial batch normalization.

        Inputs:
        - x: Input data of shape (N, C, H, W)
        - gamma: Scale parameter, of shape (C,)
        - beta: Shift parameter, of shape (C,)
        - bn_param: Dictionary with the following keys:
          - mode: 'train' or 'test'; required
          - eps: Constant for numeric stability
          - momentum: Constant for running mean / variance. momentum=0
            means that old information is discarded completely at every
            time step, while momentum=1 means that new information is never
            incorporated. The default of momentum=0.9 should work well
            in most situations.
          - running_mean: Array of shape (C,) giving running mean of
            features
          - running_var Array of shape (C,) giving running variance
            of features

        Returns a tuple of:
        - out: Output data, of shape (N, C, H, W)
        - cache: Values needed for the backward pass
        """
        out, cache = None, None

        ################################################################
        # TODO: Implement the forward pass for spatial batch           #
        # normalization.                                               #
        #                                                              #
        # HINT: You can implement spatial batch normalization by       #
        # calling the vanilla version of batch normalization you       #
        # implemented above. Your implementation should be very short; #
        # ours is less than five lines.                                #
        ################################################################
        N, C, H, W = x.shape
        x1 = x.contiguous().transpose(0, 1).reshape(C, N * H * W).transpose(0, 1) # (NHW, C)
        y, cache = BatchNorm.forward(x1, gamma, beta, bn_param)
        out = y.transpose(0, 1).reshape(C, N, H, W).transpose(0, 1) # (Back to N,C,H,W)
   
        ################################################################
        #                       END OF YOUR CODE                       #
        ################################################################

        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Computes the backward pass for spatial batch normalization.
        Inputs:
        - dout: Upstream derivatives, of shape (N, C, H, W)
        - cache: Values from the forward pass
        Returns a tuple of:
        - dx: Gradient with respect to inputs, of shape (N, C, H, W)
        - dgamma: Gradient with respect to scale parameter, of shape (C,)
        - dbeta: Gradient with respect to shift parameter, of shape (C,)
        """
        dx, dgamma, dbeta = None, None, None

        #################################################################
        # TODO: Implement the backward pass for spatial batch           #
        # normalization.                                                #
        #                                                               #
        # HINT: You can implement spatial batch normalization by        #
        # calling the vanilla version of batch normalization you        #
        # implemented above. Your implementation should be very short;  #
        # ours is less than five lines.                                 #
        #################################################################
        N, C, H, W = dout.shape
        dout_reshape = dout.transpose(0, 1).reshape(C, N * H * W).transpose(0, 1) # (NHW, C)
        dx, dgamma, dbeta = BatchNorm.backward_alt(dout_reshape, cache)
        dx = dx.transpose(0,1).reshape(C, N, H, W).transpose(0,1)
        # dout1 = dout.contiguous().transpose(0, 1).contiguous().view(C, -1).transpose(0, 1)
        # dx1, dgamma, dbeta = batchnorm_backward_alt(dout1, cache)
        # dx = dx1.contiguous().transpose(0, 1).view(C, N, H, W).transpose(0, 1).contiguous()

        # out = y.transpose(0, 1).reshape(C, N, H, W).transpose(0, 1) # (Back to N,C,H,W)
        ##################################################################
        #                       END OF YOUR CODE                         #
        ##################################################################

        return dx, dgamma, dbeta

##################################################################
#           Fast Implementations and Sandwich Layers             #
##################################################################


class FastConv(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        stride, pad = conv_param['stride'], conv_param['pad']
        layer = torch.nn.Conv2d(C, F, (HH, WW), stride=stride, padding=pad)
        layer.weight = torch.nn.Parameter(w)
        layer.bias = torch.nn.Parameter(b)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, w, b, conv_param, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, _, _, _, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
            dw = layer.weight.grad.detach()
            db = layer.bias.grad.detach()
            layer.weight.grad = layer.bias.grad = None
        except RuntimeError:
            dx, dw, db = torch.zeros_like(tx), \
                         torch.zeros_like(layer.weight), \
                         torch.zeros_like(layer.bias)
        return dx, dw, db


class FastMaxPool(object):

    @staticmethod
    def forward(x, pool_param):
        N, C, H, W = x.shape
        pool_height, pool_width = \
            pool_param['pool_height'], pool_param['pool_width']
        stride = pool_param['stride']
        layer = torch.nn.MaxPool2d(kernel_size=(pool_height, pool_width),
                                   stride=stride)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, pool_param, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, _, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
        except RuntimeError:
            dx = torch.zeros_like(tx)
        return dx


class Conv_ReLU(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        """
        A convenience layer that performs a convolution
        followed by a ReLU.
        Inputs:
        - x: Input to the convolutional layer
        - w, b, conv_param: Weights and parameters for the
          convolutional layer
        Returns a tuple of:
        - out: Output from the ReLU
        - cache: Object to give to the backward pass
        """
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        out, relu_cache = ReLU.forward(a)
        cache = (conv_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the conv-relu convenience layer.
        """
        conv_cache, relu_cache = cache
        da = ReLU.backward(dout, relu_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db


class Conv_ReLU_Pool(object):

    @staticmethod
    def forward(x, w, b, conv_param, pool_param):
        """
        A convenience layer that performs a convolution,
        a ReLU, and a pool.
        Inputs:
        - x: Input to the convolutional layer
        - w, b, conv_param: Weights and parameters for
          the convolutional layer
        - pool_param: Parameters for the pooling layer
        Returns a tuple of:
        - out: Output from the pooling layer
        - cache: Object to give to the backward pass
        """
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        s, relu_cache = ReLU.forward(a)
        out, pool_cache = FastMaxPool.forward(s, pool_param)
        cache = (conv_cache, relu_cache, pool_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the conv-relu-pool
        convenience layer
        """
        conv_cache, relu_cache, pool_cache = cache
        ds = FastMaxPool.backward(dout, pool_cache)
        da = ReLU.backward(ds, relu_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db


class Linear_BatchNorm_ReLU(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, bn_param):
        """
        Convenience layer that performs an linear transform,
        batch normalization, and ReLU.
        Inputs:
        - x: Array of shape (N, D1); input to the linear layer
        - w, b: Arrays of shape (D2, D2) and (D2,) giving the
          weight and bias for the linear transform.
        - gamma, beta: Arrays of shape (D2,) and (D2,) giving
          scale and shift parameters for batch normalization.
        - bn_param: Dictionary of parameters for batch
          normalization.
        Returns:
        - out: Output from ReLU, of shape (N, D2)
        - cache: Object to give to the backward pass.
        """
        a, fc_cache = Linear.forward(x, w, b)
        a_bn, bn_cache = BatchNorm.forward(a, gamma, beta, bn_param)
        out, relu_cache = ReLU.forward(a_bn)
        cache = (fc_cache, bn_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the linear-batchnorm-relu
        convenience layer.
        """
        fc_cache, bn_cache, relu_cache = cache
        da_bn = ReLU.backward(dout, relu_cache)
        da, dgamma, dbeta = BatchNorm.backward(da_bn, bn_cache)
        dx, dw, db = Linear.backward(da, fc_cache)
        return dx, dw, db, dgamma, dbeta


class Conv_BatchNorm_ReLU(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, conv_param, bn_param):
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        an, bn_cache = SpatialBatchNorm.forward(a, gamma,
                                                beta, bn_param)
        out, relu_cache = ReLU.forward(an)
        cache = (conv_cache, bn_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        conv_cache, bn_cache, relu_cache = cache
        dan = ReLU.backward(dout, relu_cache)
        da, dgamma, dbeta = SpatialBatchNorm.backward(dan, bn_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db, dgamma, dbeta


class Conv_BatchNorm_ReLU_Pool(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, conv_param, bn_param, pool_param):
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        an, bn_cache = SpatialBatchNorm.forward(a, gamma, beta, bn_param)
        s, relu_cache = ReLU.forward(an)
        out, pool_cache = FastMaxPool.forward(s, pool_param)
        cache = (conv_cache, bn_cache, relu_cache, pool_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        conv_cache, bn_cache, relu_cache, pool_cache = cache
        ds = FastMaxPool.backward(dout, pool_cache)
        dan = ReLU.backward(ds, relu_cache)
        da, dgamma, dbeta = SpatialBatchNorm.backward(dan, bn_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db, dgamma, dbeta
