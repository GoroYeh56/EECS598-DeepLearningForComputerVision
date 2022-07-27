"""
This module contains classes and functions that are common across both, one-stage
and two-stage detector implementations. You have to implement some parts here -
walk through the notebooks and you will find instructions on *when* to implement
*what* in this module.
"""
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.models import feature_extraction


def hello_common():
    print("Hello from common.py!")


class DetectorBackboneWithFPN(nn.Module):
    r"""
    Detection backbone network: A tiny RegNet model coupled with a Feature
    Pyramid Network (FPN). This model takes in batches of input images with
    shape `(B, 3, H, W)` and gives features from three different FPN levels
    with shapes and total strides upto that level:

        - level p3: (3, H /  8, W /  8)      stride =  8
        - level p4: (3, H / 16, W / 16)      stride = 16
        - level p5: (3, H / 32, W / 32)      stride = 32

    NOTE: We could use any convolutional network architecture that progressively
    downsamples the input image and couple it with FPN. We use a small enough
    backbone that can work with Colab GPU and get decent enough performance.
    """

    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels

        # Initialize with ImageNet pre-trained weights.
        _cnn = models.regnet_x_400mf(pretrained=True)

        # Torchvision models only return features from the last level. Detector
        # backbones (with FPN) require intermediate features of different scales.
        # So we wrap the ConvNet with torchvision's feature extractor. Here we
        # will get output features with names (c3, c4, c5) with same stride as
        # (p3, p4, p5) described above.
        self.backbone = feature_extraction.create_feature_extractor(
            _cnn,
            return_nodes={
                "trunk_output.block2": "c3",
                "trunk_output.block3": "c4",
                "trunk_output.block4": "c5",
            },
        )

        # Pass a dummy batch of input images to infer shapes of (c3, c4, c5).
        # Features are a dictionary with keys as defined above. Values are
        # batches of tensors in NHWC format, that give intermediate features
        # from the backbone network. (NCHW)
        dummy_out = self.backbone(torch.randn(2, 3, 224, 224))
        dummy_out_shapes = [(key, value.shape) for key, value in dummy_out.items()]

        # print("dummy_out.type: ", type(dummy_out))
        # print("len ", len(dummy_out))
        
        # print(dummy_out)


        print("For dummy input images with shape: (2, 3, 224, 224)")
        for level_name, feature_shape in dummy_out_shapes:
            print(f"Shape of {level_name} features: {feature_shape}")

        ######################################################################
        # TODO: Initialize additional Conv layers for FPN.                   #
        # HINT: You have to use `dummy_out_shapes` defined above to decide   #
        # the input/output channels of these layers.                         #
        ######################################################################
        # This behaves like a Python dict, but makes PyTorch understand that
        # there are trainable weights inside it.

        # dummy_out_shapes: a list of tuples.
        # each tuple: a pair of str ('c3'), torch.Size

        self.fpn_params = nn.ModuleDict()
        i = 0
        # The same in/out channels as c3~c5, but stride using fpn_strides.value
        FPN_LEVELS = ["p30","p40","p50", "p3","p4","p5"]
        # for level_name, stride_ in self.fpn_strides.items():
        for level_name in FPN_LEVELS:
            
            # channel = dummy_out_shapes[i][1][1] # ('c3', (N, C, H, W) => C: Channels
            # print("level_name ", level_name, "channel ", channel)
            channel = dummy_out_shapes[i%3][1][1]
            
            if i<3:
              kernel_size = (1,1) # padding = 0 (default)
              
              self.fpn_params[level_name] = nn.Conv2d(channel, out_channels, kernel_size, stride=(1,1))
            else:
              kernel_size = (3,3) # padding = 0 (default)
              self.fpn_params[level_name] = nn.Conv2d(out_channels, out_channels, kernel_size, stride=(1,1), padding=(1,1))
            i = i+1
        #### TODO: additional 3x3 conv layers, stride 1 pad 1 to p3~p5


        # padding: depends on the kernel size (1,1) => padding should be zero


        
        ################################################################
        #                      END OF YOUR CODE                        #
        ################################################################

    @property
    def fpn_strides(self):
        """
        Total stride up to the FPN level. For a fixed ConvNet, these values
        are invariant to input image size. You may access these values freely
        to implement your logic in FCOS / Faster R-CNN.
        """
        return {"p3": 8, "p4": 16, "p5": 32}

    def forward(self, images: torch.Tensor):

        # Multi-scale features, dictionary with keys: {"c3", "c4", "c5"}.
        backbone_feats = self.backbone(images)

        fpn_feats = {"p3": None, "p4": None, "p5": None}
        ######################################################################
        # TODO: Fill output FPN features (p3, p4, p5) using RegNet features  #
        # (c3, c4, c5) and FPN conv layers created above.                    #
        # HINT: Use `F.interpolate` to upsample FPN features.                #
        ######################################################################

        # Note: the notebook dummy_fpn_feats = backbone(dummy_images)
        # Enter this function!

        # For each level:
        # p5 = Conv2d(c5)
        # p4 = Conv2d(c4) + 2x upsample(p5)
        # p3 = Conv2d(c3) + 2x upsample(p4)
        
        # Set fpn_feats.values to be tensors! 
        # Pass input images to backbone network => get
        # backbone_feats, a dictionary (c3~c5)

        # First 2x upsample, then Final 3x3 Conv layers

        fpn_feats["p5"] = self.fpn_params["p50"](backbone_feats["c5"])
        p5_upsample = F.interpolate(fpn_feats["p5"], scale_factor=2)
        fpn_feats["p5"] = self.fpn_params["p5"](fpn_feats["p5"])

        fpn_feats["p4"] = self.fpn_params["p40"](backbone_feats["c4"]) + p5_upsample
        p4_upsample = F.interpolate(fpn_feats["p4"], scale_factor=2)
        fpn_feats["p4"] = self.fpn_params["p4"](fpn_feats["p4"])

        fpn_feats["p3"] = self.fpn_params["p30"](backbone_feats["c3"]) + p4_upsample
        fpn_feats["p3"] = self.fpn_params["p3"](fpn_feats["p3"])
        # fpn_feats["p5"] = self.fpn_params["p5"](self.fpn_params["p50"](backbone_feats["c5"]))
        # p5_upsample = F.interpolate(fpn_feats["p5"], scale_factor=2)
        # fpn_feats["p4"] = self.fpn_params["p4"](self.fpn_params["p40"](backbone_feats["c4"])) + p5_upsample
        # p4_upsample = F.interpolate(fpn_feats["p4"], scale_factor=2)
        # fpn_feats["p3"] = self.fpn_params["p3"](self.fpn_params["p30"](backbone_feats["c3"])) + p4_upsample

        ################################################################
        #                      END OF YOUR CODE                        #
        ################################################################

        return fpn_feats


def get_fpn_location_coords(
    shape_per_fpn_level: Dict[str, Tuple],
    strides_per_fpn_level: Dict[str, int],
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Map every location in FPN feature map to a point on the image. This point
    represents the center of the receptive field of this location. We need to
    do this for having a uniform co-ordinate representation of all the locations
    across FPN levels, and GT boxes.

    Args:
        shape_per_fpn_level: Shape of the FPN feature level, dictionary of keys
            {"p3", "p4", "p5"} and feature shapes `(B, C, H, W)` as values.
        strides_per_fpn_level: Dictionary of same keys as above, each with an
            integer value giving the stride of corresponding FPN level.
            See `backbone.py` for more details.

    Returns:
        Dict[str, torch.Tensor]
            Dictionary with same keys as `shape_per_fpn_level` and values as
            tensors of shape `(H * W, 2)` giving `(xc, yc)` co-ordinates of the
            centers of receptive fields of the FPN locations, on input image.
    """

    # Set these to `(N, 2)` Tensors giving absolute location co-ordinates.
    location_coords = {
        level_name: None for level_name, _ in shape_per_fpn_level.items()
    }

    for level_name, feat_shape in shape_per_fpn_level.items():
        level_stride = strides_per_fpn_level[level_name]

        ######################################################################
        # TODO: Implement logic to get location co-ordinates below.          #
        ######################################################################
        B, C, H, W = feat_shape
        H_vec = torch.arange(H, dtype=dtype, device=device).reshape(1,-1).repeat(W,1).permute(1,0).reshape(-1,1)
        W_vec = torch.arange(W, dtype=dtype, device=device).reshape(1,-1).repeat(1,H).reshape(-1,1)
        # HW = torch.hstack((H_vec, W_vec))
        HW = torch.hstack((W_vec, H_vec))
        location_coords[level_name] = (HW+0.5)*level_stride
      
        ######################### Sequential Version ################

        # pixel_coords = torch.ones((H*W, 2), dtype=dtype, device=device)
        # for i in range(H):
        #   J_vector = torch.arange(W)
        #   row = torch.full( (W,1),  (i+0.5)*level_stride )  
        #   cols = ((J_vector+0.5) * level_stride).reshape(-1,1)
        #   pixel_coords[i*W:(i+1)*W ,:] =torch.hstack((row, cols))
        #   # for j in range(W):
        #     # pixel_coords[i*W+j, :] = torch.Tensor((level_stride * (i + 0.5), level_stride * (j + 0.5)))
        # location_coords[level_name] = pixel_coords
        ######################################################################
        #                             END OF YOUR CODE                       #
        ######################################################################
    return location_coords

def iou(box1, box2):
    """
    Return the intersection-over-union of two boxes.

    Args:
        box1: Tensor of shape (N, 4) giving top-left and bottom-right coordinates
            of the bounding boxes to perform NMS on.
        box2: Tensor of shape (N, 4) giving top-left and bottom-right coordinates
            of the bounding boxes to perform NMS on.
    """
    # print("box1.shape ", box1.shape)
    # print("box2.shape ", box2.shape)

    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    w1 = x2_1 - x1_1
    h1 = y2_1 - y1_1
    w2 = x2_2 - x1_2
    h2 = y2_2 - y1_2    

    # print("--")
    # print(box1)
    # print(box2)

    width = torch.min(x2_1,x2_2) - torch.max(x1_1, x1_2)
    height = torch.min(y2_1, y2_2) - torch.max(y1_1, y1_2)
    
    intersection = max(0, width) * max(0, height)
    # if width<0:
    #   width = 0
    # if height<0:
    #   height = 0
    # print("width: ", width)
    # print("height: ", height)    
    # print("inter: ", intersection)
    # if w1<0 or h1<0 or w2<0 or h2<0:
    #   print("Error")

    union = w1*h1 + w2*h2 - intersection

    iou = intersection / union
    # print("iou: ", iou)
    return iou

def vectorized_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute intersection-over-union (IoU) between pairs of box tensors. Input
    box tensors must in XYXY format.

    Args:
        boxes1: Tensor of shape `(M, 4)` giving a set of box co-ordinates.
        boxes2: Tensor of shape `(N, 4)` giving another set of box co-ordinates.

    Returns:
        torch.Tensor
            Tensor of shape (M, N) with `iou[i, j]` giving IoU between i-th box
            in `boxes1` and j-th box in `boxes2`.
    """

    ##########################################################################
    # TODO: Implement the IoU function here.                                 #
    ##########################################################################
    # print("\n===== iou boxes.shape: ======")
    # print(boxes1.shape)
    # print(boxes2.shape)

    x1_1 = boxes1[:,0]
    y1_1 = boxes1[:,1]
    x2_1 = boxes1[:,2]
    y2_1 = boxes1[:,3]
    x1_2 = boxes2[:,0]
    y1_2 = boxes2[:,1]
    x2_2 = boxes2[:,2]
    y2_2 = boxes2[:,3]


    w1 = x2_1 - x1_1 #
    h1 = y2_1 - y1_1
    w2 = x2_2 - x1_2
    h2 = y2_2 - y1_2    

    M, _ = boxes1.shape
    N, _ = boxes2.shape
    # width = torch.zeros((M,N), device=boxes1.device)
    # height = torch.zeros((M,N), device=boxes1.device)


    # print(x2_1.shape)
    # print(x2_2.shape) # (3,)

    # print(torch.min(x2_1.reshape(-1,1).repeat(1,N), x2_2.reshape(-1,1).repeat(1,M)).shape)
    w_min = torch.min(x2_1.reshape(-1,1).repeat(1,N), x2_2.reshape(-1,1).repeat(1,M).T)
    w_max = torch.max(x1_1.reshape(-1,1).repeat(1,N), x1_2.reshape(-1,1).repeat(1,M).T) 
    width  = w_min - w_max
    h_min = torch.min(y2_1.reshape(-1,1).repeat(1,N), y2_2.reshape(-1,1).repeat(1,M).T)
    h_max = torch.max(y1_1.reshape(-1,1).repeat(1,N), y1_2.reshape(-1,1).repeat(1,M).T) 
    height  = h_min - h_max
        
    # for i in range(M):
    #   for j in range(N):
    #     width[i,j] = torch.min(x2_1[i], x2_2[j]) - torch.max(x1_1[i], x1_2[j])
    #     height[i,j] =torch.min(y2_1[i], y2_2[j]) - torch.max(y1_1[i], y1_2[j])

    # width = torch.min(x2_1,x2_2).reshape(-1,1) - torch.max(x1_1, x1_2).reshape(1,-1)
    # height = torch.min(y2_1, y2_2).reshape(-1,1) - torch.max(y1_1, y1_2).reshape(1,-1)

    # print(width.shape)
    # print(height.shape)

    intersection = torch.max(torch.zeros_like(width), width) * torch.max(torch.zeros_like(height), height)

    # print("---- Shape check ---")
    # print((w1 * h1).repeat(N,1).t().shape)
    # print( (w2 * h2).repeat(M,1).shape )
    # print(intersection.shape)
    
    union = (w1 * h1).repeat(N,1).t() + (w2 * h2).repeat(M,1) - intersection
    iou = intersection / union
    # print("iou.shape", iou.shape)
    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################
    return iou


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5):
    """
    Non-maximum suppression removes overlapping bounding boxes.

    Args:
        boxes: Tensor of shape (N, 4) giving top-left and bottom-right coordinates
            of the bounding boxes to perform NMS on.
        scores: Tensor of shpe (N, ) giving scores for each of the boxes.
        iou_threshold: Discard all overlapping boxes with IoU > iou_threshold

    Returns:
        keep: torch.long tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores;
            of shape [num_kept_boxes]
    """

    if (not boxes.numel()) or (not scores.numel()):
        return torch.zeros(0, dtype=torch.long)

    keep = None
    #############################################################################
    # TODO: Implement non-maximum suppression which iterates the following:     #
    #       1. Select the highest-scoring box among the remaining ones,         #
    #          which has not been chosen in this step before                    #
    #       2. Eliminate boxes with IoU > threshold                             #
    #       3. If any boxes remain, GOTO 1                                      #
    #       Your implementation should not depend on a specific device type;    #
    #       you can use the device of the input if necessary.                   #
    # HINT: You can refer to the torchvision library code:                      #
    # github.com/pytorch/vision/blob/main/torchvision/csrc/ops/cpu/nms_kernel.cpp
    #############################################################################
    

    # Make input_boxes be a list of tuple(box_coords, original_index)

    chosen = [] # indices of boxes being chosen to "keep"
    
    # print("num boxes: ", len(boxes))
    # print("scores.shape ", scores.shape)
    scores, orig_indices = torch.sort(scores, descending=True)
    orig_indices = list(orig_indices)
    scores = list(scores)

    # Change boxes indices
    # print(boxes[:10])
    # print("orig_indices length", len(orig_indices))
    boxes = boxes[orig_indices,:]
    # print(boxes[:10])

    while scores:

     
      # 1. Keep current max box index
      max_box_idx = orig_indices[0]
      chosen.append(max_box_idx)
      orig_indices.pop(0)
      scores = scores[1:]


      ######################## TODO: Vectorized NMS
      # Get max vs other iou
      # boxes1 = all max_boxes
      # boxes2 = other left boxes
      boxes1 = boxes[0].reshape(1,-1)
      boxes2 = boxes[1:,:]
      # print("\nboxes1.shape ", boxes1.shape) # Shoule be 1,4
      # print("boxes2.shape ", boxes2.shape) # Shoule be N-1, 4
      iou = vectorized_iou(boxes1, boxes2)

      # Note: keep <= boxes!
      mask = (iou <= iou_threshold)
      # print("mask: ", mask)
      # print("mask shape: " , mask.shape)
      
      scores = torch.as_tensor(scores).reshape(1,-1)[mask]
      orig_indices = torch.as_tensor(orig_indices).reshape(1,-1)[mask]

      # # Shape check: (187)
      # print("scores.shape ", scores.shape)
      # print("orig_indices.shape ", orig_indices.shape)

      # # Convert them back to list
      scores = list(scores)
      orig_indices = list(orig_indices)
      
      # indices = [i for i in range(len(boxes2))]
      indices = torch.arange(len(boxes2)).reshape(1,-1)
      # print("indices.shape ", indices.shape)
      keep_indices = indices[mask]
      # print("boxes indices to be kept: ", keep_indices)

      boxes = boxes[1:] # First remove the max box
      boxes = boxes[keep_indices]
      # print("boxes left: ", boxes.shape)
      # Kepp boxes with mask==True
      # boxes = boxes[orig_indices,:]
      # Note: boxes: (N,4) now N=5000


      ##################################################3

      # # remove scores / orig_indices those indices

      # # 2. Loop over the rest of the boxes and remember indices to be removed
      # # Note: len(scores might change!)
      # # A better way: first remember what index to remove, then remove it later
      # indices_to_be_pop = []
      # for i in range(len(scores)):
      #   if iou(boxes[max_box_idx], boxes[orig_indices[i]]) > iou_threshold:
      #     indices_to_be_pop.append((scores[i], orig_indices[i]))

      # # 3. Remove these boxes
      # for item in indices_to_be_pop:
      #   score, box_idx = item
      #   scores.remove(score)
      #   orig_indices.remove(box_idx)      




      # # 1. Keep current max box index
      # max_box_idx = orig_indices[0]
      # chosen.append(max_box_idx)
      # orig_indices.pop(0)
      # scores = scores[1:]

      # # 2. Loop over the rest of the boxes and remember indices to be removed
      # # Note: len(scores might change!)
      # # A better way: first remember what index to remove, then remove it later
      # indices_to_be_pop = []
      # for i in range(len(scores)):
      #   if iou(boxes[max_box_idx], boxes[orig_indices[i]]) > iou_threshold:
      #     indices_to_be_pop.append((scores[i], orig_indices[i]))

      # # 3. Remove these boxes
      # for item in indices_to_be_pop:
      #   score, box_idx = item
      #   scores.remove(score)
      #   orig_indices.remove(box_idx)      


    keep = torch.tensor(chosen)
    # print("number of boxes in keep: ", len(keep))

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    return keep







def class_spec_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
    iou_threshold: float = 0.5,
):
    """
    Wrap `nms` to make it class-specific. Pass class IDs as `class_ids`.
    STUDENT: This depends on your `nms` implementation.

    Returns:
        keep: torch.long tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores;
            of shape [num_kept_boxes]
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_coordinate = boxes.max()
    offsets = class_ids.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
    # print("boxes.shape", boxes.shape)
    # print("offsets.shape ", offsets.shape)
    # print("offsets[:,None].shape ", offsets[:,None].shape)
    
    boxes_for_nms = boxes + offsets[:, None]
    # print("boxes_for_nms: ", boxes_for_nms.shape)
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep