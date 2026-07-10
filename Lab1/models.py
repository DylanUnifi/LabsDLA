import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_feature_extractor():
    """Returns a pretrained ResNet50 with the classification head removed for feature extraction."""
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    # Replace the final fully connected layer with an Identity layer
    # This means model(x) will return the 2048-dimensional feature vector
    model.fc = nn.Identity()
    
    # Freeze all layers since we only use it for feature extraction
    for param in model.parameters():
        param.requires_grad = False
        
    return model

def get_finetune_model(num_classes, num_layers_to_unfreeze=0):
    """
    Returns a pretrained ResNet50 modified for fine-tuning on a new classification task.
    Args:
        num_classes (int): Number of output classes (e.g. 43 for GTSRB).
        num_layers_to_unfreeze (int): Number of blocks to unfreeze from the end.
                                      If 0, only the final FC layer is trained (Linear Evaluation).
    """
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze layers from the end if requested
    # ResNet50 has layer1, layer2, layer3, layer4
    if num_layers_to_unfreeze > 0:
        layers = [model.layer4, model.layer3, model.layer2, model.layer1]
        for i in range(min(num_layers_to_unfreeze, len(layers))):
            for param in layers[i].parameters():
                param.requires_grad = True

    # Replace the final layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    # Initialize weights
    nn.init.normal_(model.fc.weight, mean=0.0, std=0.01)
    nn.init.zeros_(model.fc.bias)
    model.fc.requires_grad = True

    return model

def get_detection_model(num_classes, finetuned_backbone=None):
    """
    Returns a Faster R-CNN model with a ResNet50 FPN backbone.
    If finetuned_backbone is provided, its weights are injected into the backbone.
    """
    # Load a pretrained Faster R-CNN on COCO
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    
    if finetuned_backbone is not None:
        # Inject our fine-tuned weights into the FPN backbone's body
        # strict=False is required because the detector backbone uses FrozenBatchNorm2d (no num_batches_tracked)
        # and doesn't have an 'fc' layer. This properly loads conv1, bn1, and layers 1-4.
        model.backbone.body.load_state_dict(finetuned_backbone.state_dict(), strict=False)
        
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model
