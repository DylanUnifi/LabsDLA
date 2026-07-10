import wandb
import torch
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import cv2

def extract_features(model, dataloader, device):
    """Extracts features from all images in a dataloader using a pretrained model."""
    model.eval()
    features = []
    labels_list = []
    
    pbar = tqdm(dataloader, desc="[Extracting Features]", leave=False)
    for data, labels in pbar:
        data = data.to(device, non_blocking=True)
        with torch.no_grad():
            output = model(data)
            features.append(output.cpu())
            labels_list.append(labels)
            
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels_list, dim=0)
    return features, labels

def train_svm(train_features, train_labels, val_features, val_labels, test_features, test_labels):
    """Trains a Linear SVM on the extracted features and returns the accuracies."""
    train_features = train_features.numpy()
    train_labels = train_labels.numpy()
    val_features = val_features.numpy()
    val_labels = val_labels.numpy()
    test_features = test_features.numpy()
    test_labels = test_labels.numpy()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_features)
    X_val_scaled = scaler.transform(val_features)
    X_test_scaled = scaler.transform(test_features)

    print("Training Linear SVM... This might take a minute.")
    clf = SVC(kernel='linear', max_iter=1000)
    clf.fit(X_train_scaled, train_labels)

    val_preds = clf.predict(X_val_scaled)
    test_preds = clf.predict(X_test_scaled)

    val_acc = accuracy_score(val_labels, val_preds) * 100.0
    test_acc = accuracy_score(test_labels, test_preds) * 100.0

    return val_acc, test_acc

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Trains the model for one epoch."""
    model.train()
    running_loss, correct, total = 0, 0, 0
    pbar = tqdm(loader, desc="[Train]", leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        outs = model(inputs)
        loss = criterion(outs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * labels.size(0)
        _, preds = outs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix(loss=loss.item(), acc=100.*correct/total)
        
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    if wandb.run is not None:
        wandb.log({"train/loss": epoch_loss, "train/acc": epoch_acc})
    return epoch_loss, epoch_acc

def train_detection_epoch(model, loader, optimizer, device):
    """Trains the Faster R-CNN detection model for one epoch (FP32)."""
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc="[Train Det]", leave=False)
    
    for images, targets in pbar:
        images = list(image.to(device, non_blocking=True) for image in images)
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad(set_to_none=True)
        losses.backward()
        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += losses.item()
        pbar.set_postfix(loss=f"{losses.item():.4f}")
        
    epoch_loss = running_loss / len(loader)
    if wandb.run is not None:
        wandb.log({"train_det/loss": epoch_loss})
    return epoch_loss




@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    """Evaluates the model for one epoch."""
    model.eval()
    running_loss, correct, total = 0, 0, 0
    pbar = tqdm(loader, desc="[Eval]", leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        outs = model(inputs)
        loss = criterion(outs, labels)
        
        running_loss += loss.item() * labels.size(0)
        _, preds = outs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
        
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    if wandb.run is not None:
        wandb.log({"val/loss": epoch_loss, "val/acc": epoch_acc})
    return epoch_loss, epoch_acc

def nearest_mean_classifier(train_features, train_labels, val_features, val_labels, test_features, test_labels, num_classes):
    """
    Implements a Nearest-Mean Classifier (Exercise 3.2).
    Computes the mean feature vector for each class in the training set.
    Classifies validation and test samples by finding the nearest class mean using Cosine Similarity.
    """
    # Compute mean features per class
    class_means = []
    for c in range(num_classes):
        # Mask for the current class
        mask = (train_labels == c)
        if mask.sum() > 0:
            mean_c = train_features[mask].mean(dim=0)
        else:
            mean_c = torch.zeros(train_features.size(1))
        class_means.append(mean_c)
        
    class_means = torch.stack(class_means) # Shape: (num_classes, feature_dim)
    
    # Normalize means and features for Cosine Similarity
    class_means_norm = F.normalize(class_means, p=2, dim=1)
    val_features_norm = F.normalize(val_features, p=2, dim=1)
    test_features_norm = F.normalize(test_features, p=2, dim=1)
    
    # Compute similarities: matrix multiplication gives cosine similarity
    val_sims = torch.mm(val_features_norm, class_means_norm.t()) # Shape: (num_val, num_classes)
    test_sims = torch.mm(test_features_norm, class_means_norm.t())
    
    # Predictions are the classes with highest similarity
    val_preds = val_sims.argmax(dim=1)
    test_preds = test_sims.argmax(dim=1)
    
    val_acc = (val_preds == val_labels).float().mean().item() * 100.0
    test_acc = (test_preds == test_labels).float().mean().item() * 100.0
    
    return val_acc, test_acc

def plot_confusion_matrix(model, dataloader, device, num_classes):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outs = model(inputs)
            preds = outs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
    if wandb.run is not None:
        wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(preds=all_preds, y_true=all_labels)})
    else:
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(15, 12))
        sns.heatmap(cm, cmap='cividis', fmt='d', annot=False)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Class")
        plt.ylabel("True Class")
        plt.show()

def show_worst_predictions(model, dataloader, device, num_images=5):
    model.eval()
    worst_losses = []
    worst_images = []
    worst_preds = []
    worst_labels = []
    
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels_gpu = inputs.to(device), labels.to(device)
            outs = model(inputs)
            loss = criterion(outs, labels_gpu)
            
            preds = outs.argmax(dim=1).cpu()
            
            for i in range(len(inputs)):
                if preds[i] != labels[i]:
                    worst_losses.append(loss[i].item())
                    worst_images.append(inputs[i].cpu())
                    worst_preds.append(preds[i].item())
                    worst_labels.append(labels[i].item())
                    
    # Sort by loss descending
    sorted_idx = np.argsort(worst_losses)[::-1][:num_images]
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    if wandb.run is not None:
        wandb_images = []
        for idx in sorted_idx:
            img = worst_images[idx] * std + mean
            img = img.permute(1, 2, 0).clip(0, 1).numpy()
            caption = f"True: {worst_labels[idx]}, Pred: {worst_preds[idx]}, Loss: {worst_losses[idx]:.2f}"
            wandb_images.append(wandb.Image(img, caption=caption))
        wandb.log({"Worst Predictions": wandb_images})
    else:
        fig, axes = plt.subplots(1, min(num_images, len(sorted_idx)), figsize=(15, 4))
        if len(sorted_idx) == 1:
            axes = [axes]
        for i, idx in enumerate(sorted_idx):
            img = worst_images[idx]
            img = img * std + mean
            axes[i].imshow(img.permute(1, 2, 0).clip(0, 1).numpy())
            axes[i].set_title(f"True: {worst_labels[idx]}\nPred: {worst_preds[idx]}\nLoss: {worst_losses[idx]:.2f}")
            axes[i].axis('off')
        plt.suptitle("Worst Predictions (Highest Loss)")
        plt.show()

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Store hook handles to allow cleanup and prevent memory leaks
        self._fwd_hook = target_layer.register_forward_hook(self.save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self.save_gradient)

    def remove_hooks(self):
        """Remove registered hooks to prevent memory leaks."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        self.model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward()
        
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.detach()[0]
        
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) == 0:
            return heatmap
        heatmap /= np.max(heatmap)
        return heatmap

def visualize_gradcam(model, target_layer, img_tensor):
    cam = GradCAM(model, target_layer)
    # enable gradient computation for the input
    img_tensor = img_tensor.clone().detach().requires_grad_(True)
    heatmap = cam.generate(img_tensor.unsqueeze(0).to(next(model.parameters()).device))
    # Clean up hooks to prevent memory leaks on repeated calls
    cam.remove_hooks()
    
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    img = img_tensor.detach().cpu()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img * std + mean
    img = img.permute(1, 2, 0).numpy()
    img = np.uint8(255 * img.clip(0, 1))
    
    # Convert heatmap to RGB before blending
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    superimposed = heatmap * 0.4 + img * 0.6
    superimposed = np.uint8(superimposed)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(superimposed)
    plt.title("Grad-CAM")
    plt.axis('off')
    plt.show()

import matplotlib.patches as patches

def visualize_detections(model, dataloader, device, num_images=5):
    """Visualizes predictions of the Faster R-CNN on the dataset."""
    model.eval()
    images_shown = 0
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            
            for i in range(len(images)):
                if images_shown >= num_images:
                    return
                
                img = images[i].cpu().permute(1, 2, 0).numpy()
                boxes = outputs[i]['boxes'].cpu().numpy()
                scores = outputs[i]['scores'].cpu().numpy()
                labels = outputs[i]['labels'].cpu().numpy()
                
                gt_boxes = targets[i]['boxes'].numpy()
                
                if wandb.run is not None:
                    # Create W&B native bounding box format
                    wandb_box_data = []
                    for box in gt_boxes:
                        wandb_box_data.append({
                            "position": {"minX": float(box[0]), "minY": float(box[1]), "maxX": float(box[2]), "maxY": float(box[3])},
                            "class_id": 1,
                            "box_caption": "GT",
                            "domain": "pixel"
                        })
                    
                    pred_box_data = []
                    for box, score, label in zip(boxes, scores, labels):
                        pred_box_data.append({
                            "position": {"minX": float(box[0]), "minY": float(box[1]), "maxX": float(box[2]), "maxY": float(box[3])},
                            "class_id": int(label),
                            "scores": {"confidence": float(score)},
                            "box_caption": f"Pred {label} ({score:.2f})",
                            "domain": "pixel"
                        })
                    
                    wandb_img = wandb.Image(img, boxes={
                        "predictions": {"box_data": pred_box_data},
                        "ground_truth": {"box_data": wandb_box_data}
                    })
                    wandb.log({f"Detection Image {images_shown+1}": wandb_img})
                else:
                    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                    ax.imshow(img)
                    
                    for box in gt_boxes:
                        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], 
                                                 linewidth=2, edgecolor='g', facecolor='none', label='GT')
                        ax.add_patch(rect)
                    
                    for idx, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                        if score > 0.5: # Confidence threshold
                            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], 
                                                     linewidth=2, edgecolor='r', facecolor='none')
                            ax.add_patch(rect)
                            ax.text(box[0], box[1] - 5, f"{label} ({score:.2f})", color='r', fontsize=12, weight='bold')
                    
                    plt.title(f"Detection Image {images_shown+1}")
                    plt.axis('off')
                    plt.show()
                images_shown += 1

import torchvision.ops as ops

def evaluate_detection_model(model, dataloader, device, iou_threshold=0.5, score_threshold=0.3):
    """
    Evaluates detection performance by computing Accuracy @ IoU threshold.
    Also provides diagnostic output for the first batch to debug mismatches.
    """
    model.eval()
    
    tp_total = 0
    fp_total = 0
    fn_total = 0
    
    # Class-agnostic metrics (ignore label matching, only IoU)
    tp_agnostic = 0
    fp_agnostic = 0
    fn_agnostic = 0
    
    debug_printed = False
    num_images = 0
    
    print("Evaluating Detection Model on Test Set...")
    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            
            for i in range(len(outputs)):
                num_images += 1
                pred_boxes = outputs[i]['boxes'].cpu()
                pred_scores = outputs[i]['scores'].cpu()
                pred_labels = outputs[i]['labels'].cpu()
                
                gt_boxes = targets[i]['boxes'].cpu()
                gt_labels = targets[i]['labels'].cpu()
                
                # Debug: print first image's details
                if not debug_printed and len(gt_boxes) > 0:
                    print("\n--- DIAGNOSTIC (first image with GT) ---")
                    print(f"  GT boxes:   {gt_boxes[:3].tolist()}")
                    print(f"  GT labels:  {gt_labels[:3].tolist()}")
                    print(f"  Pred boxes (top 3 by score): {pred_boxes[:3].tolist()}")
                    print(f"  Pred scores (top 3):         {pred_scores[:3].tolist()}")
                    print(f"  Pred labels (top 3):         {pred_labels[:3].tolist()}")
                    print(f"  Unique pred labels: {pred_labels.unique().tolist()}")
                    print(f"  Unique GT labels:   {gt_labels.unique().tolist()}")
                    if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                        debug_ious = ops.box_iou(pred_boxes[:3], gt_boxes[:3])
                        print(f"  IoU matrix (top 3x3): {debug_ious.tolist()}")
                    print("--- END DIAGNOSTIC ---\n")
                    debug_printed = True
                
                # Filter by score threshold
                keep = pred_scores >= score_threshold
                pred_boxes = pred_boxes[keep]
                pred_labels = pred_labels[keep]
                
                if len(gt_boxes) == 0:
                    fp_total += len(pred_boxes)
                    fp_agnostic += len(pred_boxes)
                    continue
                    
                if len(pred_boxes) == 0:
                    fn_total += len(gt_boxes)
                    fn_agnostic += len(gt_boxes)
                    continue
                
                # Compute IoU between all predicted and GT boxes
                ious = ops.box_iou(pred_boxes, gt_boxes)
                
                # ---- Class-aware matching ----
                matched_gt = set()
                tp = 0
                for p_idx in range(len(pred_boxes)):
                    best_iou = 0
                    best_gt_idx = -1
                    for g_idx in range(len(gt_boxes)):
                        if g_idx in matched_gt:
                            continue
                        if pred_labels[p_idx] == gt_labels[g_idx]:
                            iou = ious[p_idx, g_idx].item()
                            if iou > best_iou:
                                best_iou = iou
                                best_gt_idx = g_idx
                    if best_iou >= iou_threshold:
                        tp += 1
                        matched_gt.add(best_gt_idx)
                
                fp_total += len(pred_boxes) - tp
                fn_total += len(gt_boxes) - tp
                tp_total += tp
                
                # ---- Class-agnostic matching (ignore labels) ----
                matched_gt_ag = set()
                tp_ag = 0
                for p_idx in range(len(pred_boxes)):
                    best_iou = 0
                    best_gt_idx = -1
                    for g_idx in range(len(gt_boxes)):
                        if g_idx in matched_gt_ag:
                            continue
                        iou = ious[p_idx, g_idx].item()
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = g_idx
                    if best_iou >= iou_threshold:
                        tp_ag += 1
                        matched_gt_ag.add(best_gt_idx)
                
                fp_agnostic += len(pred_boxes) - tp_ag
                fn_agnostic += len(gt_boxes) - tp_ag
                tp_agnostic += tp_ag

    # Class-aware results
    denom = tp_total + fp_total + fn_total
    accuracy = tp_total / denom if denom > 0 else 0
    
    # Class-agnostic results
    denom_ag = tp_agnostic + fp_agnostic + fn_agnostic
    accuracy_ag = tp_agnostic / denom_ag if denom_ag > 0 else 0
        
    if wandb.run is not None:
        wandb.log({
            "val_det/accuracy": accuracy,
            "val_det/accuracy_agnostic": accuracy_ag
        })
    print(f"\nProcessed {num_images} images")
    print(f"[Class-Aware]    TP: {tp_total}, FP: {fp_total}, FN: {fn_total} | Accuracy @ IoU={iou_threshold}: {accuracy:.4f}")
    print(f"[Class-Agnostic] TP: {tp_agnostic}, FP: {fp_agnostic}, FN: {fn_agnostic} | Accuracy @ IoU={iou_threshold}: {accuracy_ag:.4f}")
    return accuracy