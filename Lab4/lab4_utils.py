import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def fgsm_attack(images, epsilon, data_grad):
    """Fast Gradient Sign Method (FGSM) attack.

    Generates adversarial examples by perturbing input images along the
    sign of the gradient of the loss w.r.t. the input.

    Args:
        images: Original input images tensor.
        epsilon: Perturbation magnitude.
        data_grad: Gradient of the loss w.r.t. the input images.

    Returns:
        Perturbed images clamped to [0, 1].
    """
    sign_data_grad = data_grad.sign()
    perturbed_images = images + epsilon * sign_data_grad
    perturbed_images = torch.clamp(perturbed_images, 0, 1)
    return perturbed_images

class CNN(nn.Module):
    """Simple CNN for CIFAR-10 classification with optional dropout."""
    def __init__(self, dropout=False):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)  
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  
        
        self.flatten_dim = 256 * 8 * 8  
        self.fc1 = nn.Linear(self.flatten_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        # Use nn.Dropout so it respects model.train()/model.eval() automatically
        self.drop = nn.Dropout() if dropout else nn.Identity()

    def forward(self, x):
        x = F.relu(self.conv1(x))   
        x = F.relu(self.conv2(x))   
        x = F.relu(self.conv3(x))   
        x = F.relu(self.conv4(x))   
        x = F.relu(self.conv5(x))   
        x = torch.flatten(x, 1)
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

class CNNFeatureExtractor(nn.Module):
    def __init__(self, cnn):
        super().__init__()
        self.cnn = cnn
        
    def forward(self, x):
        # If it's the old CNN
        if hasattr(self.cnn, 'conv1') and not hasattr(self.cnn, 'model'):
            x = F.relu(self.cnn.conv1(x))
            x = F.relu(self.cnn.conv2(x))
            x = F.relu(self.cnn.conv3(x))
            x = F.relu(self.cnn.conv4(x))
            x = F.relu(self.cnn.conv5(x))
            x = torch.flatten(x, 1)
            x = F.relu(self.cnn.fc1(x))
            x = F.relu(self.cnn.fc2(x))
            return x
        # If it's ResNetCIFAR
        else:
            # We want the features right before the fc layer
            base_cnn = self.cnn.module if isinstance(self.cnn, torch.nn.DataParallel) else self.cnn
            x = base_cnn.model.conv1(x)
            x = base_cnn.model.bn1(x)
            x = base_cnn.model.relu(x)
            x = base_cnn.model.maxpool(x)

            x = base_cnn.model.layer1(x)
            x = base_cnn.model.layer2(x)
            x = base_cnn.model.layer3(x)
            x = base_cnn.model.layer4(x)

            x = base_cnn.model.avgpool(x)
            x = torch.flatten(x, 1)
            return x


def calc_metrics(id_scores, ood_scores):
    labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])
    auroc = roc_auc_score(labels, scores)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    idx = np.argmax(tpr >= 0.95)
    fpr95 = fpr[idx]
    return auroc, fpr95

class RobustnessEvaluationPipeline:
    def __init__(self, device):
        self.device = device

    def evaluate_pgd(self, model, image, label, epsilon=0.05, alpha=0.01, iters=10, target_class=None):
        """
        Executes PGD attack on a single image.
        If target_class is provided, performs a Targeted Attack to minimize loss with respect to the target class.
        Otherwise, performs an Untargeted Attack to maximize loss with respect to the true label.
        Returns the perturbed image and the predicted class.
        """
        img = image.clone().detach().to(self.device)
        orig_image = img.clone().detach()
        pert_image = img.clone().detach()
        
        # Determine optimization direction
        if target_class is not None:
            t_label = torch.tensor([target_class]).to(self.device)
            multiplier = -1 # Minimize loss for target class
        else:
            t_label = torch.tensor([label]).to(self.device)
            multiplier = 1  # Maximize loss for true class
            
        for _ in range(iters):
            pert_image.requires_grad = True
            output = model(pert_image)
            loss = F.cross_entropy(output, t_label)
            
            model.zero_grad()
            loss.backward()
            
            pert_image = pert_image + alpha * multiplier * pert_image.grad.data.sign()
            eta = torch.clamp(pert_image - orig_image, min=-epsilon, max=epsilon)
            pert_image = torch.clamp(orig_image + eta, min=0, max=1).detach()
            
        pred = model(pert_image).argmax(dim=1).item()
        return pert_image, pred

    def grid_search_odin(self, model, id_loader, ood_loader, temperatures=None, epsilons=None):
        """
        Performs a grid search over temperatures and epsilons for ODIN.
        """
        if temperatures is None:
            temperatures = [1, 10, 100, 1000]
        if epsilons is None:
            epsilons = [0.0005, 0.001, 0.0014, 0.002]
        results = []
        for T in temperatures:
            for eps in epsilons:
                print(f"ODIN Grid Search -> T={T}, eps={eps}")
                id_scores = self.get_odin_scores(model, id_loader, T, eps)
                ood_scores = self.get_odin_scores(model, ood_loader, T, eps)
                auroc, fpr95 = calc_metrics(id_scores, ood_scores)
                results.append({"T": T, "eps": eps, "AUROC": auroc, "FPR@95TPR": fpr95})
        return results

    def get_odin_scores(self, model, loader, temperature=1000, epsilon=0.0014):
        scores = []
        model.eval()
        for inputs, _ in loader:
            inputs = inputs.to(self.device)
            inputs.requires_grad = True
            
            outputs = model(inputs) / temperature
            probs = F.softmax(outputs, dim=1)
            max_probs, _ = torch.max(probs, dim=1)
            
            loss = -torch.log(max_probs).sum()
            model.zero_grad()
            loss.backward()
            
            inputs_pert = inputs - epsilon * inputs.grad.sign()
            inputs_pert = torch.clamp(inputs_pert, 0, 1)
            
            with torch.no_grad():
                outputs_pert = model(inputs_pert) / temperature
                probs_pert = F.softmax(outputs_pert, dim=1)
                max_probs_pert, _ = torch.max(probs_pert, dim=1)
                scores.extend(max_probs_pert.cpu().numpy())
        return np.array(scores)

import torchvision.models as models

class ResNetCIFAR(nn.Module):
    """ResNet18 modified for 32x32 CIFAR-10 images."""
    def __init__(self, num_classes=10):
        super(ResNetCIFAR, self).__init__()
        self.model = models.resnet18(weights=None)
        # Adapt for 32x32 images (no aggressive downsampling at the beginning)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)

