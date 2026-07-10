import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import GTSRB
import torchvision.transforms.v2 as T

class DatasetWrapper(torch.utils.data.Dataset):
    """Wrapper to apply specific transforms to a dataset subset."""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)

def get_dataloaders(data_dir='./data', batch_size=128, num_workers=4, val_split=0.2):
    # Standard ImageNet normalization since we will use pretrained models
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Training transforms with augmentation
    train_transform = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.uint8, scale=True),
        T.Resize((224, 224)),
        T.RandomRotation(15),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)], p=0.5),
        # Avoid RandomHorizontalFlip because traffic signs have semantic meaning that can be reversed (e.g. turn left vs right)
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean, std)
    ])

    # Validation/Test transforms
    val_transform = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.uint8, scale=True),
        T.Resize((224, 224)),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean, std)
    ])

    # Load datasets (GTSRB download can be slow, will output to data_dir)
    train_full = GTSRB(root=data_dir, split='train', download=True)
    test_set = GTSRB(root=data_dir, split='test', download=True)

    # Split train into train and val
    val_size = int(len(train_full) * val_split)
    train_size = len(train_full) - val_size
    train_set, val_set = random_split(train_full, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    # Apply transforms manually using wrapper
    train_wrapper = DatasetWrapper(train_set, transform=train_transform)
    val_wrapper = DatasetWrapper(val_set, transform=val_transform)
    test_wrapper = DatasetWrapper(test_set, transform=val_transform)

    # Dataloaders
    train_loader = DataLoader(train_wrapper, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_wrapper, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_wrapper, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    num_classes = 43 # GTSRB has 43 classes
    return train_loader, val_loader, test_loader, num_classes

class DetectionDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
        
    def __getitem__(self, index):
        item = self.dataset[index]
        image = item['image'].convert('RGB')
        img_w, img_h = image.size  # PIL gives (width, height)
        
        objects = item['objects']
        boxes = []
        labels = []
        
        for bbox, category in zip(objects['bbox'], objects['category']):
            # The keremberke dataset actually provides bboxes in COCO format:
            # [x_min, y_min, width, height]
            x_min, y_min, w, h = bbox
            x_max = x_min + w
            y_max = y_min + h
            
            # Clamp to image boundaries
            x_min = max(0.0, x_min)
            y_min = max(0.0, y_min)
            x_max = min(float(img_w), x_max)
            y_max = min(float(img_h), y_max)
            
            # Skip degenerate boxes
            if x_max <= x_min or y_max <= y_min:
                continue
                
            boxes.append([x_min, y_min, x_max, y_max])
            # Faster R-CNN reserves label 0 for background, so shift by +1
            labels.append(category + 1)
            
        if len(boxes) == 0:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.as_tensor([], dtype=torch.int64)
            area = torch.as_tensor([], dtype=torch.float32)
            iscrowd = torch.as_tensor([], dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            iscrowd = torch.zeros(len(boxes), dtype=torch.int64)
        
        image_id = torch.tensor([index])  # Use index as image_id (always valid)
        
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
        }
        
        if self.transform:
            image = self.transform(image)
            
        return image, target
        
    def __len__(self):
        return len(self.dataset)

def detection_collate_fn(batch):
    return tuple(zip(*batch))

def get_detection_dataloaders(batch_size=8, num_workers=4):
    from datasets import load_dataset
    import torchvision.transforms as T_old
    
    # Load dataset directly from Parquet since custom dataset scripts are deprecated in datasets >= 5.0.0
    ds = load_dataset("parquet", data_files={
        "train": "hf://datasets/keremberke/german-traffic-sign-detection@refs/convert/parquet/full/train/*.parquet",
        "validation": "hf://datasets/keremberke/german-traffic-sign-detection@refs/convert/parquet/full/validation/*.parquet",
        "test": "hf://datasets/keremberke/german-traffic-sign-detection@refs/convert/parquet/full/test/*.parquet",
    })
    # Dynamically detect the actual number of categories in the dataset
    all_cats = set()
    for split_name in ds.keys():
        for item in ds[split_name]:
            all_cats.update(item['objects']['category'])
    actual_num_categories = len(all_cats)
    # +1 for background class (label 0 is reserved by Faster R-CNN)
    num_classes = actual_num_categories + 1
    print(f"Detection dataset: Found {actual_num_categories} categories, using {num_classes} classes (including background)")
    
    # FasterRCNN expects image tensors in [0, 1] range, no manual normalization is required 
    # as the model handles ImageNet normalization internally.
    transform = T_old.Compose([
        T_old.ToTensor()
    ])
    
    train_dataset = DetectionDataset(ds['train'], transform=transform)
    val_dataset = DetectionDataset(ds['validation'], transform=transform)
    test_dataset = DetectionDataset(ds['test'], transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=detection_collate_fn, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=detection_collate_fn, pin_memory=True)
    
    return train_loader, val_loader, test_loader, num_classes
