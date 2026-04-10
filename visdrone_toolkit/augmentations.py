"""
Enhanced training configuration for VisDrone.

Implements best practices for small object detection.
"""

import albumentations as A
import torch


def get_training_augmentation():
    """
    Training augmentations optimized for aerial object detection.

    VisDrone has small objects, so we focus on:
    - Multi-scale training
    - Color augmentation (varying lighting conditions)
    - Geometric augmentations
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=20, p=0.6),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=15, p=0.4),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.3),
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=5, p=1.0),
                    A.GaussianBlur(blur_limit=5, p=1.0),
                    A.MedianBlur(blur_limit=5, p=1.0),
                ],
                p=0.3,
            ),
            A.OneOf(
                [
                    A.GaussNoise(var_limit=(10, 50), p=1.0),
                    A.ISONoise(p=1.0),
                ],
                p=0.2,
            ),
            A.OneOf(
                [
                    A.RandomFog(fog_coef_range=(0.1, 0.4), p=1.0),
                    A.RandomRain(slant_range=(-10, 10), drop_width=1, blur_value=3, p=1.0),
                    A.RandomShadow(p=1.0),
                ],
                p=0.15,
            ),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.2),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            min_visibility=0.25,
            min_area=8.0,
        ),
    )


def get_validation_augmentation():
    """No augmentation for validation."""
    return None


# Training hyperparameters optimized for VisDrone
TRAINING_CONFIG = {
    # Model
    "model": "fasterrcnn_resnet50",
    "pretrained": True,
    # Training
    "epochs": 100,
    "batch_size": 2,
    "accumulation_steps": 2,  # Effective batch size = 4
    "amp": True,
    # Optimizer
    "lr": 0.005,  # Start higher
    "momentum": 0.9,
    "weight_decay": 0.0005,
    # Learning rate schedule
    "lr_schedule": "multistep",  # Step at epochs 60, 80
    "lr_milestones": [60, 80],
    "lr_gamma": 0.1,
    # Early stopping
    "patience": 15,  # Stop if no improvement for 15 epochs
    # Data
    "num_workers": 4,
    "filter_ignored": True,
    "filter_crowd": True,
    # Evaluation
    "eval_score_threshold": 0.3,  # Lower threshold for better recall
    "nms_threshold": 0.3,  # Lower NMS for dense scenes
}


def get_anchor_generator():
    """
    Custom anchor sizes for small objects in VisDrone.

    Default anchors are too large for tiny people/cars from drones.
    """
    from torchvision.models.detection.anchor_utils import AnchorGenerator

    return AnchorGenerator(
        sizes=((16,), (32,), (64,), (128,), (256,)),  # Smaller than default
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    )


# Enhanced model configuration
ENHANCED_MODEL_CONFIG = {
    "fasterrcnn_resnet50": {
        "min_size": 600,
        "max_size": 800,  # Keep at 800 to avoid OOM
        "box_score_thresh": 0.05,  # Lower threshold during training
        "box_nms_thresh": 0.3,  # More aggressive NMS
        "box_detections_per_img": 300,  # Allow more detections per image
        "rpn_pre_nms_top_n_train": 2000,
        "rpn_post_nms_top_n_train": 2000,
        "rpn_pre_nms_top_n_test": 1000,
        "rpn_post_nms_top_n_test": 1000,
    }
}


def get_optimizer_with_warmup(model, config, num_batches_per_epoch):
    """
    Create optimizer with learning rate warmup.

    Warmup helps with training stability.
    """
    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(
        params, lr=config["lr"], momentum=config["momentum"], weight_decay=config["weight_decay"]
    )

    # Warmup for first 500 iterations
    warmup_iters = min(500, num_batches_per_epoch)

    def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor=0.001):
        def f(x):
            if x >= warmup_iters:
                return 1
            alpha = float(x) / warmup_iters
            return warmup_factor * (1 - alpha) + alpha

        return torch.optim.lr_scheduler.LambdaLR(optimizer, f)

    warmup_scheduler = warmup_lr_scheduler(optimizer, warmup_iters)

    return optimizer, warmup_scheduler


def get_lr_scheduler(optimizer, config):
    """Create learning rate scheduler."""
    if config["lr_schedule"] == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=config["lr_milestones"], gamma=config["lr_gamma"]
        )
    elif config["lr_schedule"] == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    else:
        # Default step scheduler
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
