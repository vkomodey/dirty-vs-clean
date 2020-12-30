import torch
import argparse
import torchvision
import numpy as np
import albumentations as A
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import LeaveOneOut
from dataloader import get_dataset, get_train_exemplars
from albumentations.pytorch import ToTensorV2

from constants import TrainPhase
from monitor import MetricsMonitor, MetricsMonitorCV

currpath = Path('./artifacts')

resize_dict = {
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225)
}

model_list = {
    'resnet18': torchvision.models.resnet18(pretrained=True),
    'resnet50': torchvision.models.resnet50(pretrained=True),
}

layers_to_substitude = {
    'resnet18': 'fc',
    'resnet50': 'fc'
}

train_transforms = A.Compose([
    A.RandomResizedCrop(224, 224),
    A.RandomRotate90(),
    A.RandomBrightness(),
    A.Normalize(**resize_dict),
    ToTensorV2()
])

val_test_transforms = A.Compose([
    A.Resize(224, 224),
    A.Normalize(**resize_dict),
    ToTensorV2()
])

def prepare_model(params) -> torch.nn.Module:
    model = model_list[params["model"]]

    # freeze all layers for fine-tuning
    for param in model.parameters():
        param.requires_grad = False

    # substitude the layer to be trained
    layer = layers_to_substitude[params["model"]]
    layer_to_substitude = getattr(model, layer)
    setattr(model, layer, torch.nn.Linear(layer_to_substitude.in_features, 2))

    return model

def train(train_dataloader: torch.utils.data.DataLoader,
          val_dataloader: torch.utils.data.DataLoader,
          fold_num: int,
          params):
    device = torch.device(params["device"])
    criterion = torch.nn.CrossEntropyLoss()
    
    train_monitor = MetricsMonitor('train')
    validation_monitor = MetricsMonitor('validation')

    model = prepare_model(params)
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    
    for epoch in tqdm(range(params["epochs"]), desc=f'Fold #{fold_num}'):
        for phase in list(TrainPhase):
            if phase == TrainPhase.train:
                dataloader = train_dataloader
                monitor = train_monitor
                model.train()
            elif phase == TrainPhase.validation:
                dataloader = val_dataloader
                monitor = validation_monitor
                model.eval()

            for batch in dataloader:
                with torch.set_grad_enabled(phase == TrainPhase.train):
                    optimizer.zero_grad()

                    inputs, labels = batch
                    inputs = inputs.to(device)
                    # cross entropy requires that labels should be of type of tensor.Long
                    labels = labels.long().to(device)

                    outputs = model(inputs)
                    _, predictions = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    accuracy = (labels == predictions).float().mean()
                    monitor.add_loss(loss.item())
                    monitor.add_metric(accuracy.item())

                    if phase == TrainPhase.train:
                        loss.backward()
                        optimizer.step()
            
            monitor.loss_reduction()
            monitor.metric_reduction()

    return train_monitor, validation_monitor


def prepare_and_run(params):
    loo = LeaveOneOut()
    plates, classes = get_train_exemplars()
    cv_monitor = MetricsMonitorCV()
    fold_num = 0

    for train_index, val_index in loo.split(plates):
        train_paths, val_paths = plates[train_index], plates[val_index]
        train_classes, val_classes = classes[train_index], classes[val_index]

        train_dataset = get_dataset(list(zip(train_paths, classes)), train_transforms)
        train_dataloader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
        val_dataset = get_dataset(list(zip(val_paths, val_classes)), val_test_transforms)
        # we using leave one out validation, so batch_size should be 1
        val_dataloader = DataLoader(val_dataset, batch_size=1)


        train_monitor, validation_monitor = train(train_dataloader, val_dataloader, fold_num, params)

        cv_monitor.add_train_monitor(train_monitor).add_val_monitor(validation_monitor)

        fold_num+=1

    return cv_monitor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--model', default='resnet18', choices=model_list)
    arg('--lr', default=0.001, type=float)
    arg('--batch-size', default=8, type=int)
    arg('--epochs', default=10, type=int)
    arg('--device', default='cuda', choices=['cuda', 'cpu'])
    arg('--layer-to-substitude', type=str, default='fc')

    args = parser.parse_args()
    params = args.__dict__

    prepare_and_run(args)
