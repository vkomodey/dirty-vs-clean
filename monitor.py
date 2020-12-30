import numpy as np
import matplotlib as plt

class MetricsMonitor:
    def __init__(self, title):
        self.epoch_losses = []
        self.epoch_metrics = []
        
        self.batch_losses = []
        self.batch_metrics = []
        
        self.title = title

    def add_loss(self, value):
        self.batch_losses.append(value)

    def add_metric(self, value):
        self.batch_metrics.append(value)
    
    def loss_reduction(self):
        self.epoch_losses.append(np.mean(self.batch_losses))
        self.batch_losses = []

    def metric_reduction(self):
        self.epoch_metrics.append(np.mean(self.batch_metrics))
        self.batch_metrics = []

    def __dict__(self):
        return {
            'epoch_losses': self.epoch_losses,
            'epoch_metrics': self.epoch_metrics,
        }
    
    def plot(self):
        fig, axes = plt.subplots(nrows=1, ncols=2)
        
        loss_ax, metric_ax = axes
        loss_ax.plot(self.epoch_losses)
        loss_ax.set_title('Loss')
        
        metric_ax.plot(self.epoch_metrics)
        metric_ax.set_title('Metric')

def print_cv_stats(arr):
    print(f'mean={np.mean(arr)}, std={np.std(arr)}')

class MetricsMonitorCV:
    def __init__(self):
        self.train = []
        self.validation = []
    
    def add_train_monitor(self, monitor: MetricsMonitor):
        self.train.append(monitor)

        return self

    def add_val_monitor(self, monitor: MetricsMonitor):
        self.validation.append(monitor)

        return self

    def print_train_losses(self):
        print('Train losses')
        for entry in self.train:
            print_cv_stats(entry.epoch_losses)

    def print_train_metrics(self):
        print('Train metrics')
        for entry in self.train:
            print_cv_stats(entry.epoch_metrics)

    def print_validation_losses(self):
        print('Validation losses')
        for entry in self.validation:
            print_cv_stats(entry.epoch_losses)

    def print_validation_metrics(self):
        print('Validation metrics')
        for entry in self.validation:
            print_cv_stats(entry.epoch_metrics)

    def plot_validation_metrics(self):
        values = []

        for monitor in self.validation:
            values.append(np.mean(monitor.epoch_metrics))

        plt.plot(values)
