import numpy as np

from collections import defaultdict
from matplotlib import pyplot as plt

from .metrics import init_metrics
from .visualization import show_vae
from .backend import init_task


class VAETrainCML:
    def __init__(self, config, model_config, data_config):
        project_name = config["project_name"]
        task_name = f"{config['model_name']}_{config['task_name']}"

        self.task = init_task(
            config, project_name=project_name, task_name=task_name
        )

        self.config = self.task.connect(config)
        self.model_config = self.task.connect(model_config)
        self.data_config = self.task.connect(data_config)
        self.logger = self.task.get_logger()

        self.back_field = self.data_config["back_field"]
        mask_land = np.load(self.data_config["mask_path"])
        self.metrics = init_metrics(mask_land)
        self.metric_history = defaultdict(list)
        self.loss_train_history = defaultdict(list)
        self.loss_val_history = defaultdict(list)
        self.epoch = 0

    def log_loss(self, loss, train):
        if train:
            work_dict = self.loss_train_history
        else:
            work_dict = self.loss_val_history
        for key, value in loss.items():
            work_dict[key].append(value)

    def get_val_loss(self):
        return np.mean(self.loss_val_history["loss"])

    def validate(self, recon, gt):
        # shape = [batch, ch, w, h]
        metrics = {}
        for i, name in enumerate(self.back_field):
            for func, r, t in zip(
                self.metrics,
                [recon[:, i]] * len(self.metrics),
                [gt[:, i]] * len(self.metrics),
            ):
                if not (func.__name__ == "iiee" and name != "siconc"):
                    metrics[f"{name}_{func.__name__}"] = func(r, t)
        for metric, value in metrics.items():
            self.metric_history[metric].append(value)

    def log_img(self, recon, gt):
        # shape = [ch, w, h]
        for i, name in enumerate(self.back_field):
            figure = show_vae(before=gt[i], after=recon[i], title=name)
            self.logger.report_matplotlib_figure(
                title=name,
                series=name,
                figure=figure,
                iteration=self.epoch,
            )
            plt.close()

    def end_epoch(self):
        for metric, metric_hist in self.metric_history.items():
            self.logger.report_scalar(
                title=metric,
                series="",
                value=np.mean(metric_hist),
                iteration=self.epoch,
            )
        for key, hist in self.loss_train_history.items():
            self.logger.report_scalar(
                title="loss",
                series=f"{key}_train",
                value=np.mean(hist),
                iteration=self.epoch,
            )
        for key, hist in self.loss_val_history.items():
            self.logger.report_scalar(
                title="loss",
                series=f"{key}_val",
                value=np.mean(hist),
                iteration=self.epoch,
            )

        self.epoch += 1
        self.metric_history = defaultdict(list)
        self.loss_train_history = defaultdict(list)
        self.loss_val_history = defaultdict(list)

    def end_test(self):
        for metric, metric_hist in self.metric_history.items():
            self.logger.report_single_value(
                name=metric + "_test",
                value=np.mean(metric_hist),
            )
        self.metric_history = defaultdict(list)

    def close(self):
        self.task.close()
