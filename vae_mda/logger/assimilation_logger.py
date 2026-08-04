import numpy as np
import time

from pathlib import Path
from collections import defaultdict
from matplotlib import pyplot as plt


from .metrics import init_metrics
from .visualization import show_assimilation, show_vae
from .backend import init_task


class AssimilationCML:
    def __init__(self, config, model_config, data_config):
        project_name = config["project_name"]
        task_name = f"{config['model_name']}_{config['task_name']}"
        self.task = init_task(
            config, project_name=project_name, task_name=task_name
        )
        print(f"{config=}")
        print(f"{model_config=}")
        print(f"{data_config=}")
        self.config = self.task.connect(config)
        self.model_config = self.task.connect(model_config)
        self.data_config = self.task.connect(data_config)
        self.logger = self.task.get_logger()

        self.back_field = self.data_config["back_field"]
        self.val_source = self.data_config["val_source"]
        self.val_field = self.data_config["val_field"]

        mask_land = np.load(self.data_config["mask_path"])
        self.metrics = init_metrics(mask_land)
        self.sparse_metrics = init_metrics(mask_land, sparse_gt=True)
        self.metric_history = defaultdict(list)
        self.iteration = 0
        self.save_out = config["save_out"]
        if self.save_out:
            self.out_path = Path(config["out_dir"])
            self.save_tag = config["save_tag"]

    def log(
        self, loss, data, init_x, x, obs, val_1, val_2,
        stopped_step=None, best_validation_step=None,
        timings=None, day_start=None,
    ):
        log_start = time.perf_counter()
        timings = {} if timings is None else timings.copy()
        if stopped_step is not None:
            self.logger.report_scalar(
                title="early_stopping",
                series="stopped_step",
                value=stopped_step,
                iteration=self.iteration,
            )
        if best_validation_step is not None:
            self.logger.report_scalar(
                title="validation_selection",
                series="best_step",
                value=best_validation_step,
                iteration=self.iteration,
            )
        t0 = time.perf_counter()
        self.report_metrics(init_x, x, obs, val_1, val_2)
        timings["metrics"] = time.perf_counter() - t0
        if loss is not None:
            t0 = time.perf_counter()
            self.report_loss(loss, data)
            timings["loss_figure"] = time.perf_counter() - t0
        else:
            timings["loss_figure"] = 0.0
        t0 = time.perf_counter()
        self.report_imgs(init_x, x, obs, val_1, val_2, data)
        timings["field_figures"] = time.perf_counter() - t0
        if self.save_out:
            t0 = time.perf_counter()
            self.save(x, data)
            timings["save_out"] = time.perf_counter() - t0
        else:
            timings["save_out"] = 0.0
        timings["logger_total"] = time.perf_counter() - log_start
        if day_start is not None:
            timings["day_total"] = time.perf_counter() - day_start
        self.report_timings(timings)
        self.iteration += 1

    def save(self, x, data):
        out_path = self.out_path / f"run_{data}T00:00:00+00:0"
        out_path.mkdir(exist_ok=True, parents=True)
        np.save(out_path / f"{self.save_tag}_{data}T00:00:00+00:0.npy", x)

    def close(self):
        # summarization
        for metric, metric_hist in self.metric_history.items():
            self.logger.report_single_value(
                name=metric + "_mean",
                value=np.mean(metric_hist),
            )
        self.task.close()

    def report_loss(self, loss, data):
        if len(loss) == 0:
            return
        figure, ax = plt.subplots(1, 1, figsize=(8, 4))
        if isinstance(loss, dict):
            for series_name, values in loss.items():
                if len(values) == 0:
                    continue
                first_value = values[0]
                if isinstance(first_value, (list, tuple)) and len(first_value) == 2:
                    steps = [item[0] for item in values]
                    series_values = [item[1] for item in values]
                else:
                    steps = np.arange(1, len(values) + 1)
                    series_values = values
                ax.plot(steps, series_values, label=series_name)
            ax.legend(loc="best")
        else:
            ax.plot(np.arange(1, len(loss) + 1), loss)
        ax.set_title(f"Loss {data}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        self.logger.report_matplotlib_figure(
            title="assimilation_loss",
            series=data,
            figure=figure,
            iteration=self.iteration,
        )
        plt.close(figure)

    def report_timings(self, timings):
        for key, value in timings.items():
            self.logger.report_scalar(
                title="timing_seconds",
                series=key,
                value=value,
                iteration=self.iteration,
            )

    def report_imgs(self, init_x, x, obs, val_1, val_2, data):
        for field in self.back_field:
            if field == "sithic":
                continue
            x_inx = self.back_field.index(field)
            sat = obs if field == "siconc" else None
            if field in self.val_field[0]:
                val_inx = self.val_field[0].index(field)
                figure = show_assimilation(
                    init_x[x_inx],
                    sat,
                    x[x_inx],
                    val_1[val_inx],
                    title=f"{field}_{data}",
                )
                self.logger.report_matplotlib_figure(
                    title=f"VAL_{self.val_source[0]}",
                    series=field,
                    figure=figure,
                    iteration=self.iteration,
                )
                plt.close()
            if field in self.val_field[1]:
                val_inx = self.val_field[1].index(field)
                figure = show_assimilation(
                    init_x[x_inx],
                    sat,
                    x[x_inx],
                    val_2[val_inx],
                    title=f"{field}_{data}",
                )
                self.logger.report_matplotlib_figure(
                    title=f"VAL_{self.val_source[1]}",
                    series=field,
                    figure=figure,
                    iteration=self.iteration,
                )
                plt.close()
            if (field not in self.val_field[0]) and (field not in self.val_field[1]):
                figure = show_vae(
                    before=init_x[x_inx],
                    after=x[x_inx],
                    title=f"{field}_{data}",
                )
                self.logger.report_matplotlib_figure(
                    title=field,
                    series=field,
                    figure=figure,
                    iteration=self.iteration,
                )
                plt.close()

    def report_metrics(self, init_x, x, obs, val_1, val_2):
        metrics = {}
        # number of observations
        metrics["obs_count"] = obs[~np.isnan(obs)].shape[0]
        self.logger.report_scalar(
            title="count",
            series="obs",
            value=metrics["obs_count"],
            iteration=self.iteration,
        )

        # metrics: two sources for validation
        for i, (v_source, val) in enumerate(zip(self.val_source, (val_1, val_2))):
            val_len = len(self.val_field[i])
            metric_funcs = self.sparse_metrics if v_source in {"sral", "track_imitation"} else self.metrics

            # fields
            for j, field in enumerate(self.val_field[i]):
                x_inx = self.back_field.index(field)

                # metrics
                for func, mod, pred, gt in zip(
                    metric_funcs,
                    [init_x[x_inx]] * len(metric_funcs),
                    [x[x_inx]] * len(metric_funcs),
                    [val[j]] * len(metric_funcs),
                ):
                    if func.__name__ == "iiee" and field != "siconc":
                        continue
                    metrics[f"{v_source}_{field}_{func.__name__}_model"] = func(mod, gt)
                    metrics[f"{v_source}_{field}_{func.__name__}_assim"] = func(
                        pred, gt
                    )
                    metrics[f"{v_source}_{field}_{func.__name__}_delta"] = func(
                        pred, gt
                    ) - func(mod, gt)
                    # report
                    for key in ["model", "assim"]:  # to one plot
                        self.logger.report_scalar(
                            title=f"{v_source}_{field}_{func.__name__}",
                            series=key,
                            value=metrics[f"{v_source}_{field}_{func.__name__}_{key}"],
                            iteration=self.iteration,
                        )
                    # delta to different plot
                    self.logger.report_scalar(
                        title=f"{v_source}_{field}_{func.__name__}_delta",
                        series="delta",
                        value=metrics[f"{v_source}_{field}_{func.__name__}_delta"],
                        iteration=self.iteration,
                    )

        for metric, value in metrics.items():
            self.metric_history[metric].append(value)
