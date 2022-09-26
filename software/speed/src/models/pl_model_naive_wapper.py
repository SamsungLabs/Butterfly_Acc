from datetime import datetime
from typing import Optional

from torch.nn import CrossEntropyLoss
import datasets
import torch
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    # get_linear_schedule_with_warmup,
)
from typing import Any, List
from torchmetrics import MetricCollection

from torch.optim.lr_scheduler import LambdaLR

import math

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            ret = float(current_step) / float(max(1, num_warmup_steps))
            ret /= math.sqrt(max(current_step, num_warmup_steps))
            # print ("lr: with sqrt", ret)
            return ret
        # print ("lr lambda:", max(
        #         0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        #     ))
        ret = max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
        ret /= math.sqrt(min(current_step, num_warmup_steps))
        # print ("lr:", ret)
        return ret

    return LambdaLR(optimizer, lr_lambda, last_epoch)

class PL_Model_Naive(LightningModule):
    def __init__(
        self,
        model,
        config,
        metric,
        datamodule,
        num_labels: int,
        task_name: str,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        max_train_step: int = None,
        eval_splits: Optional[list] = None,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.config = config
        self.model = model
        self.datamodule = datamodule
        metrics = MetricCollection({"acc" : metric})
        self.train_metrics = metrics.clone(prefix='train/')
        self.val_metrics = metrics.clone(prefix='val/')
        self.test_metrics = metrics.clone(prefix='test/')
        self.loss_fn = CrossEntropyLoss()
        self.loss_fn_val = CrossEntropyLoss()


    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def step(self, batch: Any, is_train=True):
        try:
            x, y, lengths = batch
        except ValueError:
            x, y = batch
            lengths = None

        targets = y
        # print ("x shape:", x.shape)
        output = self.forward(x) if lengths is None else self.forward(x, lengths=lengths)
        loss = self.loss_fn(output, y) if is_train else self.loss_fn_val(output, y)

        # outputs = self.forward(input_ids=x, labels=y)
        # loss, logits = outputs[:2]

        return loss, output, targets
        # if self.hparams.num_labels >= 1:
        #     preds = torch.argmax(logits, axis=1)
        # elif self.hparams.num_labels == 1:
        #     preds = logits.squeeze()

        # # print ("preds:", preds, " targets:", targets, "  y:", y)

        # return loss, preds, targets

    def shared_step(self, batch: Any, batch_idx: int, phase='train'):
        loss, output, targets = self.step(batch, is_train=(phase == 'train'))
        metrics = getattr(self, f'{phase}_metrics')(output, targets)
        # targets = targets.to(targets.device)
        # print ("target device:", targets.get_device())
        # print ("preds device:", preds.get_device())
        self.log(f"{phase}/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return {"loss": loss, "output": output, "targets": targets}

    def training_step(self, batch: Any, batch_idx: int):
        return self.shared_step(batch, batch_idx, phase='train')

    def validation_step(self, batch: Any, batch_idx: int):
        return self.shared_step(batch, batch_idx, phase='val')

    def test_step(self, batch: Any, batch_idx: int):
        return self.shared_step(batch, batch_idx, phase='test')


    def setup(self, stage=None) -> None:
        if stage != "fit":
            return
        # Get dataloader by calling it - train_dataloader() is called after setup() by default
        # train_loader = self.train_dataloader()

        # Calculate total steps
        tb_size = self.hparams.train_batch_size * max(1, self.trainer.gpus)
        ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)
        self.total_steps = (len(self.datamodule.train_dataloader()) // tb_size) // ab_size if self.hparams.max_train_step is None else self.hparams.max_train_step

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"] # Do we need this?
        # no_decay = []
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        # optimizer_grouped_parameters = self.parameters()
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        # return optimizer
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.total_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        print ("optimizer:", optimizer)
        print ("scheduler", scheduler, " total_steps:", self.total_steps)
        return [optimizer], [scheduler]