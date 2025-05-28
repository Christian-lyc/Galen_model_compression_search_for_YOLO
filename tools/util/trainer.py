import copy
import pickle
from pathlib import Path

import torch
from torch import optim
#import wandb
from tqdm import tqdm
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
from runtime.data.data_provider import ADataProvider
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.engine.trainer import BaseTrainer
import math
import numpy as np

class Log:

    def __init__(self, log_dir: str, identifier: str):
        self._logs_dict = dict()
        self._log_dir = log_dir
        self._identifier = identifier

    def append(self, new_entry_dict):
        for key, value in new_entry_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.cpu().detach().item()
            if key in self._logs_dict:
                self._logs_dict[key].append(value)
            else:
                self._logs_dict[key] = [value]

    def store(self, additional_logs):
        self._logs_dict.update(additional_logs)
        path = Path(self._log_dir)
        path.mkdir(exist_ok=True, parents=True)
        path = path / (self._identifier + "_retrain.pickle")
        with open(path, "wb") as file:
            pickle.dump(self._logs_dict, file)


class Trainer(DetectionValidator,BaseTrainer):
    def __init__(self,
                 data_provider: ADataProvider,
                 target_device: torch.device,
                 args,
                 train_epochs: int = 50,
                 train_lr=0.001,
                 train_mom=0.4,
                 class_num=6,
                 use_adadelta=False,
                 store_dir="./results/checkpoints",
                 log_dir="./logs/train",
                 log_file_name="train.out",
                 add_identifier="",
                 model_name="model",
                 model=None):
        super(Trainer,self).__init__()
        self._validate_episodes = 1
        self._data_provider = data_provider
        self._target_device = target_device
        self._train_epochs = train_epochs
        self._train_lr = train_lr
        self._train_mom = train_mom

        self._criterion = torch.nn.CrossEntropyLoss()
        self._store_dir = store_dir
        self._add_identifier = add_identifier
        self._model_name = model_name
        self._use_adadelta = use_adadelta
        self._logs = Log(log_dir, log_file_name)
        self.args=get_cfg(DEFAULT_CFG)

        vars(self.args).update(model.model.args)
        model.model.args=self.args

        self.args.batch=args.batch_size
        self.criterion=v8DetectionLoss(model.model)
        self.accumulate = max(round(self.args.nbs / args.batch_size), 1)
        self.weight_decay = args.weight_decay * args.batch_size * self.accumulate / self.args.nbs  # scale weight_decay
        self.iterations = math.ceil(len(self._data_provider.train_loader.dataset) / max(args.batch_size, self.args.nbs)) * self._train_epochs


    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by scaling and converting to float."""
        batch['img'] = batch['img'].to(self._target_device, non_blocking=True).float() / 255
        return batch

    def train(self, model: torch.nn.Module):
        self.data = check_det_dataset(self.args.data)
        optimizer = self.build_optimizer(model=model,name=self.args.optimizer,lr=self.args.lr0,
                                             momentum=self.args.momentum,decay=self.weight_decay,iterations=self.iterations)
        lf = lambda x: (1 - x / self._train_epochs) * (1.0 - self.args.lrf) + self.args.lrf
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        scheduler.last_epoch = - 1

        train_loader = self._data_provider.train_loader
        best_acc = 0.0
        best_model = model
        model.to(self._target_device)
        nb = len(train_loader)
        nw = max(round(self.args.warmup_epochs *
                       nb), 100) if self.args.warmup_epochs > 0 else -1  # number of warmup iterations
        last_opt_step = -1
        for epoch in tqdm(range(self._train_epochs), desc="[Train Epochs]", dynamic_ncols=True):
            model.train()
            for batch_idx, data in tqdm(enumerate(train_loader), desc=f"[Train Epoch {epoch}]",
                                        total=len(train_loader), dynamic_ncols=True):

                ni = batch_idx + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, np.interp(ni, xi, [1, self.args.nbs / self.args.batch]).round())
                    for j, x in enumerate(optimizer.param_groups):
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x['initial_lr'] * lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                inputs, targets = data['img'],data['bboxes']
                inputs=self.preprocess_batch(data)
                inputs = inputs['img'].to(self._target_device)
                targets = targets.to(self._target_device)
                data['bboxes']=data['bboxes'].to(self._target_device)
                data['cls']=data['cls'].to(self._target_device)
                data['batch_idx']=data['batch_idx'].to(self._target_device)

                optimizer.zero_grad()
                logits = model(inputs)
                loss,loss_items=self.criterion([t.to('cuda') for t in logits],data)

                loss.backward()
                if ni - last_opt_step >= self.accumulate:
                    optimizer.step()
                    last_opt_step = ni
                # optimizer.step()
                #preds = torch.argmax(logits, dim=1)
                #batch_acc[batch_idx] = ((preds == targets).sum() / len(preds)).detach().cpu()
                #batch_losses[batch_idx] = loss.detach().cpu()
                #self.log_train(batch_acc, batch_idx, batch_losses, epoch, scheduler)


            scheduler.step()
            if epoch % self._validate_episodes == 0:
                with torch.no_grad():
                    val_acc = self.validate(model)
                    if val_acc['mAP50'] > best_acc:
                        print(f"epoch, {epoch},mAP50, {val_acc['mAP50']}")
                        best_model = copy.deepcopy(model)
                        best_acc = val_acc['mAP50']
                        self.store_model(model, epoch)
        return best_model

    def log_train(self, batch_acc, batch_idx, batch_losses, epoch, scheduler):
        if scheduler:
            lr = scheduler.get_lr()[0]
        else:
            lr = 0.0
        batch_logs = {
            'loss': batch_losses[batch_idx],
            'acc': batch_acc[batch_idx],
            'epoch': epoch,
            'lr': lr
        }
        self._logs.append(batch_logs)
        #wandb.log(batch_logs)

    def store_model(self, model, epoch_idx):
        path = Path(self._store_dir)
        path.mkdir(exist_ok=True, parents=True)
        path = path / (self.create_identifier(epoch_idx) + ".pth")
        torch.save(model.state_dict(), path)

    def create_identifier(self, epoch_idx):
        return f"{self._model_name}_{self._add_identifier}_lr{self._train_lr}_mom{self._train_mom}_ep{epoch_idx}"

    def validate(self, model):
        return self._validate(model, self._data_provider.val_loader, "val")

    def test(self, model):
        return self._validate(model, self._data_provider.test_loader, "test")

    def store_logs(self, additional_logs):
        self._logs.store(additional_logs)

    def _validate(self, model, data_loader, prefix):
        if self.args.conf is None:
            self.args.conf = 0.001 # default conf=0.001
        model.to(self._target_device)
        model.eval()
        with torch.no_grad():
            self.data = check_det_dataset(self.args.data)
            self.init_metrics(model)
            self.device=self._target_device
            for batch_idx, data in tqdm(enumerate(data_loader), total=len(data_loader), desc="[Validate]",
                                        dynamic_ncols=True):
                inputs=self.preprocess_batch(data)
                inputs = inputs['img'].to(self._target_device)
                data['bboxes']=data['bboxes'].to(self._target_device)
                data['cls']=data['cls'].to(self._target_device)

                logits = model(inputs)
                logits = self.postprocess(logits)
                self.update_metrics(logits, data)


            stats = self.get_stats()
            evaluation_metrics = {
            "mAP50": stats['metrics/mAP50(B)']
            }
            return evaluation_metrics

    def log_test(self, acc, loss, prefix):
        test_logs = {
            f"{prefix}-loss": loss,
            f"{prefix}-acc": acc
        }
        self._logs.append(test_logs)
        #wandb.log(test_logs)
