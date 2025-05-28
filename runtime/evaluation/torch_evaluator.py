import numpy as np
import torch
import torch.nn.functional as F
#from keras.src.backend.jax.numpy import empty
# from jsonschema.benchmarks.nested_schemas import validator
from tqdm import tqdm

from runtime.compress.compression_policy import CompressionProtocolEntry
from runtime.data.data_provider import ADataProvider
from runtime.evaluation.evaluator import AModelEvaluator
from runtime.feature_extraction.torch_extractor import TorchMACsBOPsExtractor
from runtime.log.logging import LoggingService
from runtime.model.torch_model import TorchExecutableModel
from ultralytics.utils import DEFAULT_CFG, ops
from ultralytics.cfg import get_cfg
from ultralytics.utils.metrics import ConfusionMatrix
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.engine.trainer import BaseTrainer
import math
from torch import optim


class TorchOnlyEvaluator(AModelEvaluator,DetectionValidator,BaseTrainer):

    def __init__(self,
                 data_provider: ADataProvider,
                 logging_service: LoggingService,
                 target_device: torch.device,
                 retrain_epochs=10,
                 retrain_lr=0.01,
                 retrain_mom=0.4,
                 retrain_weight_decay=5e-4,
                 model_reference=None):
        super(TorchOnlyEvaluator,self).__init__()
        self._data_provider = data_provider
        self._logging_service = logging_service
        self._target_device = target_device
        self._retrain_epochs = retrain_epochs
        self._retrain_lr = retrain_lr
        self._retrain_mom = retrain_mom
        self._retrain_weight_decay = retrain_weight_decay
        self.args=get_cfg(DEFAULT_CFG) #
        self._mac_extractor = TorchMACsBOPsExtractor(data_provider.get_random_tensor_with_input_shape())
        self.lb=None
        self.iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.seen = 0
        self.nc=model_reference._reference_model.nc
        self.confusion_matrix=ConfusionMatrix(nc=self.nc)

        vars(self.args).update(model_reference._reference_model.args)
        model_reference._reference_model.args=self.args
        #model_reference._reference_model.args.update(self.args) #**Unpacks key-value pairs from a dictionary as keyword arguments
        #vars convert namespace to dict
        self.criterion=v8DetectionLoss(model_reference._reference_model)
        self.iterations = math.ceil(len(self._data_provider.train_loader.dataset) / max(data_provider.cfg.batch, self.args.nbs)) * self._retrain_epochs



    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by scaling and converting to float."""
        batch['img'] = batch['img'].to(self._target_device, non_blocking=True).float() / 255
        return batch

    def preprocess(self, batch):
        """Preprocesses batch of images for YOLO training."""
        batch['img'] = batch['img'].to(self._target_device, non_blocking=True)
        batch['img'] = (batch['img'].half() if self.args.half else batch['img'].float()) / 255
        for k in ['batch_idx', 'cls', 'bboxes']:
            batch[k] = batch[k].to(self._target_device)

        nb = len(batch['img'])
        self.lb = [torch.cat([batch['cls'], batch['bboxes']], dim=-1)[batch['batch_idx'] == i]
                   for i in range(nb)] if self.args.save_hybrid else []  # for autolabelling

        return batch
    def from_numpy(self, x):
        """
         Convert a numpy array to a tensor.

         Args:
             x (np.ndarray): The array to be converted.

         Returns:
             (torch.Tensor): The converted tensor
         """
        return torch.tensor(x).to(self._target_device) if isinstance(x, np.ndarray) else x

    def retrain(self, executable_model: TorchExecutableModel):

        self.args.conf=None

        pytorch_model=executable_model._reference_model

        pytorch_model.to(self._target_device)
        pytorch_model.train()###


        optimizer = self.build_optimizer(model=pytorch_model,
                                              name=self.args.optimizer,
                                              lr=self._retrain_lr,
                                              momentum=self._retrain_mom,
                                              decay=self._retrain_weight_decay,
                                              iterations=self.iterations)
        lf = lambda x: (1 - x / self._retrain_epochs) * (1.0 - self.args.lrf) + self.args.lrf
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        
        train_loader = self._data_provider._train_loader
        nb = len(train_loader)
        nw = max(round(self.args.warmup_epochs *
                       nb), 100) if self.args.warmup_epochs > 0 else -1  # number of warmup iterations

        epoch_losses = torch.zeros(self._retrain_epochs)
        epoch_acc = torch.zeros(self._retrain_epochs)
        for epoch in tqdm(range(self._retrain_epochs), desc="[Retrain Epochs]", dynamic_ncols=True):
            batch_losses = torch.zeros(len(train_loader))
            batch_acc = torch.zeros((len(train_loader)))
        #
            for batch_idx, data in tqdm(enumerate(train_loader), desc=f"[Retrain Epoch {epoch}]",
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

                logits = pytorch_model(inputs)


                loss,loss_items=self.criterion([t.to('cuda') for t in logits],data)

                loss.backward()
                optimizer.step()

            scheduler.step()
            epoch_losses[epoch] = torch.mean(batch_losses)
            epoch_acc[epoch] = torch.mean(batch_acc)
            self._logging_service.retrain_epoch_completed(batch_losses.numpy(), batch_acc.numpy())
        self._logging_service.retrain_completed(epoch_losses.numpy(), epoch_acc.numpy())

    @staticmethod
    def _acc(preds, targets, topk=(1, 5)):
        with torch.no_grad():
            maxk = max(topk)
            batch_size = targets.size(0)

            _, pred = preds.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(targets.view(1, -1).expand_as(pred))

            res = torch.zeros((len(topk)))
            for i, k in enumerate(topk):
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res[i] = correct_k / batch_size
            return res

    def evaluate(self, executable_model: TorchExecutableModel,
                 compression_protocol: list[CompressionProtocolEntry]) -> dict[str, float]:
        if self.args.conf is None:
            self.args.conf = 0.001 # default conf=0.001
        pytorch_model = executable_model.pytorch_model
        pytorch_model.to(self._target_device)
        pytorch_model.eval()
        self.data = check_det_dataset(self.args.data)
        # self.dataloader =self.get_dataloader(self.data.get(self.args.split), self.args.batch)
        with torch.no_grad():
            val_loader = self._data_provider.val_loader
            # batch_acc = torch.zeros((2, len(val_loader)))

            self.init_metrics(pytorch_model)
            self.device=self._target_device
            for batch_idx, data in tqdm(enumerate(val_loader), total=len(val_loader), desc="[Validate Accuracy]",
                                        dynamic_ncols=True):
                inputs=self.preprocess_batch(data)
                inputs = inputs['img'].to(self._target_device)

                data['bboxes']=data['bboxes'].to(self._target_device)
                data['cls']=data['cls'].to(self._target_device)

                logits = pytorch_model(inputs)
                logits = self.postprocess(logits)
                self.update_metrics(logits, data)

            stats = self.get_stats()
        evaluation_metrics = {
            "mAP50": stats['metrics/mAP50(B)'],

        }
        evaluation_metrics.update(self._compute_with_extractor(executable_model, self._mac_extractor))
        # for now skip mac and bop extractions for phase steps
        return evaluation_metrics

    def sample_log_probabilities(self, executable_model: TorchExecutableModel) -> torch.Tensor:
        pytorch_model = executable_model.pytorch_model
        pytorch_model.to(self._target_device)

        pytorch_model.eval()
        self.data = check_det_dataset(self.args.data)
        eps = 1e-8
        with torch.no_grad():
            self.init_metrics(pytorch_model)
            self.device=self._target_device
            all_outputs = list()
            sens_loader = self._data_provider.sens_loader
            with torch.no_grad():
                for batch_idx, inputs in enumerate(sens_loader):
                    inputs_=self.preprocess_batch(inputs)
                    inputs_ = inputs_['img'].to(self._target_device)
                    inputs['bboxes']=inputs['bboxes'].to(self._target_device)
                    inputs['cls']=inputs['cls'].to(self._target_device)

                    logits = pytorch_model(inputs_)
                    logits_1 = self.postprocess(logits)
                    logits_stack = [t[0,:-2] if t.numel() !=0 else torch.zeros(4).to(self._target_device) for t in logits_1]
                    logits_2 = torch.stack(logits_stack, dim=0)
                    logits_2=logits_2/inputs_.shape[-1]
                    safe_x = torch.clamp(logits_2, min=eps)
                    outputs = torch.log(safe_x).detach()

                return outputs

    @staticmethod
    def _compute_with_extractor(executable_model: TorchExecutableModel, extractor):
        all_layer_metrics = dict()
        with extractor(executable_model) as ex:
            for layer_idx, layer_key in enumerate(executable_model.all_layer_keys()):
                layer_metrics = ex.compute_metric_for_layer(layer_key)
                for metric_key, metric_value in layer_metrics.items():
                    if metric_key not in all_layer_metrics:
                        all_layer_metrics[metric_key] = [metric_value]
                    else:
                        all_layer_metrics[metric_key].append(metric_value)
        return {m_key: np.sum(m_val) for m_key, m_val in all_layer_metrics.items()}
