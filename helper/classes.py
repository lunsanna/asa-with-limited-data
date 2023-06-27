#!/usr/bin/env python3
from dataclasses import dataclass
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor, Trainer, is_apex_available

from typing import Union, Optional, List, Dict, Any

if is_apex_available():
    from apex import amp

_is_native_cuda_amp_available = True
from torch.cuda.amp import autocast

@dataclass
class DataCollatorCTCWithPadding:
    """
    A custom data collator that handles inputs and labels differently due to their stark difference in seq length.
    Args:
        processor (transformers.Wav2Vec2Processor)
        padding (default True) select padding strategy to the returned seq
            - True or longest: pad to the longest seq in the batch
            - 'max_lenght': pad to a max length specified with the arg max_length, if not provided, pad to max acceptable length for the model
            - False or 'do_not_pad': no padding, output batch with different seq length
        max_length (int, optional): max length of the 'input_values' of the returned list, optionally padded
        max_length_labels (int, optional): max length of the 'labels' of the returned list, optionally padded
        pad_to_multiple of (int, optional): if set, will pad the seq to the multiple of the provided val
    """
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split the inputs and labels
        input_features = [{"input_values": feature["input_values"]}
                          for feature in features]
        label_features = [{"input_ids": feature["labels"]}
                          for feature in features]

        batch = self.processor.pad(input_features,
                                   padding=self.padding,
                                   max_length=self.max_length,
                                   pad_to_multiple_of=self.pad_to_multiple_of,
                                   return_tensors="pt")  # return PyTorch torhc.Tensor objects

        labels_batch = self.processor.pad(labels=label_features,
                                          padding=self.padding,
                                          max_length=self.max_length_labels,
                                          pad_to_multiple_of=self.pad_to_multiple_of_labels,
                                          return_tensors="pt")  # return PyTorch torhc.Tensor objects

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels
        return batch
    


class CTCTrainer(Trainer):
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """

        model.train()
        inputs = self._prepare_inputs(inputs)

        if self.use_cuda_amp:
            with autocast():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            if model.module.config.ctc_loss_reduction == "mean":
                loss = loss.mean()
            elif model.module.config.ctc_loss_reduction == "sum":
                loss = loss.sum() / (inputs["labels"] >= 0).sum()
            else:
                raise ValueError(f"{model.config.ctc_loss_reduction} is not valid. Choose one of ['mean', 'sum']")

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_cuda_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()