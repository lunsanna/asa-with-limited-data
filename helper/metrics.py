#!/usr/bin/env python3
import numpy as np
from evaluate import Metric
from transformers import Wav2Vec2Processor, EvalPrediction
from typing import Dict
import logging 

logger = logging.getLogger(__name__)

def compute_metrics(processor: Wav2Vec2Processor, 
                    wer_metric: Metric, 
                    cer_metric: Metric, 
                    pred: EvalPrediction, 
                    print_examples: bool = False) -> Dict:
    """_summary_

    Args:
        processor (Wav2Vec2Processor): _description_
        wer_metric (Metric): wer metric locaded from evaluate pkg
        cer_metric (Metric): cer metric loaded from evaluate pkg
        pred (EvalPrediction): contains predictions, labels_ids and inputs, all `np.ndarray`

    Returns:
        Dict: dict containing the metrics 
    """

    # 1. Get predictions
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    # 2. Replace -100 with padding token id
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    # 3. Decode, convert token ids to chars, without grouping tokens
    # i.e. `h-e-l-l-oo` will not be collapsed to `hello`
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    # 4. Compute metrics
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    # 5. Print results
    logger.info(f"wer: {wer}, cer: {cer}")

    if print_examples and logger.isEnabledFor(logging.DEBUG):
        for prediction, reference in zip(pred_str, label_str):
            logger.debug(f'reference: "{reference}"')
            logger.debug(f'prediction: "{prediction}"')

    return {"wer": wer, "cer": cer}

def round_off(x:float):
    """Used to round off the raitings. 
    Randomly choose between rounding up or down if score is x.5. 

    Args:
        x (float): mean rating 

    Returns:
        int: the rounded rating
    """
    x_truncated = int(x) 
    if x > x_truncated + 0.5: 
        return int(x_truncated + 1)
    elif x < x_truncated + 0.5:
        return x_truncated
    else:
        return x_truncated + int(np.random.uniform() < 0.5)