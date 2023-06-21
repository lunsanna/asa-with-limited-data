import numpy as np
from evaluate import Metric
from transformers import Wav2Vec2Processor, EvalPrediction
from typing import Dict

def compute_metrics(processor: Wav2Vec2Processor, 
                    wer_metric: Metric, 
                    cer_metric: Metric, 
                    pred:EvalPrediction) -> Dict:
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
    cer = cer_metric.computer(predictions=pred_str, references=label_str)

    # 5. Print results
    # TODO: use logger?

    print("wer: ", wer, "cer", cer)
    for prediction, reference in zip(pred_str[:10], label_str[:10]):
        print("REFERENCE: ", reference)
        print("PREDICTION: ", prediction)

    return {"wer": wer, "cer": cer}
