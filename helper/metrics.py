import numpy as np

def compute_metrics(processor, wer_metric, cer_metric, pred):
    """_summary_

    Args:
        processor (_type_): _description_
        wer_metric (_type_): _description_
        cer_metric (_type_): _description_
        pred (_type_): _description_

    Returns:
        _type_: _description_
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
