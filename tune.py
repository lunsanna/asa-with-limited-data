print("IMPORTING LIBRARIES")
import torch
from torch import nn
import json
import numpy as np
import pandas as pd
import jiwer
import torchaudio
from datasets import load_metric
from transformers import Trainer, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2Model, TrainingArguments

eval=False 
save=True
garbage = False
subtask = True
#model_name = "KBLab/wav2vec2-large-voxpopuli-sv-swedish"
model_name = "KBLab/wav2vec2-large-voxrex-swedish" # a bit better results for common voice

if garbage:
    garbage_print = "GARBAGE"
    model_folder = "models/wav2vec2_garb"
    df_column = 'filtered_transcript'
else:
    garbage_print = "NO GARBAGE"
    model_folder = "models/wav2vec2_no_garb"
    df_column = 'filtered_transcript_no_garbage'

if subtask:
    train_test_column = 'subtask_test_train'
    task_print = "SUBTASK"
else:
    train_test_column = 'task_test_train'
    task_print = "TASK"

if "voxrex" in model_name:
    model_print = "VOXREX"
else:
    model_print = "VOXPOPULI"

print(garbage_print, task_print, model_print)

model_folder=model_folder+'_'+task_print.lower()+"_"+model_print.lower()

class Wav2VecDataset(torch.utils.data.Dataset):
    def __init__(self, arrays, transcripts):
        self.arrays = arrays # array of floats
        self.transcripts = transcripts # list of transcripts

    def __len__(self):
        return len(self.arrays)

    def __getitem__(self, recording_id):
        item = {}
        item['input_values'] = self.arrays[recording_id]
        item['input_labels'] = self.transcripts[recording_id]

        return item


class Wav2VecCollator(object):
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, batch):
        transcripts = [item['input_labels'] for item in batch]
        recordings = [item['input_values'] for item in batch]
        
        batch = self.processor(recordings,
                               sampling_rate=16_000,
                               return_tensors="pt",
                               padding=True)

        with self.processor.as_target_processor():
            
            label_features = self.processor(transcripts,
                                            return_tensors="pt",
                                            padding=True)        

        # replace label padding with -100 to ignore loss correctly
        label_features = label_features["input_ids"].masked_fill(label_features.attention_mask.ne(1), -100)
        
        batch["labels"] = label_features
        
        return batch

def create_vocab(transcripts, garbage=True):

    if garbage:
        vocab = ['<garbage>']
        clean_transcripts = [tr.replace('<garbage>','') for tr in transcripts]
        all_text = " ".join(clean_transcripts)
        vocab_chars = list(set(all_text))
        print(vocab_chars)
        vocab_chars.remove(' ')
        vocab+=vocab_chars
        
    else:
        all_text = " ".join(transcripts)
        vocab = list(set(all_text))
        vocab.remove(' ')

    vocab_dict = {v: k+3 for k, v in enumerate(vocab)}
    vocab_dict["[PAD]"] = 0
    vocab_dict["[UNK]"] = 1  
    vocab_dict["|"] = 2

    return vocab_dict

def add_specials(transcripts):
    refined_transcripts = []
    for tr in transcripts:
        if tr[-1]!=" ":
            tr+=" "
            tr = tr.replace(" ", "|")
        else:
            tr = tr.replace(" ", "|")
        refined_transcripts.append(tr)
    
    return refined_transcripts

def main():
    print("CUDA AVAILABLE", torch.cuda.is_available())
    print('START')

    # load dataset dataframe
    asr_df = pd.read_csv('asr_df.csv')

    # get transcripts
    train_transcripts = list(asr_df[asr_df[train_test_column]=='train'][df_column])
    print("TRAIN!!!!!", len(train_transcripts))
    if eval:
        eval_df = asr_df[asr_df[train_test_column]=='eval']
        test_transcripts = list(eval_df[df_column])
    else:
        test_transcripts = list(asr_df[asr_df[train_test_column]=='test'][df_column])
  
    # get paths
    train_paths = list(asr_df[asr_df[train_test_column]=='train']['recording_path'])
    if eval:
        test_paths = list(eval_df['recording_path'])
    else:
        test_paths = list(asr_df[asr_df[train_test_column]=='test']['recording_path'])

    # read arrays
    print("LOADING AUDIO")
    train_arrays = [torchaudio.load(file)[0].squeeze().tolist() for file in train_paths]
    test_arrays = [torchaudio.load(file)[0].squeeze().tolist() for file in test_paths]

    # create vocab
    print("CREATING VOCAB")
    vocab = create_vocab(train_transcripts, garbage)
    with open('vocab.json', 'w') as vocab_file:
        json.dump(vocab, vocab_file)
    
    # prepare transcripts for tokenization
    train_transcripts = add_specials(train_transcripts)
    test_transcripts = add_specials(test_transcripts)

    #create datasets
    train_dataset = Wav2VecDataset(train_arrays, train_transcripts)
    test_dataset = Wav2VecDataset(test_arrays, test_transcripts)

    # create tokenizer
    tokenizer = Wav2Vec2CTCTokenizer('vocab.json', unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    print(len(tokenizer.get_vocab()))
    print(tokenizer.get_vocab())
    # create feature extractor
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    # create and save processor
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    processor.save_pretrained(model_folder)
    print("processor is saved")

    # create data collator
    data_collator = Wav2VecCollator(processor)

    # initialize model
    model = Wav2Vec2ForCTC.from_pretrained(model_name, gradient_checkpointing=True, 
                                            ctc_loss_reduction="mean")
    print(model.wav2vec2.encoder.layers[23].feed_forward.output_dense.weight[0])
    print(model.lm_head.weight.shape)
    model.lm_head  = nn.Linear(model.config.hidden_size, len(tokenizer))
    model.config.pad_token_id=processor.tokenizer.pad_token_id
    model.config.vocab_size=len(tokenizer)
    print(model.config)
    print(model.lm_head.weight.shape)
    model.freeze_feature_extractor()

    # setting training arguments
    training_args = TrainingArguments(
    output_dir=model_folder,
    group_by_length=True,
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    num_train_epochs=100,
    learning_rate=0.0005,
    warmup_steps=1000,
    fp16=True,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    )
    if save:
        training_args.save_steps=500
        training_args.save_total_limit=5
    else:
        training_args.save_steps=100000

    cer_metric = load_metric("cer")
    wer_metric = load_metric("wer")
    
    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        i=0
        for prediction, reference in zip(pred_str[:10], label_str[:10]):
            print("\n")
            print("REFERENCE", reference)
            print("PREDICTION", prediction)
            i+=1
        
        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)    

        return {"wer" : wer, "cer": cer}

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=processor.feature_extractor
)
    print('weights before')
    print(model.lm_head.weight[0])
    print(model.wav2vec2.encoder.layers[12].feed_forward.output_dense.weight[0])
    
    trainer.train()

    print('weights after')
    print(model.lm_head.weight[0])
    print(model.wav2vec2.encoder.layers[12].feed_forward.output_dense.weight[0])
if __name__ == "__main__":
    main()
