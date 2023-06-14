print("IMPORTING LIBRARIES")
import torch
import numpy as np
import pandas as pd
import jiwer
import torchaudio
import glob
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

model_names = glob.glob("models/wav2vec2_no_garb_subtask_voxrex/check*")
batch_size = 12

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


def get_results(data_generator, model, processor):
        model.to("cuda")
        predictions = []
        for i, batch in enumerate(data_generator):
            print("Working on batch ", i)
            with torch.no_grad():
                logits = model(batch.input_values.to("cuda")).logits
                #logits = model(batch.input_values.to("cuda"),
                #attention_mask=batch.attention_mask.to("cuda")).logits
                pred_ids = torch.argmax(logits, dim=-1)
                pred_str = processor.batch_decode(pred_ids)
                predictions+=pred_str
        return predictions

def main():
    print(torch.cuda.is_available())
    print('START')
    
    # get a dataframe to evaluate
    asr_df = pd.read_csv('asr_df.csv')
    if "subtask"in model_names[0]:
        train_test_column = 'subtask_test_train'
    else:
        train_test_column = 'task_test_train'
    asr_df = asr_df[asr_df[train_test_column]=='test'].copy()

    print('LOADING STUFF')

    for model_name in model_names:
        # load a processor and a model
        print("PREDICTING", model_name)
        processor_name = "/".join(model_name.split('/')[:-1])
        print(processor_name)
        processor = Wav2Vec2Processor.from_pretrained(processor_name)
        model = Wav2Vec2ForCTC.from_pretrained(model_name)
 
        if ("no_garb") in model_name:
            transcript_column = 'filtered_transcript_no_garbage'
        else:
            transcript_column = 'filtered_transcript'
        
        txt_name = "predictions/predictions_"+model_name.replace("/","_")+".txt"

        # prepare input
        df_paths = asr_df['recording_path'].values
        arrays = list([torchaudio.load(file)[0].squeeze().tolist() for file in df_paths])
        transcripts = list(asr_df[transcript_column])
    
        # put stuff into a dataset
        dataset = Wav2VecDataset(arrays, transcripts)

        # create a data_generator
        data_collator = Wav2VecCollator(processor)
        data_generator = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)

        predictions = get_results(data_generator=data_generator, model=model, processor=processor)

        print("WRITING PREDICTION INTO A TXT")
        print(txt_name)
        with open(txt_name,'w', encoding="utf-8") as file:
            for sent in predictions:
                file.write(str(sent)+"\n")

        print("DONE!")

if __name__ == "__main__":
    main()
