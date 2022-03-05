#!/usr/bin/env python
# coding: utf-8

# In[14]:


get_ipython().system('pip install -q datasets accelerate sacrebleu wandb')


# In[15]:


from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM

import torch
from torch import optim
from torch.utils.data import DataLoader

from accelerate import Accelerator
from tqdm.auto import tqdm
import wandb


# In[5]:


raw_datasets = load_dataset("news_commentary", lang1="en", lang2="fr")
print(raw_datasets)


# In[6]:


split_datasets = raw_datasets['train'].train_test_split(train_size=0.5, test_size=0.1, seed=42)
print(split_datasets)


# In[7]:


# model name
checkpoint = "Helsinki-NLP/opus-mt-en-fr"
# load the tokenizer for the model
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


# In[8]:


sample = split_datasets['train']['translation'][0]
print(sample)


# In[9]:


tokenizer(sample['en'])


# In[10]:


max_length = 128

def tokenize(examples):
    en_sentences = [sent['en'] for sent in examples['translation']]
    fr_sentences = [sent['fr'] for sent in examples['translation']]

    # tokenize english sentences
    model_inputs = tokenizer(en_sentences, max_length=max_length, truncation=True)

    # tokenize french sentences
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(fr_sentences, max_length=max_length, truncation=True)

    # add tokenized french sentences as labels
    model_inputs['labels'] = labels['input_ids']
    return model_inputs


# In[10]:


tokenized_datasets = split_datasets.map(tokenize, batched=True, remove_columns=['id', 'translation'])


# In[11]:


batch_size = 32

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
collate_fn = DataCollatorForSeq2Seq(tokenizer, model=model)

train_dl = DataLoader(
    tokenized_datasets['train'], 
    batch_size=batch_size, 
    shuffle=True, 
    collate_fn=collate_fn
)

test_dl = DataLoader(
    tokenized_datasets['test'], 
    batch_size=batch_size, 
    shuffle=False, 
    collate_fn=collate_fn
)


# In[12]:


batch = next(iter(train_dl))

print(batch.keys())


# In[13]:


opt = optim.AdamW(model.parameters(), lr=5.34e-6)

accelerator = Accelerator()
train_dl, test_dl, model, opt = accelerator.prepare(
    train_dl, test_dl, model, opt
)


# In[14]:


metric = load_metric('sacrebleu')


# In[15]:


prediction = ['So it can happen anywhere.']
label = ['So it is happen anywhere.']

metric.compute(predictions=prediction, references=[label])


# In[16]:


def process_preds_and_labels(preds, labels):
    preds = preds.detach().cpu()
    labels = labels.detach().cpu()
    # replace all -100 with the token id of <pad>
    labels = torch.where(labels==-100, tokenizer.pad_token_id, labels)
    
    # decode all token ids to its string/text format
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # additional cleaning by removing begining and trailing spaces
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]
    
    return decoded_preds, decoded_labels


# In[17]:


def run_training(train_dl):
    model.train()
    for batch in tqdm(train_dl, total=len(train_dl)):
        opt.zero_grad()
        out = model(**batch)
        accelerator.backward(out.loss)
        opt.step()
        
def run_evaluation(test_dl):
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_dl, total=len(test_dl)):
            # generate predictions one by one
            preds = model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_length=max_length,
            )
            
            # convert target labels and predictions to string format for computing accuracy
            preds, labels = process_preds_and_labels(preds, batch['labels'])
            # add the target labels and predictions of this batch to seqeval
            metric.add_batch(predictions=preds, references=labels)


# In[22]:


epochs = 3

for epoch in range(epochs):
    run_training(train_dl)
    
    run_evaluation(test_dl)
    # calculate BLEU score on test set
    test_acc = metric.compute()['score']
    
    # save the model at the end of epoch
    torch.save(model.state_dict(), f"model-v{epoch}.pt")
    print(f"epoch: {epoch} test_acc: {test_acc}")


# ### Testing

# In[ ]:


import wandb

run = wandb.init()
artifact = run.use_artifact('bipin/machine-translation/model:v4', type='model')
artifact_dir = artifact.download()

model_file = "./artifacts/model:v4/model-v2.pt"
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
model.load_state_dict(torch.load(model_file, map_location='cpu'))
model.eval()


# In[19]:


split_datasets['test'][0]['translation']


# In[22]:


sample = split_datasets['test'][0]['translation']
inputs = sample['en']
label = sample['fr']

inputs = tokenizer(inputs, return_tensors='pt')
out = model.generate(**inputs)

# convert token ids to string
with tokenizer.as_target_tokenizer():
    decoded_out = tokenizer.decode(out[0], skip_special_tokens=True)

print("Label: \n", label)
print("Prediction: \n", decoded_out)


# In[ ]:




