#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install -q datasets accelerate seqeval transformers ')
get_ipython().system('pip install gensim==4.1.2')


# In[ ]:


from datasets import load_dataset, load_metric
from transformers import (
    AutoTokenizer, 
    DataCollatorForTokenClassification, 
    AutoModelForTokenClassification
)

from accelerate import Accelerator
import gensim.downloader as api
import gradio as gr

import torch
from torch.utils.data import DataLoader
from torch import optim


# In[ ]:


checkpoint = 'bert-base-cased'
raw_datasets = load_dataset("conllpp")


# In[ ]:


tokenizer = AutoTokenizer.from_pretrained(checkpoint)


# In[ ]:


train_row_1 = raw_datasets['train'][0]
inputs = tokenizer(train_row_1['tokens'], is_split_into_words=True)


# In[ ]:


labels = raw_datasets['train'].features['ner_tags'].feature.names
ids = range(len(labels))
id2label = dict(zip(ids, labels))

id2label


# In[ ]:


def align_tokens_and_labels(word_ids, labels):
    previous_word_id = None
    new_labels = []
    
    for word_id in word_ids:
        
        if word_id!=previous_word_id:
            label = -100 if word_id==None else labels[word_id]
        elif word_id==None:
            label = -100
        else:
            label = labels[word_id]
            if label%2==1:
                label += 1
                
        previous_word_id = word_id
        new_labels.append(label)
                
    return new_labels


# In[ ]:


ner_labels = train_row_1['ner_tags']
inputs = tokenizer(train_row_1['tokens'], is_split_into_words=True)
word_ids = inputs.word_ids()

align_tokens_and_labels(word_ids, ner_labels)


# In[ ]:


def prepare_inputs_and_labels(ds):
    inputs = tokenizer(ds['tokens'], truncation=True, padding=True, is_split_into_words=True)
    labels_batch = ds['ner_tags']
    
    new_labels = []
    for idx, labels in enumerate(labels_batch):
        word_ids = inputs.word_ids(idx)
        new_label = align_tokens_and_labels(word_ids, labels)
        new_labels.append(new_label)
        
    inputs['labels'] = new_labels
    return inputs

prepared_datasets = raw_datasets.map(
    prepare_inputs_and_labels, 
    batched=True, 
    remove_columns=raw_datasets['train'].column_names
)


# In[ ]:


batch_size = 16
collate_fn = DataCollatorForTokenClassification(tokenizer=tokenizer)

# training dataloader
train_dl = DataLoader(
    prepared_datasets['train'], 
    batch_size=batch_size, 
    shuffle=True, 
    collate_fn=collate_fn
    )

# validation dataloader
val_dl = DataLoader(
    prepared_datasets['validation'], 
    batch_size=batch_size, 
    shuffle=True, 
    collate_fn=collate_fn
    )

# test dataloader
test_dl = DataLoader(
    prepared_datasets['test'], 
    batch_size=batch_size, 
    shuffle=True, 
    collate_fn=collate_fn
    )


# In[ ]:


model = AutoModelForTokenClassification.from_pretrained(
    checkpoint, 
    num_labels=len(labels)
    )
opt = optim.AdamW(model.parameters(), lr=1.23e-4)


# In[ ]:


accelerator = Accelerator()
train_dl, val_dl, test_dl, model, opt = accelerator.prepare(
    train_dl, 
    val_dl, 
    test_dl, 
    model, 
    opt
)


# In[ ]:


metric = load_metric('seqeval')

targets = ['O', 'B-PER', 'I-PER', 'O', 'O', 'O']
predictions = ['O', 'B-PER', 'O', 'O', 'O', 'O']

metric.compute(predictions=[predictions], references=[targets])


# In[ ]:


def process_preds_and_labels(preds, targets):
    preds = preds.detach().cpu()
    preds = preds.argmax(dim=-1)
    targets = targets.detach().cpu()

    true_targets = [
        [labels[t.item()] for t in target if t!=-100] 
        for target in targets
        ]
    true_preds = [
        [labels[p.item()] for p, t in zip(pred, target) if t!=-100] 
        for pred, target in zip(preds, targets)
        ]

    return true_preds, true_targets


# In[ ]:


first_batch = next(iter(train_dl))

preds = model(**first_batch)
process_preds_and_labels(preds.logits, first_batch['labels'])


# In[ ]:


def run_training_loop(train_dl):
    model.train()
    for batch in tqdm(train_dl, total=len(train_dl)):
        opt.zero_grad()
        out = model(**batch)
        accelerator.backward(out.loss)
        opt.step()
        
        # convert target labels and predictions to string format for computing accuracy
        preds, labels = process_preds_and_labels(out.logits, batch['labels'])
        # add the target labels and predictions of this batch to seqeval
        metric.add_batch(predictions=preds, references=labels)

def run_evaluation_loop(test_dl):
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_dl, total=len(test_dl)):
            out = model(**batch)
            
            # convert target labels and predictions to string format for computing accuracy
            preds, labels = process_preds_and_labels(out.logits, batch['labels'])
            # add the target labels and predictions of this batch to seqeval
            metric.add_batch(predictions=preds, references=labels) 


# In[ ]:


epochs = 3

for epoch in range(epochs):
    run_training_loop(train_dl)
    train_acc = metric.compute()['overall_accuracy']
    
    run_evaluation_loop(val_dl)
    val_acc = metric.compute()['overall_accuracy']
    
    print(f"epoch: {epoch} train_acc: {train_acc} val_acc: {val_acc}")
    
    # save the model at the end of epoch
    torch.save(model.state_dict(), f"model-v{epoch}.pt")


# In[ ]:


run_evaluation_loop(test_dl)
metric.compute()


# In[ ]:


def prepare_output(text):
  text_split = text.split()
  tokens = tokenizer(text_split, is_split_into_words=True, truncation=True, return_tensors='pt')
  preds = model(**tokens)['logits'].argmax(dim=-1)

  out = {}
  last_b_tag = ""
  for p, w_id in zip(preds[0], tokens.word_ids()):
    if w_id!=None:
      label = labels[p]
      label_split = label.split('-')
      word = text_split[w_id]
      
      if word not in out.keys():
        if label_split[0]=='I' and label_split[-1]==last_b_tag.split('-')[-1]:
          old_key = list(out.keys())[-1]
          new_key = old_key+f" {word}"
          out.pop(old_key)
          out[new_key] = last_b_tag
        else:
          out[word] = label
          
        if (label_split[0]=='B') and (label_split[-1] in ['ORG', 'LOC']):
          last_b_tag = label

  out_text = ""
  for word, tag in out.items():
    if tag.split('-')[-1] in ['PER', 'LOC', 'ORG']:
      try:
        word = word2vec.most_similar(positive=['India', word.replace(' ', '_')], negative=['USA'], topn=1)[0][0]
      except KeyError:
        pass
    out_text += f"{word.replace('_', ' ')} "
  return out_text


# In[ ]:


prepare_output("My name is Sarah and I work at San Francisco")


# In[ ]:


interface = gr.Interface(
    prepare_output,
    inputs=gr.inputs.Textbox(label="Input text", lines=3),
    outputs=gr.outputs.Textbox(label="Output text"),
)

# launch the demo
interface.launch()

