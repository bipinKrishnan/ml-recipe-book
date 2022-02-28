#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip install -q datasets accelerate')


# In[4]:


from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    DataCollatorForLanguageModeling,
    default_data_collator,
    AutoModelForMaskedLM,
)

from accelerate import Accelerator
import math
from tqdm.auto import tqdm

import torch
from torch import optim
from torch.utils.data import DataLoader


# In[5]:


raw_datasets = load_dataset('xsum')
raw_datasets


# In[6]:


train_size = 10000
test_size = int(0.2*train_size)
seed = 42

# 10,000 rows for training and 2000 rows for testing
downsampled_datasets = raw_datasets['train'].train_test_split(
    train_size=train_size,
    test_size=test_size,
    seed=seed,
)

print(downsampled_datasets)


# In[7]:


checkpoint = 'distilbert-base-uncased'

tokenizer = AutoTokenizer.from_pretrained(checkpoint)


# In[8]:


chunk_size = 128

def create_chunks(examples):
    # tokenize the inputs
    inputs = tokenizer(examples['document'])
    # cocatenate the inputs
    concatenated_examples = {k: sum(v, []) for k, v in inputs.items()}
    total_len = (len(concatenated_examples['input_ids'])//chunk_size)*chunk_size
    
    # create chunks of size 128
    results = {
        k: [v[i: (i+chunk_size)] for i in range(0, total_len, chunk_size)] 
        for k, v in concatenated_examples.items()
        }
    
    results['labels'] = results['input_ids'].copy()
    return results


# In[9]:


preprocessed_datasets = downsampled_datasets.map(
    create_chunks, 
    batched=True, 
    remove_columns=['document', 'summary', 'id']
)


# In[10]:


sample = preprocessed_datasets['train'][:5]

for i in sample['input_ids']:
    input_length = len(i)
    
    print(input_length)


# In[11]:


sample_inputs = sample['input_ids'][0]
sample_labels = sample['labels'][0]

# decode the tokens
print("INPUTS:\n", tokenizer.decode(sample_inputs))
print("\nLABELS:\n", tokenizer.decode(sample_labels))


# In[12]:


collate_fn = DataCollatorForLanguageModeling(
    tokenizer, 
    mlm_probability=0.15
    )


# In[13]:


for i in zip(sample):
    print(i)


# In[14]:


# first 5 examples from train set
first_5_rows = preprocessed_datasets['train'][:5]
input_list = [dict(zip(first_5_rows, v)) for v in zip(*first_5_rows.values())]


# In[ ]:


collate_fn(input_list)


# In[16]:


batch_size = 64

train_dl = DataLoader(
    preprocessed_datasets['train'], 
    batch_size=batch_size, 
    shuffle=True,
    collate_fn=collate_fn
    )


# In[17]:


next(iter(train_dl))['input_ids'].shape


# In[18]:


def apply_random_mask(examples):
    example_list = [dict(zip(examples, v)) for v in zip(*examples.values())]
    output = collate_fn(example_list)
    # we need to return a dictionary
    return {k: v.numpy() for k, v in output.items()}

test_dataset = preprocessed_datasets['test'].map(apply_random_mask, batched=True)


# In[19]:


test_dl = DataLoader(
    test_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    collate_fn=default_data_collator
    )


# In[20]:


tokenizer.decode(next(iter(test_dl))['input_ids'][0])


# In[28]:


model = AutoModelForMaskedLM.from_pretrained(checkpoint)
opt = optim.AdamW(model.parameters(), lr=1.23e-5)

accelerator = Accelerator()
train_dl, test_dl, model, opt = accelerator.prepare(train_dl, test_dl, model, opt)


# In[29]:


def run_training_loop(train_dl):
    losses = 0
    model.train()
    for batch in tqdm(train_dl, total=len(train_dl)):
        opt.zero_grad()
        out = model(**batch)
        accelerator.backward(out.loss)
        opt.step()

        losses += out.loss.item()
#         break
    losses /= len(train_dl)
    # exponential of cross entropy
    perplexity = math.exp(losses)
    return perplexity

def run_evaluation_loop(test_dl):
    losses = 0
    model.eval()
    
    with torch.no_grad():
        for batch in tqdm(test_dl, total=len(test_dl)):
            out = model(**batch)
            losses += out.loss.item()
#             break
            
    losses /= len(test_dl)
    # exponential of cross entropy
    perplexity = math.exp(losses)
    return perplexity


# In[30]:


epochs = 3

for epoch in range(epochs):
    train_perplexity = run_training_loop(train_dl)
    test_perplexity = run_evaluation_loop(test_dl)
    
    print(f"epoch: {epoch} train_acc: {train_perplexity} val_acc: {test_perplexity}")
    
    # save the model at the end of epoch
    torch.save(model.state_dict(), f"model-v{epoch}.pt")


# ### Test example

# In[35]:


text = """
Rajesh Shah, one of the shop's co-owners, told the [MASK] 
there would be a new name.
"""

# tokenize the inputs
inputs = tokenizer(text, return_tensors='pt')
inputs = inputs.to(accelerator.device)
out = model(**inputs)

# find the position in input where [MASK] is present
mask_token_id = tokenizer.mask_token_id
mask_idx = torch.where(inputs['input_ids']==mask_token_id)[1]

# decode the prediction corresponding to [MASK]
preds = out.logits.argmax(dim=-1)[0]
mask_pred = tokenizer.decode(preds[mask_idx])

# replace [MASK] with predicted word
final_text = text.replace('[MASK]', mask_pred)
print(final_text)


# In[ ]:




