#!/usr/bin/env python
# coding: utf-8

# In[1]:


from transformers import (
    AutoFeatureExtractor, 
    AutoTokenizer, 
    VisionEncoderDecoderModel,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer, 
    default_data_collator,
)

from torch.utils.data import Dataset

import pandas as pd
from sklearn.model_selection import train_test_split

from pathlib import Path
from PIL import Image


# In[2]:


df = pd.DataFrame(columns=['imgs'])
imgs, captions = [], []
root_dir = Path("../input/flickr8k")

with open(root_dir/"captions.txt", "r") as f:
    content = f.readlines()
    
for line in content:
    line = line.strip().split("|")
    if line[1]=='1':
        imgs.append(root_dir/"images"/line[0])
        captions.append(line[-1])
        
df.loc[:, 'imgs'] = imgs
df.loc[:, 'captions'] = captions


# In[3]:


df.head()


# In[4]:


encoder_checkpoint = "google/vit-base-patch16-224-in21k"
decoder_checkpoint = "gpt2"

feature_extractor = AutoFeatureExtractor.from_pretrained(encoder_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(decoder_checkpoint)
tokenizer.pad_token = tokenizer.eos_token


# In[5]:


# maximum length for the captions
max_length = 128
sample = df.iloc[0]

# sample image
image = Image.open(sample['imgs']).convert('RGB')
# sample caption
caption = sample['captions']

# apply feature extractor on the sample image
inputs = feature_extractor(images=image, return_tensors='pt')
# apply tokenizer
outputs = tokenizer(
            caption, 
            max_length=max_length, 
            truncation=True, 
            padding='max_length',
            return_tensors='pt',
        )


# In[ ]:


print(f"Inputs:\n{inputs}\nOutputs:\n{outputs}")


# In[6]:


class LoadDataset(Dataset):
    def __init__(self, df):
        self.images = df['imgs'].values
        self.captions = df['captions'].values
    
    def __getitem__(self, idx):
        # everything to return is stored inside this dict
        inputs = dict()

        # load the image and apply feature_extractor
        image_path = str(self.images[idx])
        image = Image.open(image_path).convert("RGB")
        image = feature_extractor(images=image, return_tensors='pt')

        # load the caption and apply tokenizer
        caption = self.captions[idx]
        labels = tokenizer(
            caption, 
            max_length=max_length, 
            truncation=True, 
            padding='max_length',
            return_tensors='pt',
        )['input_ids'][0]
        
        # store the inputs and labels in the dict we created
        inputs['pixel_values'] = image['pixel_values'].squeeze()   
        inputs['labels'] = labels
        return inputs
    
    def __len__(self):
        return len(self.images)


# In[7]:


train_df, test_df = train_test_split(df, test_size=0.2, shuffle=True, random_state=42)

train_ds = LoadDataset(train_df)
test_ds = LoadDataset(test_df)


# In[ ]:


next(iter(test_ds))


# In[8]:


model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
    encoder_checkpoint, 
    decoder_checkpoint
)
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id
# model.config.vocab_size = model.config.decoder.vocab_size
model.config.num_beams = 4


# In[ ]:


batch = next(iter(train_ds))

model(pixel_values=batch['pixel_values'].unsqueeze(0), labels=batch['labels'].unsqueeze(0))


# In[9]:


training_args = Seq2SeqTrainingArguments(
    output_dir="image-caption-generator", # name of the directory to store training outputs
    evaluation_strategy="epoch",          # evaluate after each epoch
    per_device_train_batch_size=8,        # batch size during training
    per_device_eval_batch_size=8,         # batch size during evaluation
    learning_rate=5e-5,
    weight_decay=0.01,                    # weight decay for AdamW optimizer
    num_train_epochs=5,                   # number of epochs to train
    save_strategy='epoch',                # save checkpoints after each epoch
    report_to='none',                     # prevents logging to wandb, mlflow...
)

trainer = Seq2SeqTrainer(
    model=model, 
    tokenizer=feature_extractor, 
    data_collator=default_data_collator,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    args=training_args,
)


# In[10]:


trainer.train()


# ### Inference

# In[11]:


import torch
import matplotlib.pyplot as plt


# In[26]:


inputs = test_ds[93]['pixel_values']

model.eval()
with torch.no_grad():
    # uncomment the below line if feature extractor is not applied to the image already
    # inputs = feature_extractor(images=inputs, return_tensors='pt').pixel_values

    # model prediction 
    out = model.generate(
        inputs.unsqueeze(0).to('cuda'), # move inputs to GPU
        num_beams=4, 
#         max_length=17
        )

# convert token ids to string format
decoded_out = tokenizer.decode(out[0], skip_special_tokens=True)

print(decoded_out)
plt.axis('off')
plt.imshow(torch.permute(inputs, (1, 2, 0)));


# In[ ]:




