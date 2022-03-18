#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install -q datasets')


# In[2]:


from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    GPT2LMHeadModel, 
    AutoConfig, 
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


# In[17]:


complete_ds = load_dataset("huggingface-course/codeparrot-ds-train")


# In[18]:


raw_datasets = complete_ds['train'].train_test_split(train_size=0.1, test_size=0.01, seed=42)


# In[19]:


raw_datasets


# In[20]:


print(raw_datasets['train'][0]['content'][:500])


# In[21]:


tokenizer = AutoTokenizer.from_pretrained("huggingface-course/code-search-net-tokenizer")


# In[22]:


sample = raw_datasets['train']['content'][:2]

o = tokenizer(
        sample, 
        truncation=True, 
        max_length=128,
        return_overflowing_tokens=True,
        return_length=True,
    )


# In[23]:


max_length = 128

def tokenize(examples):
    outputs = tokenizer(
        examples['content'], 
        truncation=True, 
        max_length=max_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    
    for input_ids, length in zip(outputs['input_ids'], outputs['length']):
        if length==max_length:
            input_batch.append(input_ids)
            
    return {"input_ids": input_batch}


# In[ ]:


tokenized_datasets = raw_datasets.map(
    tokenize, 
    batched=True, 
    remove_columns=raw_datasets['train'].column_names
)


# In[ ]:


sample_data = DatasetDict(
    {
        "train": tokenized_datasets['train'].shuffle(seed=42).select(range(30)),
        "test": tokenized_datasets['test'].shuffle(seed=42).select(range(20))
    }
)


# In[7]:


config = AutoConfig.from_pretrained(
    "gpt2", 
    vocab_size=len(tokenizer),
    n_ctx=max_length,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
model = GPT2LMHeadModel(config)


# In[27]:


tokenizer.pad_token = tokenizer.eos_token
collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)


# In[ ]:


a = [dict(zip(o, v)) for v in zip(*o.values())]
tokenizer.convert_ids_to_tokens(collator(a)['input_ids'][0])


# In[ ]:


a = collator([tokenized_datasets["train"][i] for i in range(5)])


# In[ ]:


for key in a:
    print(f"{key} shape: {a[key].shape}")


# In[ ]:


args = TrainingArguments(
    output_dir="model_outputs",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    evaluation_strategy="epoch",
    gradient_accumulation_steps=8,
    num_train_epochs=1, 
    weight_decay=0.01,
    learning_rate=5e-4,
    fp16=True,
    run_name="casual-lm",
    report_to="wandb",
)


# In[ ]:


trainer = Trainer(
    model=model, 
    tokenizer=tokenizer, 
    args=args, 
    data_collator=collator, 
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
)


# In[ ]:


trainer.train()


# In[ ]:


txt = """
# import random forest regressor from scikit-learn
from sklearn.ensemble import RandomForestRegressor

# fit random forest model with 300 estimators on X, y:
"""

inputs = tokenizer(txt, return_tensors='pt')
inputs = inputs.to('cuda')

out = trainer.model.generate(**inputs, max_length=130)
print(tokenizer.decode(out[0]))


# #### Output:
# 
# ```python
# # import random forest regressor from scikit-learn
# from sklearn.ensemble import RandomForestRegressor
# 
# # fit random forest model with 300 estimators on X, y:
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=0)
# 
# # Fit classifier with all parameters
# classifier = RandomForestRegressor(n_estimators=300, max_depth=3, n_estimators=100, random_state=0)
# 
# classifier.fit(X_train, y_train)
# ```

# In[ ]:




