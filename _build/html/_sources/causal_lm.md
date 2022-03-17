## Causal language modeling

Causal language modeling is nothing but predicting the next token given a sequence of text. Here is an example showing how causal language modeling works:

If you give an input text like this: ```'I am going'``` and you specify that you want the model to predict the next 2 tokens, the output will be like this - ```'I am going to Mumbai'```.

You can increase the number of tokens to be predicted as per your needs.

In this chapter we will not be training a model for completing sentences but code. Yes, you read it right, we are going to train a GPT-2 model from scratch for code completion.

When we give a partial code snippet, our model will autocomplete it.

### Dataset

We will be using the stripped down version of the dataset used to train the 'code parrot' model. You can view the dataset by going [here](https://huggingface.co/datasets/huggingface-course/codeparrot-ds-train).

We will strip down it further because for training it using openly available platforms like kaggle and Google colab. If you have more compute than what these platforms provide, then you can definitely go ahead with the complete dataset for training the model.

We will download this dataset and use 0.1% and 0.01% of the whole dataset for training and evaluation respectively.

```python
from datasets import load_dataset

complete_ds = load_dataset("huggingface-course/codeparrot-ds-train")
# further strip down of the dataset
raw_datasets = complete_ds['train'].train_test_split(train_size=0.1, test_size=0.01, seed=42)
```

```{note}
This dataset only contain python code related to machine learning libraries like pandas, scikit-learn, matplotlib and seaborn. Hence, this model will work best for code snippets related to these libraries.
```

Now let's load in our tokenizer. We cannot use normal tokenizers that are used for tokenizing natural languages like english because our dataset contain python code. So, we will use a tokenizer that was trained to tokenize python code:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("huggingface-course/code-search-net-tokenizer")
```

We will write a function to tokenize our dataset. If we truncate our dataset, we will loose a lot of information, instead, we will combine the rows in our dataset and divide it into chunks of length 128(as we did in the masked language modelling chapter).

So after the preprocessing the dataset, each row will have a length of 128.

Let's take a sample of rows from our dataset and see the outputs after tokenization:

```python
# code snippets sample
sample = raw_datasets['train']['content'][0]

tokenizer(
        sample, 
        truncation=True, 
        max_length=128,
        return_overflowing_tokens=True,
        return_length=True,
    )
```

Setting ```return_overflowing_tokens``` to ```True``` will split the sample into chunks. And we also return the sequence length for each chunk by setting ```return_length``` to ```True```. These are the keys in the ouput from the tokenizer: ```['input_ids', 'attention_mask', 'length', 'overflow_to_sample_mapping']```

The ```'length'``` keys contains the length of each chunk and ```'overflow_to_sample_mapping'``` keys contains the sample or row to which the chunk belongs to. For example, if the first row in the dataset is split into 5 chunks of size 128, each chunk will have ```'overflow_to_sample_mapping'``` equal to 0(index of first row).

Now let's write the function to tokenize the whole dataset:

```python
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
```

As you can see, we only take chunks whose length is equal to ```max_length```, i.e, 128. The rest of the chunks are dropped.

Another thing is that, we only return the ```input_ids```, that is because we will be using a data collator which will create the labels from these ```input_ids```.

Now let's apply the tokenization function to the whole dataset:

```python
tokenized_datasets = raw_datasets.map(
    tokenize, 
    batched=True, 
    remove_columns=raw_datasets['train'].column_names
)
```


