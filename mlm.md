## Masked language modelling

As the title suggests, this chapter is all about masked language modelling. The term may feel a bit alien to some of you, we will dig a bit deeper in this section to clear all your doubts about masked language modelling(MLM).

### What is masked language modelling?

As humans we adapt to a field like medicine by going through an extensive 5 year MBBS course and then we apply our skills, similarly we make our transformer model knowledgeable in a specific domain like medicine by pretraining it using **Masked Langauage Modelling(MLM)**, so that our model will perform better, say for example, on a classification task related to medical domain.

As you have an understanding of why masked language modelling is used, I will show you how it's done.

This is how the final inference will look like:

```{image} ./assets/mlm_inference.png
:alt: Maked language modelling
:class: bg-primary mb-1
:align: center
```

We input a sentence with some words replaced/masked with a special token ```[MASK]```. The job of the model is to predict the correct word to fill in place of ```[MASK]```. In the above figure the model predicts 'happiness' which makes input sentence ```'The goal of life is [MASK].'``` to ```'The goal of life is happiness.'```.

The training data will have a randomly masked sentence as inputs to the model and the complete sentence(without any masks) as the label, just like shown below:

Input: ```'The full cost of damage in [MASK] Stewart, one of the areas [MASK] affected, is still being [MASK].'```

Label: ```'The full cost of damage in Newton Stewart, one of the areas worst affected, is still being assessed.'```

The model will have to predict the correct words corresponding to the masks. In this way the model will learn about relationship between different words in a sentence. The task is some what similar to 'fill in the blanks' type questions that you might have encountered in your high school.

Now let's look into the dataset that we will be using for this task.

### Preparing the dataset

The dataset that we will be using is the [extreme summarization dataset](https://huggingface.co/datasets/xsum) which is a collection of news articles and their corresponding summaries. We will drop the 'summary' feature and use only the new articles.

Now let's look into the structure of our dataset.

#### Downloading the dataset

We will dounload the dataset from huggingface using their 'datasets' library.

```python
from datasets import load_dataset

# xsum -> eXtreme SuMmarization
raw_datasets = load_dataset('xsum')
print(raw_datasets)
```
Output:
```python
DatasetDict({
    train: Dataset({
        features: ['document', 'summary', 'id'],
        num_rows: 204045
    })
    validation: Dataset({
        features: ['document', 'summary', 'id'],
        num_rows: 11332
    })
    test: Dataset({
        features: ['document', 'summary', 'id'],
        num_rows: 11334
    })
})
```

As you can see from the above figure, we've a train, validation and test set. Of which, the training set is a huge one. So, for the sake of simplicity and for faster training, we will take a subset of the train set.

We will take 10,000 rows from the train set for training and 2000 rows for testing:

```python
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
```
Output:
```python
DatasetDict({
    train: Dataset({
        features: ['document', 'summary', 'id'],
        num_rows: 10000
    })
    test: Dataset({
        features: ['document', 'summary', 'id'],
        num_rows: 2000
    })
})
```

#### Preprocessing the dataset

Now let's prepare our dataset in a way that is needed for our model. Since our task is masked language modelling which is done to give domain knowledge to our model, we cannot afford to lose too much information from our input sentences due to truncation.

Since the maximum length of input sentence for the model we are using is 512, all the inputs that are longer than this will be truncated, which we don't want to happen.

So, we will concatenate all of the sentences in a batch and divide them into smaller chunks, and then each chunk will be the inputs to the model as shown in the figure below:

```{image} ./assets/mlm_preprocess.png
:align: center
```

The only difference is that, instead of strings it will be tokens. So we will write a function to tokenize the input sentences and then do the above mentioned steps on the tokenized outputs.

Let's load the tokenizer for our model:

```python
from transformers import AutoTokenizer

# model name
checkpoint = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```
We will be splitting our inputs into chunks of size 128. When splitting the inputs, the last chunk will be smaller than 128, so we will drop that for now.

```python
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
```

Let's try out our function on a the dataset:

```python
preprocessed_datasets = downsampled_datasets.map(
    create_chunks, 
    batched=True, 
    remove_columns=['document', 'summary', 'id']
)
```

Let's check the size of our inputs now:

```python
sample = preprocessed_datasets['train'][:5]

for i in sample['input_ids']:
    input_length = len(i)
    print(input_length) 
```
Output:
```python
128
128
128
128
128
```

As you can see, the size of every input is now 128.