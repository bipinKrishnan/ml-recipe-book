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

The only difference is that, instead of words it will be tokens. So we will write a function to tokenize the input sentences and then do the above mentioned steps on the tokenized outputs.

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

Now let's see how our inputs and labels look like,

```python
sample_inputs = sample['input_ids'][0]
sample_labels = sample['labels'][0]

# decode the tokens
print("INPUTS:\n", tokenizer.decode(sample_inputs))
print("\nLABELS:\n", tokenizer.decode(sample_labels))
```
Output:
```python
INPUTS:
 [CLS] media playback is not supported on this device varnish and james were third in the women's team sprint but the two men's squads failed to reach their respective medal ride - offs. the sprint team were fifth, while the pursuit quartet finished eighth. " we've had some problems, " said pursuit rider ed clancy. britain won the four - man pursuit event in 2012 and took silver in 2013. they also won gold at the 2008 and 2012 olympic games. but two - time olympic gold medallist clancy, sam harrison, owain doull and jon dibben finished eighth this time in four minutes 4. 419 seconds

LABELS:
 [CLS] media playback is not supported on this device varnish and james were third in the women's team sprint but the two men's squads failed to reach their respective medal ride - offs. the sprint team were fifth, while the pursuit quartet finished eighth. " we've had some problems, " said pursuit rider ed clancy. britain won the four - man pursuit event in 2012 and took silver in 2013. they also won gold at the 2008 and 2012 olympic games. but two - time olympic gold medallist clancy, sam harrison, owain doull and jon dibben finished eighth this time in four minutes 4. 419 seconds
```

Both of them looks the same, there are no masked words at all. But what we require is inputs with randomly masked words like this:

```This [MASK] is going to the park [MASK].```

and the corresponding labels with no masked words like below:

```This man is going to the park tomorrow.```

So, the only part that's remaining is randomly masking the inputs which can be done with 'DataCollatorForLanguageModeling' from transformers library just like this:

```python
from transformers import DataCollatorForLanguageModeling

collate_fn = DataCollatorForLanguageModeling(
    tokenizer, 
    mlm_probability=0.15
    )
```

We have set an additional parameter ```mlm_probability=0.15``` which means that each token has a 15% chance to be masked. We cannot pass all the inpputs directly to this ```collate_fn```, instead we need to put each example(containing ```input_ids```, ```attention_mask``` and ```labels```) into a list as shown below:

```python
# first 5 examples from train set
first_5_rows = preprocessed_datasets['train'][:5]
input_list = [dict(zip(first_5_rows, v)) for v in zip(*first_5_rows.values())]
```

```input_list``` is a list containing examples and will have a format like below:

```python
[
    {'input_ids': [...], 'attention_mask': [...], 'labels': [...]},
    {'input_ids': [...], 'attention_mask': [...], 'labels': [...]},
    {'input_ids': [...], 'attention_mask': [...], 'labels': [...]},
    {'input_ids': [...], 'attention_mask': [...], 'labels': [...]},
    {'input_ids': [...], 'attention_mask': [...], 'labels': [...]},
]
```

Now we can apply our collator on this list:

```python
collate_fn(input_list)
```

While applying the above function we will get a new set of masked ```input_ids``` and a new set of ```labels```. All the tokens in the labels except for the tokens corresponding to the mask will have a value of -100, which is a special number because it is ignored by our loss function :) So while calculating the loss, we will only consider the losses corresponding to the masked words and ignore others.

Here is an example to illustrate the same:

Suppose we have a set of tokens(converted to integers) like this: ```[23, 25, 100, 134, 78, 56]```

Once we pass the above inputs to our collator, we will get a randomly masked output(where 103 is the id corresponding to the mask): ```[23, 103, 100, 134, 103, 56]``` and the labels corresponding to the new inputs will be these: ```[-100, 25, -100, -100, 78, -100]``` where the real token ids are shown only for the masked tokens, for other it's just -100.

As we have our training dataset processed and our collator in place, we can create our training dataloader:

```python
from torch.utils.data import DataLoader

batch_size = 64

train_dl = DataLoader(
    preprocessed_datasets['train'], 
    batch_size=batch_size, 
    shuffle=True,
    collate_fn=collate_fn
    )
```

As we've said that our collator applies random masking each time we call it. But we need a fixed set with no variability during evaluation so that we have a fair comparison after each epoch. 

So, instead of directly using 'DataCollatorForLanguageModeling' directly in our test dataloader, we will wrap it in a function and apply to the test data using ```.map()``` method:

```python
def apply_random_mask(examples):
    example_list = [dict(zip(examples, v)) for v in zip(*examples.values())]
    output = collate_fn(example_list)
    # we need to return a dictionary
    return {k: v.numpy() for k, v in output.items()}

test_dataset = preprocessed_datasets['test'].map(
    apply_random_mask, 
    batched=True
    )
```

and then use the 'default_data_collator' from transformers library to collate our data for the test dataloader.

```python
from transformers import default_data_collator

test_dl = DataLoader(
    test_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    collate_fn=default_data_collator
    )
```
We've our training and testing dataloader in place, now it's time to train our model.

### Training the model