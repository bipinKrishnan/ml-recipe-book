## Machine translation

Unlike other chapters that we've completed so far, this will be a bit more familiar to all of you guys. We don't need a separate section to discuss "What is machine translation?". It's as simple as saying that given an english sentence, our machine learning model translates it to another language, say, Spanish.

In the above example, the inputs to our model will be an english sentence and the label will be it's corresponding Spanish sentence.

Let's directly jump into the datasets that we are going to use.

### Preparing the dataset

#### Downloading the dataset

We will be using the [news commentary dataset](https://huggingface.co/datasets/news_commentary) for our task, and specifically we will be using the english to french translation subset.

We will retrieve the dataset by specifying the languages we require for our task(that is, english and french).

```python
from datasets import load_dataset

raw_datasets = load_dataset("news_commentary", lang1="en", lang2="fr")
print(raw_datasets)
```

From the dataset, we will use 50% for training and 10% for evaluation purpose.

```python
split_datasets = raw_datasets['train'].train_test_split(train_size=0.5, test_size=0.1, seed=42)
print(split_datasets)
```
Output:
```python
DatasetDict({
    train: Dataset({
        features: ['id', 'translation'],
        num_rows: 104739
    })
    test: Dataset({
        features: ['id', 'translation'],
        num_rows: 20948
    })
})
```

#### Preprocessing the dataset

The model we are going to use is already trained for translating english to french, we will finetune it for our news commentary dataset.

Since our inputs and labels are sentences, we need to tokenize both of them before using for training. 

We will load the tokenizer of our model as we did in other chapters:

```python
from transformers import AutoTokenizer

# model name
checkpoint = "Helsinki-NLP/opus-mt-en-fr"
# load the tokenizer for the model
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```

Let's take a sample containing english and french sentences from the training set for quick experiments:

```python
sample = split_datasets['train']['translation'][0]
print(sample)
```
Output:
```python
{
   'en': 'It is important to note that these were unintended consequences of basically sensible policy decisions.', 
   'fr': 'Il est important de noter qu’il s’agit là de conséquences non voulues de décisions politiques raisonnables au départ.'
}
```

```'en'```(english) part will be the inputs and ```'fr'```(french) part will be the labels for our model.

Let's tokenize our inputs,

```python
tokenizer(sample['en'])
```
Since our model is already trained for english to french translation, tokenizing the input english sentence is simple as that. But for our french sentences, we need to let the tokenizer know that we are passing the labels inorder to get the correct tokenized output as shown below:

```python
with tokenizer.as_target_tokenizer():
    french_tokens = tokenizer(sample['fr'])
```

If we use the tokenizer without specifying anything, it will tokenize the french sentence as if it were an english sentence.

Now let's wrap this inside a function and apply it to all the english-french sentences in our dataset. Apart from that we will truncate our sentences to a maximum length of 128:

```python
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
```

Let's apply the function to our train and test set:

```python
tokenized_datasets = split_datasets.map(
    tokenize, 
    batched=True, 
    remove_columns=['id', 'translation']
    )
```

#### Preparing the dataloader

Since this is a sequence to sequence task, we will be using ```DataCollatorForSeq2Seq``` as our collator, which requires both the tokenizer and the model, so we will load our pretrained model:

```python
from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
```

Now let's define our collator:

```python
from transformers import DataCollatorForSeq2Seq

collate_fn = DataCollatorForSeq2Seq(tokenizer, model=model)
```

We used this collator specifically because it adds some extra information inside the dataloader, specific to sequence to sequence tasks, like, ```decoder_input_ids``` which is used by the decoder part of the model during training.

Let's create our training and testing dataloader:

```python
from torch.utils.data import DataLoader

batch_size = 32
# training dataloader
train_dl = DataLoader(
    tokenized_datasets['train'], 
    batch_size=batch_size, 
    shuffle=True, 
    collate_fn=collate_fn
)

# test dataloader
test_dl = DataLoader(
    tokenized_datasets['test'], 
    batch_size=batch_size, 
    shuffle=False, 
    collate_fn=collate_fn
)
```

Let's see what information present inside the dataloaders:

```python
batch = next(iter(train_dl))
print(batch.keys())
```
Output:
```python
dict_keys(['input_ids', 'attention_mask', 'labels', 'decoder_input_ids'])
```

As you can see that, apart from ```input_ids```, ```attention_mask``` and ```labels```, we've one more key called ```decoder_input_ids``` which is added by our collator.






