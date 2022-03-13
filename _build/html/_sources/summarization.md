## Summarization

Most of the steps in this chapter will be familiar to you because this is almost similar to the previous chapter on machine translation. Instead of translation, we are summarizing the given input text.

The only major difference will be in the preparation of the dataset. We will train a model which will work with two languages - english and french. These types of models are called bilingual models. Our model will be able to summarize documents in english as well as french.

Now let's get straight into preparing our dataset.

### Dataset

We will be using the [amazon reviews dataset](https://huggingface.co/datasets/amazon_reviews_multi) which provide reviews in multiple languages, and from that we will download the english and french ones and combine them together into a single dataset.

#### Downloading the dataset

First let's download our datasets:

```python
from datasets import load_dataset

# name of dataset
ds = "amazon_reviews_multi"

# english reviews
english_dataset = load_dataset(ds, "en")
# french reviews
french_dataset = load_dataset(ds, "fr")
```

Let's see what's inside our english dataset by printing it.

```python
DatasetDict({
    train: Dataset({
        features: ['review_id', 'product_id', 'reviewer_id', 'stars', 'review_body', 'review_title', 'language', 'product_category'],
        num_rows: 200000
    })
    validation: Dataset({
        features: ['review_id', 'product_id', 'reviewer_id', 'stars', 'review_body', 'review_title', 'language', 'product_category'],
        num_rows: 5000
    })
    test: Dataset({
        features: ['review_id', 'product_id', 'reviewer_id', 'stars', 'review_body', 'review_title', 'language', 'product_category'],
        num_rows: 5000
    })
})
```

There is a train, validation and test set with 8 features in each of them. For this chapter, we only need ```review_body``` and ```review_title```. We will use ```review_body``` as our inputs and ```review_title``` as the summary.

#### Preprocessing the dataset

The training set itself is very huge, so we will filter out the reviews of a specific category from the ```product_category``` feature. Before that let's see the different product categories in our dataset.

```python
product_categories = english_dataset['train'][:]['product_category']

print(set(product_categories))
```
Output:
```python
{
    'apparel',
    'automotive',
    'baby_product',
    'beauty',
    'book',
    'camera',
    'digital_ebook_purchase',
    'digital_video_download',
    'drugstore',
    'electronics',
    'furniture',
    'grocery',
    'home',
    'home_improvement',
    'industrial_supplies',
    'jewelry',
    'kitchen',
    'lawn_and_garden',
    'luggage',
    'musical_instruments',
    'office_product',
    'other',
    'pc',
    'personal_care_appliances',
    'pet_products',
    'shoes',
    'sports',
    'toy',
    'video_games',
    'watch',
    'wireless'
 }
```

For the time being, let's filter out all reviews for the product category "kitchen". We will use the ```.filter()``` method for this:

```python
# select all reviews where the product category equal to 'kitchen'
english_dataset = english_dataset.filter(lambda x: x['product_category']=='kitchen')
french_dataset = french_dataset.filter(lambda x: x['product_category']=='kitchen')
```

Now let's combine our english and french reviews into a single dataset. We need to use the ```DatasetDict``` object to create our dataset as shown below:

```python
from dataset import DatasetDict

combined_dataset = DatasetDict()
```

Now we will concatenate english and french dataset, shuffle it and store it inside ```combined_dataset```:

```python
from dataset import concatenate_datasets

splits = ["train", "validation", "test"]

for split in splits:
    # concatenate english and french datasets
    combined_dataset[split] = concatenate_datasets([english_dataset[split], french_dataset[split]])
    # shuffle the concatenated dataset
    combined_dataset[split] = combined_dataset[split].shuffle(seed=42)
```

For better results, we will only take those samples where the length of review title is greater than 3:

```python
combined_dataset = combined_dataset.filter(lambda x: len(x['review_title']) > 3)
```

Now let's load the tokenizer and tokenize the dataset:

```python
from transformers import AutoTokenizer

checkpoint = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```

Our problem here is a sequence to sequence problem, so our model will have an encoder and a decoder. The input text is used by our encoder and the labels/outputs are used by the decoder. So while tokenizing input text, we could use the tokenizer as we use it normally as shown below:

```python
input_text = "This is the input text that is used by the encoder"

tokens = tokenizer(input_text)
```

But while tokenizing our output text or the labels(which is used by our decoder), we should tokenize it like this:

```python
output_text = "This is the output text that is used by the decoder"

with tokenizer.as_target_tokenizer():
    tokens = tokenizer(output_text)
```

Since we are using mT5(multilingual T5) model which is already trained in multiple language setting(which includes english and french), it will take care of tokenizing both english and french reviews without doing any modifications in the code we use normally.

```python
max_input_length = 512
max_output_length = 30

def tokenize(examples):
    inputs = tokenizer(examples['review_body'], max_length=max_input_length, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples['review_title'], max_length=max_output_length, truncation=True
            )
    inputs['labels'] = labels['input_ids']
    return inputs
```

Apply the above function on the whole dataset:
```python
tokenized_datasets = combined_dataset.map(
    tokenize, 
    batched=True, 
    remove_columns=combined_dataset['train'].column_names
)
```

#### Creating the dataloaders

Finally, let's create the dataloaders using the same data collator we used in the last chapter - ```DataCollatorForSeq2Seq```. As you know, we need to pass in the tokenizer as well as the model we are using to this collator, so let's load our model using ```AutoModelForSeq2SeqLM```:

```python
from transformers import DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
# collator
collate_fn = DataCollatorForSeq2Seq(tokenizer, model=model)
```

And here is the code to prepare our dataloaders:

```python
from torch.utils.data import DataLoader

batch_size = 16

# training dataloader
train_dl = DataLoader(
    tokenized_datasets['train'], 
    batch_size=batch_size, 
    shuffle=False, 
    collate_fn=collate_fn
)

# validation dataloader
val_dl = DataLoader(
    tokenized_datasets['validation'], 
    batch_size=batch_size, 
    shuffle=False, 
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

### Training the model

We've our dataloaders and model in place. Now let's write some code to train our model. This is almost similar to the one in translation chapter, the only difference is the metric used. Instead of BLEU score, we will use something called [ROUGE score](https://huggingface.co/metrics/rouge).

In short, the rouge score reports the harmonic mean of precision and recall, similar to what our f1-score does.

Here is a refresher on precision and recall:

* Precision - of the total number of words predicted, how many of them where correct/overlapping with the labels.
* Recall - of the total number of words in the labels, how many of them were predicted correctly.

So, let's first create the optimizer and move everything to GPU using accelerate:

```python
from torch import optim
from accelerate import Accelerator

opt = optim.AdamW(model.parameters(), lr=2e-5)

accelerator = Accelerator()
train_dl, val_dl, test_dl, model, opt = accelerator.prepare(train_dl, val_dl, test_dl, model, opt)
```

We will load the rouge metric and then write a function that converts the predicted token ids to tokens for calculating the metric.

```{note}
You may have to run ```pip install rouge_score``` before loading the metric.
```

```python
from datasets import load_metric

metric = load_metric('rouge')
```

The function that convert token ids to tokens does the following things:

1. Replace all -100 values in the labels(created by our collator) with the ```<pad>``` token id.
2. Convert tokens to token ids.
3. Do some additional processing by removing begining and trailing spaces in the tokens.
4. The metric we are using require each sentence in the summary to be separated by a new line, so we use NLTK's sentence tokenizer to split each summary(predicted as well as target summary) into a list of sentences and then join the by ```'\n'```.


```python
import torch
import nltk

def process_preds_and_labels(preds, labels):
    preds = preds.detach().cpu()
    labels = labels.detach().cpu()
    # replace all -100 with the token id of <pad>
    labels = torch.where(labels==-100, tokenizer.pad_token_id, labels)
    
    # decode all token ids to its string/text format
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # join sentences by "\n"
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    return decoded_preds, decoded_labels
```

Whoa, everything is set up. The only thing remaining is the training and evaluation loop, let's go ahead and finish it up:

```python
def run_training(train_dl):
    model.train()
    for batch in train_dl:
        opt.zero_grad()
        out = model(**batch)
        accelerator.backward(out.loss)
        opt.step()

def run_evaluation(test_dl):
    model.eval()
    with torch.no_grad():
        for batch in test_dl:
            # generate predictions one by one
            preds = model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_length=max_output_length,
            )
            
            # convert target labels and predictions to string format for computing ROUGE score
            preds, labels = process_preds_and_labels(preds, batch['labels'])
            # add the target labels and predictions of this batch to metrics
            metric.add_batch(predictions=preds, references=labels)
```

Let's train the model for 10 epochs:

```python
epochs = 10

for epoch in range(epochs):
    # training
    run_training(train_dl)

    # validation
    run_evaluation(val_dl)
    val_acc = metric.compute()
    # validation ROUGE score
    print(f"epoch: {epoch} val_acc: {val_acc}")

    # save the model at the end of epoch
    torch.save(model.state_dict(), f"model-v{epoch}.pt")
```

### Testing the model

Once that is finished, we test the model on the test set:

```python
run_evaluation(test_dl)

# ROUGE score on test set
test_acc = metric.compute()
print(f"test_acc: {test_acc}")
```