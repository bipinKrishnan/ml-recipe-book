## Summarization

Most of the steps in this chapter will be familiar to you as this is almost similar to the previous chapter on machine translation. Instead of translation, we are summarizing the given input text.

The only major difference will be in the preparation of the dataset. We will train a model which will work with two languages - english and french. These types of models are called bilingual models. Our model will be able to summarize documents in english as well as french.

Now let's get straight into preparing our dataset.

### Dataset

We will be using the [amazon reviews dataset](https://huggingface.co/datasets/amazon_reviews_multi) which provide reviews in multiple languages, and from that we will download the english and french ones and join them together into a single dataset.

#### Downloading the dataset

First let's download the reviews in both languages:

```python
from datasets import load_dataset

# name of dataset
ds = "amazon_reviews_multi"

# english reviews
english_dataset = load_dataset(ds, "en")
# french reviews
french_dataset = load_dataset(ds, "fr")
```

This is what you get when print ```english_dataset```(the same happens for french dataset as well):

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

There is a train, validation and test set as well as a lot of features. We will only use the ```review_body``` and ```review_title``` for this chapter. We will take ```review_body``` as our inputs and ```review_title``` as it's summary.

#### Preprocessing the dataset

The training set itself is very huge, so we will filter out the reviews of a specific category product using the ```product_category``` feature. Before that let's see the different product categories in our dataset.

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
# filter all reviews that has the 'product_category' equal to 'kitchen'
english_dataset = english_dataset.filter(lambda x: x['product_category']=='kitchen')
french_dataset = french_dataset.filter(lambda x: x['product_category']=='kitchen')
```

The product category will be in english itself even for our french dataset, only the review body and title will be in that specific language.

Now let's combine our english and french reviews into a single dataset. We need to use the ```DatasetDict``` object create our dataset as shown below:

```python
from dataset import DatasetDict

combined_dataset = DatasetDict()
```

Now we will concatenate english and french dataset, shuffle it and put it into ```combined_dataset``` as shown below:

```python
from dataset import concatenate_datasets

splits = ["train", "validation", "test"]

for split in splits:
    # concatenate english and french datasets
    combined_dataset[split] = concatenate_datasets(
        [english_dataset[split], french_dataset[split]]
        )
    # shuffle th final dataset
    combined_dataset[split] = combined_dataset[split].shuffle(seed=42)
```

For better results, we will only take those samples whose review title is greater than 3 using the same ```.filter()``` method.

```python
combined_dataset = combined_dataset.filter(lambda x: len(x['review_title']) > 3)
```

Now let's load the tokenizer and tokenize the dataset:

```python
from transformers import AutoTokenizer

checkpoint = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```

When dealing with text to text problems like this, we have to always use ```tokenizer.as_target_tokenizer()``` while tokenizing target sentences. According to the models used in the encoder and decoder of our model, there may be difference in the tokenization approach for both of them but using ```.as_target_tokenizer()``` takes care of it for us.

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

Apply the above function and remove all other unwanted columns from the dataset:
```python
tokenized_datasets = combined_dataset.map(
    tokenize, 
    batched=True, 
    remove_columns=combined_dataset['train'].column_names
)
```

#### Creating the dataloaders

Finally, let's create the dataloaders using the same collator we used in the last chapter, 'DataCollatorForSeq2Seq'. As you know, we need to pass in the tokenizer as well as the model we are using to this collator(for creating the ```decoder_input_ids```), so let's load our model using 'AutoModelForSeq2SeqLM':

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