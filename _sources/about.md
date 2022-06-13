# About this book

### Brief idea about each section

This book is divided into 3 sections:

1. Natural language processing(NLP)
2. Computer vision(CV)
3. Image and text

Most of the chapters in NLP section was written when I was going through the [huggingface course](https://huggingface.co/course/). Huge shout out to them for the awesome course. The chapters in this section makes great use of the [transformers](https://github.com/huggingface/transformers), [datasets](https://github.com/huggingface/datasets) and [accelerate](https://github.com/huggingface/accelerate) libraries from huggingface. Here is a brief overview of each chapter in this section:

1. **Named entity recognition** - Discusses about training transformer model for recognizing named entities using [conllpp dataset](https://huggingface.co/datasets/conllpp). The specific model we will use is called [bert-base-cased](https://huggingface.co/bert-base-cased). The model is a smaller version of original BERT and is case sensitive, which means, it treats upper-cased and lower-cased letters as different.

2. **Masked language modelling** - Similar to fill in the blanks question, we train a model to predict the masked word in a sentence using [xsum dataset](https://huggingface.co/datasets/xsum). The specific model we will use is called [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased). This is a distilled version of the bert base uncased model which treats upper-cased and lower-cased letters in the same manner.

3. **Machine translation** - In this chapter, a model is trained to translate text from english to spanish. We will train a transformer model from Helsinki NLP group on the [news commentary dataset](https://huggingface.co/datasets/news_commentary). 

4. **Summarization** - In this chapter, a multi-lingual model is trained to summarize english and spanish sentences. The model used is a multi-lingual version of T5 transformer model and the dataset used is [amazon reviews dataset](https://huggingface.co/datasets/amazon_reviews_multi).

5. **Causal language modeling** - This chapter focuses on training a model to autocomplete python code. For this, we will use the data used to train [code parrot model](https://huggingface.co/lvwerra/codeparrot).

The computer vision section covers the most common tasks under this domain. The chapters in this section makes use of [pytorch lightning](https://github.com/PyTorchLightning/pytorch-lightning), [pytorch image models(timm)](https://github.com/rwightman/pytorch-image-models), [albumentations library](https://albumentations.ai/) and [weights and biases platform](https://wandb.ai/). Here is a brief overview of each chapter in this section:

1. **Image classification** - We will train a convolutional neural network(CNN) model to classify animal images. The CNN model we will be using is "resnet34" and the dataset used is the [animal images dataset](https://www.kaggle.com/datasets/lasaljaywardena/animal-images-dataset).

2. **Image segmentation** - This chapter focuses on training a model to segment roads in a given image. We will use a U-net model for this task.

3. **Object detection** - In this chapter we will focus on detecting cars in an image. We will predict the coordinates corresponding to the bounding box that encloses the cars in the image. For this task, we will use faster-rcnn model.

The final section contains a chapter that trains a model to generate a caption given an image. It will have a vision transformer as the encoder and gpt-2 model as the decoder. 

```{note}
1. Some chapters may point to topics that are described in the previous chapters. So, it is good to go through each chapter in order(as displayed in the sidebar of the book).

2. Notebooks containing code for each chapter can be found [here](https://github.com/bipinKrishnan/ml-powered-apps/tree/main/notebooks).

3. If you face any version issue with the libraries discussed in the book, you can refer to this [requirements file](https://github.com/bipinKrishnan/ml-powered-apps/blob/main/ml_book_final_requirements.txt) to get the list of libraries and corresponding versions used while writing this book.
```

### Why this book was written

This book can be useful in the following two ways:

1. If you are a person who has some experience building deep learning models for the task of classification(in NLP & computer vision) and you know that there exist other tasks that can be solved using deep learning models. But, you haven't had the chance to explore those areas. If so, this book will definitely help you in your journey.

2. This book can also be used as a reference book while you are training models for any of the task discussed in this book.

### Feedback and support

Everything in the book is discussed in the most simplest language possible. If you feel something is missing or can be improved, please reach out to me personally(via [twitter](https://twitter.com/bkrish_)/[linkedin](https://www.linkedin.com/in/bipin-krishnan)) or raise an issue in the [github repo](https://github.com/bipinKrishnan/ml-powered-apps/issues).

If you like what I'm doing, star🌟 the [repo](https://github.com/bipinKrishnan/ml-powered-apps) and show some love❤️. 



