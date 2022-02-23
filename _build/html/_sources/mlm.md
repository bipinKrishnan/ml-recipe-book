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