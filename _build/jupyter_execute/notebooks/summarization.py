#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install -q datasets accelerate wandb rouge_score')


# In[ ]:


from datasets import load_dataset, DatasetDict, concatenate_datasets, load_metric
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM

import torch
from torch.utils.data import DataLoader

from accelerate import Accelerator
import nltk
import wandb
from tqdm.auto import tqdm


# In[ ]:


english_dataset = load_dataset("amazon_reviews_multi", "en")
french_dataset = load_dataset("amazon_reviews_multi", "fr")


# In[ ]:


english_dataset.set_format('pandas')


# In[ ]:


english_dataset['train'][:]['product_category'].value_counts()


# In[ ]:


english_dataset.reset_format()


# In[ ]:


def filter_reviews(examples):
    return examples['product_category']=='kitchen'

english_dataset = english_dataset.filter(filter_reviews)
french_dataset = french_dataset.filter(filter_reviews)


# In[ ]:


english_dataset['train'].shuffle(seed=42)[:3]


# In[ ]:


checkpoint = "google/mt5-small"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)


# In[ ]:


final_dataset = DatasetDict()

for split in english_dataset.keys():
    final_dataset[split] = concatenate_datasets([english_dataset[split], french_dataset[split]])
    final_dataset[split] = final_dataset[split].shuffle(seed=42)


# In[ ]:


final_dataset['train'][:3]


# In[ ]:


final_dataset = final_dataset.filter(lambda x: len(x['review_title']) > 3)


# In[ ]:


max_input_length = 512
max_output_length = 30

def tokenize(examples):
    inputs = tokenizer(examples['review_body'], truncation=True, max_length=max_input_length)
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['review_title'], truncation=True, max_length=max_output_length)
        
    inputs['labels'] = labels['input_ids']
    return inputs


# In[ ]:


tokenized_datasets = final_dataset.map(
    tokenize, 
    batched=True, 
    remove_columns=final_dataset['train'].column_names
)


# In[ ]:


bs = 8

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
collate_fn = DataCollatorForSeq2Seq(tokenizer, model=model)

train_dl = DataLoader(
    tokenized_datasets['train'], 
    batch_size=bs, 
    shuffle=False, collate_fn=collate_fn
)
val_dl = DataLoader(
    tokenized_datasets['validation'], 
    batch_size=bs, 
    shuffle=False, 
    collate_fn=collate_fn
)
test_dl = DataLoader(
    tokenized_datasets['test'], 
    batch_size=bs, 
    shuffle=False, 
    collate_fn=collate_fn
)


# In[ ]:


opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
metric = load_metric('rouge')

accelerator = Accelerator()
train_dl, val_dl, test_dl, model, opt = accelerator.prepare(
    train_dl, val_dl, test_dl, model, opt
)


# In[ ]:


generated_summary = "I absolutely loved reading the Hunger Games.\n I absolutely loved reading the Hunger Games"
reference_summary = "I loved reading the Hunger Games.\n I loved reading the Hunger Games"

scores = metric.compute(
    predictions=[generated_summary], references=[reference_summary]
)
scores


# In[ ]:


def process_preds_and_labels(preds, labels):
    preds = preds.detach().cpu()
    labels = labels.detach().cpu()
    # replace all -100 with the token id of <pad>
    labels = torch.where(labels==-100, tokenizer.pad_token_id, labels)
    
    # decode all token ids to its string/text format
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    return decoded_preds, decoded_labels


# In[ ]:


def run_training(train_dl):
    model.train()
    for batch in tqdm(train_dl, total=len(train_dl)):
        opt.zero_grad()
        out = model(**batch)
        accelerator.backward(out.loss)
        opt.step()
        
def run_evaluation(test_dl):
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_dl, total=len(test_dl)):
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


# In[ ]:


def save_model(epoch):
    model_path = f"model-v{epoch+1}.pt"
    torch.save(model.state_dict(), model_path)
    artifact = wandb.Artifact("model", type="model")
    artifact.add_file(model_path)
    run.log_artifact(artifact)


# In[ ]:


epochs = 10

run = wandb.init(
    project="summarization", 
    group='training', 
    entity="bipin", 
    name="run-lr-1e-2-rerun", 
    reinit=True
)

with run:
    for epoch in range(epochs):
        run_training(train_dl)
        
        run_evaluation(test_dl)
        # calculate ROUGE score on test set
        test_acc = metric.compute()
        print(f"epoch: {epoch} test_acc: {test_acc}")
    
        save_model(epoch)


# ### LRFinder

# In[ ]:


import copy
import os
import torch
import numpy as np
from tqdm.autonotebook import tqdm
from torch.optim.lr_scheduler import _LRScheduler
import matplotlib.pyplot as plt


# In[ ]:


class LRFinder(object):
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device=None,
    ):
        # Check if the optimizer is already attached to a scheduler
        self.optimizer = optimizer
        self._check_for_scheduler()

        self.model = model
        self.criterion = criterion
        self.history = {"lr": [], "loss": []}
        self.best_loss = None

        # If device is None, use the same as the model
        self.accelerator = Accelerator()
        self.device = self.accelerator.device

    def range_test(
        self,
        train_iter,
        val_iter=None,
        start_lr=None,
        end_lr=10,
        num_iter=100,
        step_mode="exp",
        smooth_f=0.05,
        diverge_th=5,
        accumulation_steps=1,
        non_blocking_transfer=True,
    ):
        # Reset test results
        self.history = {"lr": [], "loss": []}
        self.best_loss = None

        # Move the model to the proper device
        self.model.to(self.device)

        # Check if the optimizer is already attached to a scheduler
        self._check_for_scheduler()

        # Set the starting learning rate
        if start_lr:
            self._set_learning_rate(start_lr)

        # Initialize the proper learning rate policy
        if step_mode.lower() == "exp":
            lr_schedule = ExponentialLR(self.optimizer, end_lr, num_iter)
        elif step_mode.lower() == "linear":
            lr_schedule = LinearLR(self.optimizer, end_lr, num_iter)
        else:
            raise ValueError("expected one of (exp, linear), got {}".format(step_mode))

        if smooth_f < 0 or smooth_f >= 1:
            raise ValueError("smooth_f is outside the range [0, 1[")

        for iteration in tqdm(range(num_iter)):
            # Train on batch and retrieve loss
            loss = self._train_batch(
                train_iter,
                accumulation_steps,
                non_blocking_transfer=non_blocking_transfer,
            )
            if val_iter:
                loss = self._validate(
                    val_iter, non_blocking_transfer=non_blocking_transfer
                )

            # Update the learning rate
            self.history["lr"].append(lr_schedule.get_lr()[0])
            lr_schedule.step()

            # Track the best loss and smooth it if smooth_f is specified
            if iteration == 0:
                self.best_loss = loss
            else:
                if smooth_f > 0:
                    loss = smooth_f * loss + (1 - smooth_f) * self.history["loss"][-1]
                if loss < self.best_loss:
                    self.best_loss = loss

            # Check if the loss has diverged; if it has, stop the test
            self.history["loss"].append(loss)
            if loss > diverge_th * self.best_loss:
                print("Stopping early, the loss has diverged")
                break

        print("Learning rate search finished. See the graph with {finder_name}.plot()")

    def _set_learning_rate(self, new_lrs):
        if not isinstance(new_lrs, list):
            new_lrs = [new_lrs] * len(self.optimizer.param_groups)
        if len(new_lrs) != len(self.optimizer.param_groups):
            raise ValueError(
                "Length of `new_lrs` is not equal to the number of parameter groups "
                + "in the given optimizer"
            )

        for param_group, new_lr in zip(self.optimizer.param_groups, new_lrs):
            param_group["lr"] = new_lr

    def _check_for_scheduler(self):
        for param_group in self.optimizer.param_groups:
            if "initial_lr" in param_group:
                raise RuntimeError("Optimizer already has a scheduler attached to it")

    def _train_batch(self, train_iter, accumulation_steps, non_blocking_transfer=True):
        self.model.train()
        total_loss = None  # for late initialization
        train_iter = self.accelerator.prepare(train_iter)

        self.optimizer.zero_grad()
        for i in range(accumulation_steps):
            train_iter = next(iter(train_iter))
            inputs = train_iter

            # Forward pass
            model_outputs = self.model(**inputs)
            loss, outputs = model_outputs.loss, model_outputs.logits

            # Loss should be averaged in each step
            loss /= accumulation_steps

            # Backward pass
            self.accelerator.backward(loss)

            if total_loss is None:
                total_loss = loss
            else:
                total_loss += loss

        self.optimizer.step()

        return total_loss.item()

    def _move_to_device(self, inputs, labels, non_blocking=True):
        def move(obj, device, non_blocking=True):
            if hasattr(obj, "to"):
                return obj.to(device, non_blocking=non_blocking)
            elif isinstance(obj, tuple):
                return tuple(move(o, device, non_blocking) for o in obj)
            elif isinstance(obj, list):
                return [move(o, device, non_blocking) for o in obj]
            elif isinstance(obj, dict):
                return {k: move(o, device, non_blocking) for k, o in obj.items()}
            else:
                return obj

        inputs = move(inputs, self.device, non_blocking=non_blocking)
        labels = move(labels, self.device, non_blocking=non_blocking)
        return inputs, labels

    def _validate(self, val_iter, non_blocking_transfer=True):
        # Set model to evaluation mode and disable gradient computation
        running_loss = 0
        self.model.eval()
        val_iter = self.accelerator.prepare(val_iter)
        
        with torch.no_grad():
            for batch in val_iter:
                # Move data to the correct device
                inputs, labels = batch, batch['labels']

                # Forward pass and loss computation
                model_outputs = self.model(**inputs)
                loss, outputs = model_outputs.loss, model_outputs.logits
                running_loss += loss.item() * len(labels)

        return running_loss / len(val_iter.dataset)

    def plot(
        self,
        skip_start=10,
        skip_end=5,
        log_lr=True,
        show_lr=None,
        ax=None,
        suggest_lr=True,
    ):
        """Plots the learning rate range test.
        Arguments:
            skip_start (int, optional): number of batches to trim from the start.
                Default: 10.
            skip_end (int, optional): number of batches to trim from the start.
                Default: 5.
            log_lr (bool, optional): True to plot the learning rate in a logarithmic
                scale; otherwise, plotted in a linear scale. Default: True.
            show_lr (float, optional): if set, adds a vertical line to visualize the
                specified learning rate. Default: None.
            ax (matplotlib.axes.Axes, optional): the plot is created in the specified
                matplotlib axes object and the figure is not be shown. If `None`, then
                the figure and axes object are created in this method and the figure is
                shown . Default: None.
            suggest_lr (bool, optional): suggest a learning rate by
                - 'steepest': the point with steepest gradient (minimal gradient)
                you can use that point as a first guess for an LR. Default: True.
        Returns:
            The matplotlib.axes.Axes object that contains the plot,
            and the suggested learning rate (if set suggest_lr=True).
        """

        if skip_start < 0:
            raise ValueError("skip_start cannot be negative")
        if skip_end < 0:
            raise ValueError("skip_end cannot be negative")
        if show_lr is not None and not isinstance(show_lr, float):
            raise ValueError("show_lr must be float")

        # Get the data to plot from the history dictionary. Also, handle skip_end=0
        # properly so the behaviour is the expected
        lrs = self.history["lr"]
        losses = self.history["loss"]
        if skip_end == 0:
            lrs = lrs[skip_start:]
            losses = losses[skip_start:]
        else:
            lrs = lrs[skip_start:-skip_end]
            losses = losses[skip_start:-skip_end]

        # Create the figure and axes object if axes was not already given
        fig = None
        if ax is None:
            fig, ax = plt.subplots()

        # Plot loss as a function of the learning rate
        ax.plot(lrs, losses)

        # Plot the suggested LR
        if suggest_lr:
            # 'steepest': the point with steepest gradient (minimal gradient)
            print("LR suggestion: steepest gradient")
            min_grad_idx = None
            try:
                min_grad_idx = (np.gradient(np.array(losses))).argmin()
            except ValueError:
                print(
                    "Failed to compute the gradients, there might not be enough points."
                )
            if min_grad_idx is not None:
                print("Suggested LR: {:.2E}".format(lrs[min_grad_idx]))
                ax.scatter(
                    lrs[min_grad_idx],
                    losses[min_grad_idx],
                    s=75,
                    marker="o",
                    color="red",
                    zorder=3,
                    label="steepest gradient",
                )
                ax.legend()

        if log_lr:
            ax.set_xscale("log")
        ax.set_xlabel("Learning rate")
        ax.set_ylabel("Loss")

        if show_lr is not None:
            ax.axvline(x=show_lr, color="red")

        # Show only if the figure was created internally
        if fig is not None:
            plt.show()

        if suggest_lr and min_grad_idx is not None:
            return ax, lrs[min_grad_idx]
        else:
            return ax
        
class ExponentialLR(_LRScheduler):
    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr

        if num_iter <= 1:
            raise ValueError("`num_iter` must be larger than 1")
        self.num_iter = num_iter

        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # In earlier Pytorch versions last_epoch starts at -1, while in recent versions
        # it starts at 0. We need to adjust the math a bit to handle this. See
        # discussion at: https://github.com/davidtvs/pytorch-lr-finder/pull/42
#         if PYTORCH_VERSION < version.parse("1.1.0"):
#             curr_iter = self.last_epoch + 1
#             r = curr_iter / (self.num_iter - 1)
#         else:
        r = self.last_epoch / (self.num_iter - 1)

        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]


# In[ ]:


class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.out = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
        
    def forward(self, **x):
        return self.out(**x, return_dict=True)
    
model = MyModel()
criterion = torch.nn.NLLLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-7)
lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
lr_finder.range_test(train_dl, end_lr=100, num_iter=100)
lr_finder.plot()


# In[ ]:




