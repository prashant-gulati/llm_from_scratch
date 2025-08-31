##############################################################################################################################
# FINETUNING FOR CLASSIFICATION

### DOWNLOADING DATASET

import urllib.request
import ssl
import zipfile
import os
from pathlib import Path

url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return

    # Create an unverified SSL context
    ssl_context = ssl._create_unverified_context()

    # Downloading the file
    with urllib.request.urlopen(url, context=ssl_context) as response:
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())

    # Unzipping the file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    # Add .tsv file extension
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File downloaded and saved as {data_file_path}")

download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)


# After executing the preceding code, the dataset is saved as a tab-separated text file,
# SMSSpamCollection.tsv, in the sms_spam_collection folder.

# We can load it into a pandas
# DataFrame as follows:

import pandas as pd

df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
df

# When we check the class distribution, we see that the data contains "ham" (i.e., "not spam") much more frequently than "spam"

print(df["Label"].value_counts())


# For simplicity, and because we prefer a small dataset for educational purposes anyway (it will make it possible to finetune the LLM faster), we subsample (undersample) the dataset so that it contains 747 instances from each class

def create_balanced_dataset(df):

    # Count the instances of "spam"
    num_spam = df[df["Label"] == "spam"].shape[0]

    # Randomly sample "ham" instances to match the number of "spam" instances
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)

    # Combine ham "subset" with "spam"
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])

    return balanced_df

balanced_df = create_balanced_dataset(df)
print(balanced_df["Label"].value_counts())

# After executing the previous code to balance the dataset, we can see that we now have
# equal amounts of spam and non-spam messages:

# Next, we convert the "string" class labels "ham" and "spam" into integer class labels 0 and
# 1, respectively:

balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

# This process is similar to converting text into token IDs.

# However, instead of using the GPT
# vocabulary, which consists of more than 50,000 words, we are dealing with just two token
# IDs: 0 and 1.

# We create a random_split function to split the dataset into three parts: 70% for
# training, 10% for validation, and 20% for testing.

# (These ratios are common in machine
# learning to train, adjust, and evaluate models.)

def random_split(df, train_frac, validation_frac):
    # Shuffle the entire DataFrame
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # Calculate split indices
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    # Split the DataFrame
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df

train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
# Test size is implied to be 0.2 as the remainder


print(len(train_df))
print(len(validation_df))
print(len(test_df))

# Additionally, we save the dataset as CSV (comma-separated value) files, which we can
# reuse later:

train_df.to_csv("train.csv", index=None)
validation_df.to_csv("validation.csv", index=None)
test_df.to_csv("test.csv", index=None)

### CREATING DATALOADERS

# Previously, we utilized a sliding window technique to generate uniformly
# sized text chunks, which were then grouped into batches for more efficient model training.
# Each chunk functioned as an individual training instance

# In the case of email spam classification, have two primary options:

# (1) Truncate all messages to the length of the shortest message in the
# dataset or batch.

# (2) Pad all messages to the length of the longest message in the dataset or
# batch.

# Option 1 is computationally cheaper, but it may result in significant information loss if
# shorter messages are much smaller than the average or longest messages, potentially
# reducing model performance.

# So, we opt for the second option, which preserves the entire
# content of all messages.

# To implement option 2, where all messages are padded to the length of the longest
# message in the dataset, we add padding tokens to all shorter messages.

# For this purpose,
# we use "<|endoftext|>" as a padding token, as discussed in chapter 2.


# However, instead of appending the string "<|endoftext|>" to each of the text messages
# directly, we can add the token ID corresponding to "<|endoftext|>" to the encoded text

# As we have seen earlier, we first need to implement a PyTorch Dataset, which
# specifies how the data is loaded and processed, before we can instantiate the data loaders.

# For this purpose, we define the SpamDataset class.

# This SpamDataset class handles several key tasks: it identifies the
# longest sequence in the training dataset, encodes the text messages, and ensures that all
# other sequences are padded with a padding token to match the length of the longest sequence.

import torch
from torch.utils.data import Dataset


class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)

        # Pre-tokenize texts
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length

            # Truncate sequences if they are longer than max_length
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]

        # Pad sequences to the longest sequence
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length

# Step 1: Pre-tokenize texts

# Step 2: Truncate sequences if they are longer than max_length

# Step 3: Pad sequences to the longest sequence

# The SpamDataset class loads data from the CSV files we created earlier, tokenizes the text
# using the GPT-2 tokenizer from tiktoken and allows us to pad or truncate the sequences to
# a uniform length determined by either the longest sequence or a predefined maximum
# length.

# This ensures each input tensor is of the same size, which is necessary to create the
# batches in the training data loader we implement next:

train_dataset = SpamDataset(
    csv_file="train.csv",
    max_length=None,
    tokenizer=tokenizer
)

print(train_dataset.max_length)

# The code outputs 120, showing that the longest sequence contains no more than 120
# tokens, a common length for text messages.

# It's worth noting that the model can handle
# sequences of up to 1,024 tokens, given its context length limit.

# If your dataset includes
# longer texts, you can pass max_length=1024 when creating the training dataset in the
# preceding code to ensure that the data does not exceed the model's supported input
# (context) length.

# Next, we pad the validation and test sets to match the length of the longest training
# sequence.

# It's important to note that any validation and test set samples exceeding the
# length of the longest training example are truncated using
# encoded_text[:self.max_length] in the SpamDataset code we defined earlier.

# This
# truncation is optional; you could also set max_length=None for both validation and test
# sets, provided there are no sequences exceeding 1,024 tokens in these sets

val_dataset = SpamDataset(
    csv_file="validation.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)
test_dataset = SpamDataset(
    csv_file="test.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)

print(test_dataset.max_length)

# Using the datasets as inputs, we can instantiate the data loaders similarly to what we did earlier.

# However, in this case, the targets represent class labels rather than the next
# tokens in the text.

# For instance, choosing a batch size of 8, each batch will consist of 8
# training examples of length 120 and the corresponding class label of each example.

from torch.utils.data import DataLoader

num_workers = 0
batch_size = 8

torch.manual_seed(123)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True,
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)

# To ensure that the data loaders are working and are indeed returning batches of the
# expected size, we iterate over the training loader and then print the tensor dimensions of
# the last batch:

print("Train loader:")
for input_batch, target_batch in train_loader:
    pass

print("Input batch dimensions:", input_batch.shape)
print("Label batch dimensions", target_batch.shape)

# As we can see, the input batches consist of 8 training examples with 120 tokens each, as
# expected.

# The label tensor stores the class labels corresponding to the 8 training examples.

# Lastly, to get an idea of the dataset size, let's print the total number of batches in each
# dataset:

print(f"{len(train_loader)} training batches")
print(f"{len(val_loader)} validation batches")
print(f"{len(test_loader)} test batches")

# This concludes the data preparation. Next, we will prepare the model for
# finetuning.

## INITIALIZING A MODEL WITH PRETRAINED WEIGHTS

# In this section, we prepare the model we will use for the classification-finetuning to identify
# spam messages.

# We start with initializing the pretrained model we worked with in the
# previous chapter

CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Every effort moves"

BASE_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

assert train_dataset.max_length <= BASE_CONFIG["context_length"], (
    f"Dataset length {train_dataset.max_length} exceeds model's context "
    f"length {BASE_CONFIG['context_length']}. Reinitialize data sets with "
    f"`max_length={BASE_CONFIG['context_length']}`"
)

# Next, we import the download_and_load_gpt function from the gpt_download3.py file we
# downloaded earlier.

# Furthermore, we also reuse the GPTModel class and
# load_weights_into_gpt function from chapter 5 to load the downloaded weights into the
# GPT model:

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

# Corrected the module name from gpt_download3 to gpt_download
from gpt_download import download_and_load_gpt2

settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval();

# To ensure that the model was loaded correctly, let's double-check that it generates coherent text

text_1 = "Every effort moves you"

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(text_1, tokenizer),
    max_new_tokens=15,
    context_size=BASE_CONFIG["context_length"]
)

print(token_ids_to_text(token_ids, tokenizer))

# Now, before we start finetuning the model as a spam classifier, let's see if the model can
# perhaps already classify spam messages by by prompting it with instructions:

text_2 = (
    "Is the following text 'spam'? Answer with 'yes' or 'no':"
    " 'You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award.'"
)

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(text_2, tokenizer),
    max_new_tokens=23,
    context_size=BASE_CONFIG["context_length"]
)

print(token_ids_to_text(token_ids, tokenizer))

# Based on the output, it's apparent that the model struggles with following instructions.

# This is anticipated, as it has undergone only pretraining and lacks instruction-finetuning,
# which we will explore in the upcoming chapter

# The next section prepares the model for classification-finetuning

## ADDING A CLASSIFICATION HEAD

# In this section, we modify the pretrained large language model to prepare it for
# classification-finetuning.

# To do this, we replace the original output layer, which maps the
# hidden representation to a vocabulary of 50,257, with a smaller output layer that maps to
# two classes: 0 ("not spam") and 1 ("spam"),

# We could technically use a single output node since we are dealing with a binary
# classification task.

# However, this would require modifying the loss function.

# Therefore, we choose a
# more general approach where the number of output nodes matches the number of
# classes.

# For example, for a 3-class problem, such as classifying news articles as
# "Technology", "Sports", or "Politics", we would use three output nodes, and so forth.

# Before we attempt to construct the modified architecture, let's print the model
# architecture via print(model), which prints the following:

print(model)


# Above, we can see the GPT architecture neatly laid out.

# As
# discussed earlier, the GPTModel consists of embedding layers followed by 12 identical
# transformer blocks (only the last block is shown for brevity), followed by a final LayerNorm
# and the output layer, out_head.

# Next, we replace the out_head with a new output layer, as illustrated in figure 6.9, that
# we will finetune.

# To get the model ready for classification-finetuning, we first freeze the model, meaning that
# we make all layers non-trainable:

for param in model.parameters():
    param.requires_grad = False

# Then, we replace the output layer (model.out_head), which
# originally maps the layer inputs to 50,257 dimensions (the size of the vocabulary):

torch.manual_seed(123)

num_classes = 2
model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)

# Note that in the preceding code, we use BASE_CONFIG["emb_dim"], which is equal to 768 in
# the "gpt2-small (124M)" model, to keep the code below more general.

# This means we
# can also use the same code to work with the larger GPT-2 model variants.

# This new model.out_head output layer has its requires_grad attribute set to True by
# default, which means that it's the only layer in the model that will be updated during
# training.

# This new model.out_head output layer has its requires_grad attribute set to True by
# default, which means that it's the only layer in the model that will be updated during
# training.

# Additionally, we configure the last transformer block and the final LayerNorm module,
# which connects this block to the output layer, to be trainable

for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True

for param in model.final_norm.parameters():
    param.requires_grad = True

# Even though we added a new output layer and marked certain layers as trainable or nontrainable, we can still use this model in a similar way to previous chapters.

# For instance, we
# can feed it an example text identical to how we have done it in earlier chapters. For
# example, consider the following example text:

inputs = tokenizer.encode("Do you have time")
inputs = torch.tensor(inputs).unsqueeze(0)
print("Inputs:", inputs)
print("Inputs dimensions:", inputs.shape) # shape: (batch_size, num_tokens)

# Then, we can pass the encoded token IDs to the model as usual:

with torch.no_grad():
    outputs = model(inputs)

print("Outputs:\n", outputs)
print("Outputs dimensions:", outputs.shape) # shape: (batch_size, num_tokens, num_classes)

# In earlier chapters, a similar input would have produced an output tensor of [1, 4, 50257],
# where 50,257 represents the vocabulary size.

# As in previous chapters, the number of
# output rows corresponds to the number of input tokens (in this case, 4).

# However, each
# output's embedding dimension (the number of columns) is now reduced to 2 instead of
# 50,257 since we replaced the output layer of the model.

# Remember that we are interested in finetuning this model so that it returns a class label
# that indicates whether a model input is spam or not spam.

# To achieve this, we don't need to
# finetune all 4 output rows but can focus on a single output token.

# In particular, we will
# focus on the last row corresponding to the last output token

# To extract the last output token, illustrated in figure 6.11, from the output tensor, we
# use the following code:

print("Last output token:", outputs[:, -1, :])

# Having modified the model, the next section will detail the process of transforming the
# last token into class label predictions and calculate the model's initial prediction accuracy.

# Following this, we will finetune the model for the spam classification task in the subsequent
# section.

## CALCULATING THE CLASSIFICATION LOSS AND ACCURACY

# So far in this chapter, we have prepared the dataset, loaded a pretrained model, and
# modified it for classification-finetuning.

# Before we proceed with the finetuning itself, only
# one small part remains: implementing the model evaluation functions used during
# finetuning,

# Before implementing the evaluation utilities, let's briefly discuss how we convert the model
# outputs into class label predictions.

# In the previous chapter, we computed the token ID of the next token generated by the
# LLM by converting the 50,257 outputs into probabilities via the softmax function and then
# returning the position of the highest probability via the argmax function.

# In this chapter, we
# take the same approach to calculate whether the model outputs a "spam" or "not spam"
# prediction for a given input, with the only difference being that we
# work with 2-dimensional instead of 50,257-dimensional outputs.

# Let's consider the last token output from
# the previous section:

print("Last output token:", outputs[:, -1, :])

# We can obtain the class label via the following code:

probas = torch.softmax(outputs[:, -1, :], dim=-1)
label = torch.argmax(probas)
print("Class label:", label.item())

# In this case, the code returns 1, meaning the model predicts that the input text is "spam."

# Using the softmax function here is optional because the largest outputs directly correspond
# to the highest probability scores.

# Hence, we can simplify the
# code as follows, without using softmax:

logits = outputs[:, -1, :]
label = torch.argmax(logits)
print("Class label:", label.item())

# This concept can be used to compute the so-called classification accuracy, which measures
# the percentage of correct predictions across a dataset.

# To determine the classification accuracy, we apply the argmax-based prediction code to
# all examples in the dataset and calculate the proportion of correct predictions by defining a
# calc_accuracy_loader function:

def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]  # Logits of last output token
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    return correct_predictions / num_examples

# Let's use the function to determine the classification accuracies across various datasets
# estimated from 10 batches for efficiency:

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Note:
# Uncommenting the following lines will allow the code to run on Apple Silicon chips, if applicable,
# which is approximately 2x faster than on an Apple CPU (as measured on an M3 MacBook Air).
# As of this writing, in PyTorch 2.4, the results obtained via CPU and MPS were identical.
# However, in earlier versions of PyTorch, you may observe different results when using MPS.

#if torch.cuda.is_available():
#    device = torch.device("cuda")
#elif torch.backends.mps.is_available():
#    device = torch.device("mps")
#else:
#    device = torch.device("cpu")
#print(f"Running on {device} device.")

model.to(device) # no assignment model = model.to(device) necessary for nn.Module classes

torch.manual_seed(123) # For reproducibility due to the shuffling in the training data loader

train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=10)
val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=10)
test_accuracy = calc_accuracy_loader(test_loader, model, device, num_batches=10)

print(f"Training accuracy: {train_accuracy*100:.2f}%")
print(f"Validation accuracy: {val_accuracy*100:.2f}%")
print(f"Test accuracy: {test_accuracy*100:.2f}%")

# As we can see, the prediction accuracies are near a random prediction, which would be
# 50% in this case.

# To improve the prediction accuracies, we need to finetune the model.

# Classification accuracy is not a differentiable function, so we use cross entropy
# loss as a proxy to maximize accuracy.

# This is the same cross entropy loss discussed earlier.

# Accordingly, the calc_loss_batch function remains the same as in earlier, with one
# adjustment: we focus on optimizing only the last token, model(input_batch)[:, -1, :],
# rather than all tokens, model(input_batch):

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]  # Logits of last output token
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss

# We use the calc_loss_batch function to compute the loss for a single batch obtained from
# the previously defined data loaders. To calculate the loss for all batches in a data loader, we
# define the calc_loss_loader function

# Same as in chapter 5
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

# Similar to calculating the training accuracy, we now compute the initial loss for each
# data set:

with torch.no_grad(): # Disable gradient tracking for efficiency because we are not training, yet
    train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
    test_loss = calc_loss_loader(test_loader, model, device, num_batches=5)

print(f"Training loss: {train_loss:.3f}")
print(f"Validation loss: {val_loss:.3f}")
print(f"Test loss: {test_loss:.3f}")

# In the next section, we will implement a training function to finetune the model, which
# means adjusting the model to minimize the training set loss.

# Minimizing the training set
# loss will help increase the classification accuracy, our overall goal

## FINETUNING THE MODEL ON SUPERVISED DATA

# In this section, we define and use the training function to finetune the pretrained LLM and
# improve its spam classification accuracy.

# The training loop is the
# same overall training loop we used earlier, with the only difference being that we
# calculate the classification accuracy instead of generating a sample text for evaluating the
# model.

# The training function also closely mirrors
# the train_model_simple function used for pretraining the model earlier.

# The only two distinctions are that we now track the number of training examples seen
# (examples_seen) instead of the number of tokens, and we calculate the accuracy after each
# epoch instead of printing a sample text:

# Step 1: Set model to training mode
# Step 2: Reset loss gradients from previous batch iteration
# Step 3: Calculate loss gradients
# Step 4: Update model weights using loss gradients
# Step 5: New: track examples instead of tokens
# Step 6: Optional evaluation step
# Step 7: Calculate accuracy after each epoch

# Overall the same as `train_model_simple` in chapter 5
def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter):
    # Initialize lists to track losses and examples seen
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # Calculate loss gradients
            optimizer.step() # Update model weights using loss gradients
            examples_seen += input_batch.shape[0] # New: track examples instead of tokens
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Calculate accuracy after each epoch
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen

# The evaluate_model function used in the train_classifier_simple is the same as the one we used earlier.

# Same as chapter 5
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

# Next, we initialize the optimizer, set the number of training epochs, and initiate the training
# using the train_classifier_simple function.

# We will discuss the choice of the the number
# of training epochs after we evaluated the results.

# The training takes about 6 minutes on an
# M3 MacBook Air laptop computer and less than half a minute on a V100 or A100 GPU:

import time

start_time = time.time()

torch.manual_seed(123)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

num_epochs = 5
train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=50, eval_iter=5,
)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

# We then use matplotlib to plot the loss function for the training and
# validation set:

import matplotlib.pyplot as plt

def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    # Create a second x-axis for examples seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(examples_seen, train_values, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig(f"{label}-plot.pdf")
    plt.show()

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))

plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)

# As we can see based on the sharp downward slope, the model is learning well
# from the training data, and there is little to no indication of overfitting; that is, there is no
# noticeable gap between the training and validation set losses).

# Using the same plot_values function, let's now also plot the classification accuracies:

epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))

plot_values(epochs_tensor, examples_seen_tensor, train_accs, val_accs, label="accuracy")

# Based on the accuracy plot in figure 6.17, the model achieves a relatively high training and
# validation accuracy after epochs 4 and 5.

# However, it's important to note that we previously set eval_iter=5 when using the
# train_classifier_simple function, which means our estimations of training and
# validation performance were based on only 5 batches for efficiency during training.

# Now, we will calculate the performance metrics for the training, validation, and test sets
# across the entire dataset by running the following code, this time without defining the
# eval_iter value:

train_accuracy = calc_accuracy_loader(train_loader, model, device)
val_accuracy = calc_accuracy_loader(val_loader, model, device)
test_accuracy = calc_accuracy_loader(test_loader, model, device)

print(f"Training accuracy: {train_accuracy*100:.2f}%")
print(f"Validation accuracy: {val_accuracy*100:.2f}%")
print(f"Test accuracy: {test_accuracy*100:.2f}%")

# The training and test set performances are almost identical.

# A slight discrepancy between the training and test set accuracies suggests minimal
# overfitting of the training data.

# Typically, the validation set accuracy is somewhat higher
# than the test set accuracy because the model development often involves tuning
# hyperparameters to perform well on the validation set, which might not generalize as
# effectively to the test set.

# This situation is common, but the gap could potentially be minimized by adjusting the
# model's settings, such as increasing the dropout rate (drop_rate) or the weight_decay
# parameter in the optimizer configuration.

## USING THE LLM AS A SPAM CLASSIFIER

# After finetuning and evaluating the model in the previous sections, we are now in the final
# stage of this chapter:  using the model to classify spam
# messages.

# Finally, let's use the finetuned GPT-based spam classification model.

# The following
# classify_review function follows data preprocessing steps similar to those we used in the
# SpamDataset implemented earlier in this chapter.

# And then, after processing text into token
# IDs, the function uses the model to predict an integer class label, similar to what we have
# implemented earlier, and then returns the corresponding class name:

# Step 1: Prepare inputs to the model
# Step 2: Truncate sequences if they too long
# Step 3: Pad sequences to the longest sequence
# Step 4: Add batch dimension
# Step 5: Model inference without gradient tracking
# Step 6: Logits of the last output token
# Step 7: Return the classified result

def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    model.eval()

    # Prepare inputs to the model
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]
    # Note: In the book, this was originally written as pos_emb.weight.shape[1] by mistake
    # It didn't break the code but would have caused unnecessary truncation (to 768 instead of 1024)

    # Truncate sequences if they too long
    input_ids = input_ids[:min(max_length, supported_context_length)]

    # Pad sequences to the longest sequence
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0) # add batch dimension

    # Model inference
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]  # Logits of the last output token
    predicted_label = torch.argmax(logits, dim=-1).item()

    # Return the classified result
    return "spam" if predicted_label == 1 else "not spam"

# Let's try this classify_review function on an example text:

text_1 = (
    "You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award."
)

print(classify_review(
    text_1, model, tokenizer, device, max_length=train_dataset.max_length
))

# The resulting model correctly predicts "spam".

text_2 = (
    "Hey, just wanted to check if we're still on"
    " for dinner tonight? Let me know!"
)

print(classify_review(
    text_2, model, tokenizer, device, max_length=train_dataset.max_length
))

# Also, here, the model makes a correct prediction and returns a "not spam" label.

# Finally, let's save the model in case we want to reuse the model later without having to
# train it again using the torch.save method

torch.save(model.state_dict(), "review_classifier.pth")

# Once saved, the model can be loaded as follows:

model_state_dict = torch.load("review_classifier.pth")
model.load_state_dict(model_state_dict)

##############################################################################################################################
# INSTRUCTION FINE-TUNING

## STEP 1: PREPARING DATASET

# In this section, we download and format the instruction dataset for instruction finetuning a
# pretrained LLM in this chapter. The dataset consists of 1100 instruction-response pairs.

# The following code implements and executes a function to download this dataset, which
# is a relatively small file, only 204 KB in size, in JSON format. JSON, or JavaScript Object
# Notation, mirrors the structure of Python dictionaries, providing a simple structure for data
# interchange that is both human-readable and machine-friendly.

import json
import os
import urllib
import ssl

def download_and_load_file(file_path, url):
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    if not os.path.exists(file_path):
        with urllib.request.urlopen(url, context=ssl_context) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data


file_path = "instruction-data.json"
url = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
    "/main/ch07/01_main-chapter-code/instruction-data.json"
)

data = download_and_load_file(file_path, url)
print("Number of entries:", len(data))


# The data list , which we loaded from the JSON file contains the 1100 entries of the
# instruction dataset.

# Let's print one of the entries to see how each entry is structured:

print("Example entry:\n", data[50])

print("Another example entry:\n", data[999])

### CONVERTING INSTRUCTIONS INTO ALPACA FORMAT

def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text

# This format_input function takes a dictionary entry as input and constructs a formatted
# string.

# Let's test it to dataset entry data[50], which to looked at earlier:

model_input = format_input(data[50])
desired_response = f"\n\n### Response:\n{data[50]['output']}"

print(model_input + desired_response)

# Note that the format_input skips the optional ### Input: section if the 'input' field is
# empty, which we can test out by applying the format_input function to entry data[999]
# that we inspected earlier:

model_input = format_input(data[999])
desired_response = f"\n\n### Response:\n{data[999]['output']}"

print(model_input + desired_response)

### SPLITTING DATASET INTO TRAIN-TEST-VALIDATION

train_portion = int(len(data) * 0.85)  # 85% for training
test_portion = int(len(data) * 0.1)    # 10% for testing
val_portion = len(data) - train_portion - test_portion  # Remaining 5% for validation

train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]

print("Training set length:", len(train_data))
print("Validation set length:", len(val_data))
print("Test set length:", len(test_data))

# Having successfully downloaded and partitioned the dataset, and gained a clear
# understanding of the dataset prompt formatting, we are now ready for the core
# implementation of the instruction finetuning process.

## STEP 2: ORGANIZING DATA INTO TRAINING BATCHES

# In the previous chapter, the training batches were created automatically by the PyTorch
# DataLoader class, which employs a default collate function to combine lists of samples into
# batches.

# A collate function is responsible for taking a list of individual data samples and
# merging them into a single batch that can be processed efficiently by the model during
# training.

# However, the batching process for instruction finetuning in this chapter is a bit more
# involved and requires us to create our own custom collate function that we will later plug
# into the DataLoader.

# We implement this custom collate function to handle the specific
# requirements and formatting of our instruction finetuning dataset.

# First, we code an
# InstructionDataset class that applies format_input from the previous section and pretokenizes all inputs in the dataset, similar to the SpamDataset in chapter 6.

import torch
from torch.utils.data import Dataset


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data

        # Pre-tokenize texts
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)

# Similar to the approach in chapter 6, we aim to accelerate training by collecting multiple
# training examples in a batch, which necessitates padding all inputs to a similar length.

# As with the previous chapter, we use the <|endoftext|> token as a padding token.

# Instead of appending the <|endoftext|> tokens to the text inputs, we can append its
# token ID to the pre-tokenized inputs directly.

# To remind us which token ID we should use,
# we can use the tokenizer's .encode method on an <|endoftext|> token:

import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")

print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))

# In chapter 6, we padded all examples in a dataset to the same length.

# Moving on, here, we adopt a more sophisticated approach by developing a custom
# collate function that we can pass to the data loader.

# This custom collate function pads the
# training examples in each batch to have the same length, while allowing different batches
# to have different lengths.

# This approach minimizes unnecessary
# padding by only extending sequences to match the longest one in each batch, not the
# whole dataset.

# We can implement the padding process with a custom collate
# function as follows:

# Step 1: Find the longest sequence in the batch

# Step 2: Pad and prepare inputs

# Step 3: Remove extra padded token added earlier

# Step 4: Convert list of inputs to tensor and transfer to target device

def custom_collate_draft_1(
    batch,
    pad_token_id=50256,
    device="cpu"
):
    # Find the longest sequence in the batch
    # and increase the max length by +1, which will add one extra
    # padding token below
    batch_max_length = max(len(item)+1 for item in batch)

    # Pad and prepare inputs
    inputs_lst = []

    for item in batch:
        new_item = item.copy()
        # Add an <|endoftext|> token
        new_item += [pad_token_id]
        # Pad sequences to batch_max_length
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        # Via padded[:-1], we remove the extra padded token
        # that has been added via the +1 setting in batch_max_length
        # (the extra padding token will be relevant in later codes)
        inputs = torch.tensor(padded[:-1])
        inputs_lst.append(inputs)

    # Convert list of inputs to tensor and transfer to target device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    return inputs_tensor

# The custom_collate_draft_1 we implemented is designed to be integrated into a PyTorch
# DataLoader, but it can also function as a standalone tool.

# Here, we use it independently to
# test and verify that it operates as intended.

# Let's try it on three different inputs that we
# want to assemble into a batch, where each example gets padded to the same length:

inputs_1 = [0, 1, 2, 3, 4]
inputs_2 = [5, 6]
inputs_3 = [7, 8, 9]

batch = (
    inputs_1,
    inputs_2,
    inputs_3
)

print(custom_collate_draft_1(batch))

# As we can see based on the preceding output, all inputs have been padded to the length of
# the longest input list, inputs_1 containing 5 token IDs.

# So far, we have just implemented our first custom collate function to create batches from
# lists of inputs.

# However, as you learned in previous lessons, we also need to create batches
# with the target token IDs, corresponding to the batch of input IDs.

# These target IDs are crucial because they represent what we want the model to
# generate and what we need during training to calculate the loss for the weight updates,
# similar to previous chapters.

#### CREATING TARGET TOKEN IDS FOR TRAINING

# Similar to the process described for pretraining an LLM, the target token IDs
# match the input token IDs but are shifted one position to the right.

# This setup allows the LLM to learn how to predict the next token in a sequence.

# The following updated collate function generates the target token IDs from the input token IDs:

def custom_collate_draft_2(
    batch,
    pad_token_id=50256,
    device="cpu"
):
    # Find the longest sequence in the batch
    batch_max_length = max(len(item)+1 for item in batch)

    # Pad and prepare inputs
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        # Add an <|endoftext|> token
        new_item += [pad_token_id]
        # Pad sequences to max_length
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])  # Truncate the last token for inputs
        targets = torch.tensor(padded[1:])  # Shift +1 to the right for targets
        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Convert list of inputs to tensor and transfer to target device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor

# Step 1: Truncate the last token for inputs

# Step 2: Shift +1 to the right for targets

inputs_1 = [0, 1, 2, 3, 4]
inputs_2 = [5, 6]
inputs_3 = [7, 8, 9]

batch = (
    inputs_1,
    inputs_2,
    inputs_3
)

inputs, targets = custom_collate_draft_2(batch)
print(inputs)
print(targets)


# The 1st tensor represents inputs.

# The 2nd tensor represents the targets

# In the next step, we assign a -100 placeholder value to all padding tokens.

# This special value allows us to exclude these padding tokens from contributing to
# the training loss calculation, ensuring that only meaningful data influences model learning.

# In classification fine-tuning, we did not have to worry about this since we only trained the model based on
# the last output token.)

# Note that we retain one end-of-text token, ID 50256, in the target list.

# This allows the LLM to learn when to generate an end-of-text token
# in response to instructions, which we use as an indicator that the generated response is
# complete.

# In the following code, we modify our custom collate function to replace tokens with ID
# 50256 with -100 in the target lists.

# Additionally, we introduce
# an allowed_max_length parameter to optionally limit the length of the samples.

# This
# adjustment will be useful if you plan to work with your own datasets that exceed the 1024-
# token context size supported by the GPT-2 model.

# The code for this updated collate function
# is as follows:

def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    # Find the longest sequence in the batch
    batch_max_length = max(len(item)+1 for item in batch)

    # Pad and prepare inputs and targets
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        # Add an <|endoftext|> token
        new_item += [pad_token_id]
        # Pad sequences to max_length
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])  # Truncate the last token for inputs
        targets = torch.tensor(padded[1:])  # Shift +1 to the right for targets

        # New: Replace all but the first padding tokens in targets by ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # New: Optionally truncate to maximum sequence length
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Convert list of inputs and targets to tensors and transfer to target device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor

# Step 1: Replace all but the first padding tokens in targets by ignore_index

# Step 2: Optionally truncate to maximum sequence length

# Again, let's try the collate function on the sample batch that we created earlier to check
# that it works as intended:

inputs_1 = [0, 1, 2, 3, 4]
inputs_2 = [5, 6]
inputs_3 = [7, 8, 9]

batch = (
    inputs_1,
    inputs_2,
    inputs_3
)

inputs, targets = custom_collate_fn(batch)
print(inputs)
print(targets)

# The modified collate function works as expected, altering the target list by inserting the
# token ID -100.

# What is the logic behind this adjustment? Let's explore the underlying
# purpose of this modification.

# For demonstration purposes, consider the following simple and self-contained example
# where each output logit can correspond to a potential token from the model's vocabulary.

# Here's how we might calculate the cross entropy loss (introduced in chapter 5) during
# training when the model predicts a sequence of tokens, similar to what we have done in
# chapter 5 when pretraining the model, or in chapter 6 when finetuning the model for
# classification:

logits_1 = torch.tensor(
    [[-1.0, 1.0],  # 1st training example
     [-0.5, 1.5]]  # 2nd training example
)
targets_1 = torch.tensor([0, 1])


loss_1 = torch.nn.functional.cross_entropy(logits_1, targets_1)
print(loss_1)

# Adding an additional token ID will, as we would expect, affect the loss calculation.

logits_2 = torch.tensor(
    [[-1.0, 1.0],
     [-0.5, 1.5],
     [-0.5, 1.5]]  # New 3rd training example
)
targets_2 = torch.tensor([0, 1, 1])

loss_2 = torch.nn.functional.cross_entropy(logits_2, targets_2)
print(loss_2)

# Now, let's get to the interesting part and see what happens if we replace the third target
# token ID with -100:

logits_2 = torch.tensor(
    [[-1.0, 1.0],
     [-0.5, 1.5],
     [-0.5, 1.5]]  # New 3rd training example
)

targets_3 = torch.tensor([0, 1, -100])

loss_3 = torch.nn.functional.cross_entropy(logits_2, targets_3)
print(loss_3)
print("loss_1 == loss_3:", loss_1 == loss_3)

# Based on this result, we can see that the resulting loss on these 3 training examples is
# identical to the loss we calculated from the 2 training examples earlier.

# In other words, the
# cross entropy loss function ignored the third entry in the targets_3 vector, the token ID
# corresponding to -100.

# (Interested readers can try to replace the -100 value with another
# token IDs that is not 0 or 1, and will see that this results in an error.)

# So, what's so special about -100 that it's ignored by the cross entropy loss? The default
# setting of the cross entropy function in PyTorch is cross_entropy(...,
# ignore_index=-100).

# This means that it ignores targets labeled with -100.

# In this chapter, we take advantage of this ignore_index to ignore the additional end-oftext (padding) tokens that we used to pad the training examples to have the same length in
# each batch.

# However, we want to keep one 50256 (end-of-text)
# token ID in the targets because it helps the LLM to learn to generate end-of-text tokens,
# which we can use as an indicator that a response is complete.

#### MASKING TARGET TOKEN IDS

# In addition to masking out padding tokens, it is also common to mask out the target
# token IDs that correspond to the instruction

# By masking out the target token IDs that correspond to the instruction, the LLM cross entropy loss is only computed for the generated response target
# IDs.

# By masking out the instruction tokens, the model is trained to focus on generating
# accurate responses rather than additionally also memorizing instructions, which can help
# with reducing overfitting.

# Currently, researchers are divided on whether masking the instructions is universally beneficial during instruction finetuning.

# For instance, a recent
# paper titled "Instruction Tuning With Loss Over Instructions" demonstrated that not
# masking the instructions benefits the LLM performance.

# In this chapter, we do not apply masking and leave it as an optional
# exercise for the reader.

## STEP 3: CREATING DATALOADERS FOR AN INSTRUCTION DATASET

# The custom_collate_fn includes code to move the input and target tensors (for
# example, torch.stack(inputs_lst).to(device)) to a specified device, which can be
# either "cpu" or "cuda" (for GPUs), or optionally "mps" for Macs with Apple Silicon chips.

# In previous chapters, we moved the data onto the target device (for example, the GPU
# memory when device="cuda") in the main training loop. Having this as part of the collate
# function offers the advantage of performing this device transfer process as a background
# process outside the training loop, preventing it from blocking the GPU during model
# training.

# The following code initializes the device variable:

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Note:
# Uncommenting the following lines will allow the code to run on Apple Silicon chips, if applicable,
# which is much faster than on an Apple CPU (as measured on an M3 MacBook Air).
# However, the resulting loss values may be slightly different.

#if torch.cuda.is_available():
#    device = torch.device("cuda")
#elif torch.backends.mps.is_available():
#    device = torch.device("mps")
#else:
#    device = torch.device("cpu")

print("Device:", device)

# Next, to reuse the chosen device setting in custom_collate_fn when we plug it into the
# PyTorch DataLoader class later in this section, we use the partial function from Python's
# functools standard library to create a new version of the function with the device
# argument pre-filled.

# Additionally, we set the allowed_max_length to 1024, which truncates
# the data to the maximum context length supported by the GPT-2 model we finetune later in
# this chapter:

from functools import partial
customized_collate_fn = partial(custom_collate_fn, device=device, allowed_max_length=1024)

# Next, we can set up the data loaders as we did in previous chapters, but this time we will
# use our custom collate function for the batching process:

from torch.utils.data import DataLoader


num_workers = 0
batch_size = 8

torch.manual_seed(123)

train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers
)

val_dataset = InstructionDataset(val_data, tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

test_dataset = InstructionDataset(test_data, tokenizer)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

# Let's examine the dimensions of the input and target batches generated by the training
# loader:

print("Train loader:")
for inputs, targets in train_loader:
    print(inputs.shape, targets.shape)

# In the preceding output, we can see that the first input and target batch have dimensions
# 8×61, where 8 represents the batch size, and 61 is the number of tokens in each training
# example in this batch.

# The second input and target batch have a different number of
# tokens, for instance, 76.


# As we saw in the preceding code output, thanks to our custom collate function, the data
# loader is able to create batches of different lengths.

# In the next section, we load a
# pretrained LLM that we can then finetune with this data loader.

## STEP 4: LOADING A PRETRAINED LLM

# Before beginning instruction finetuning, we first load a pretrained GPT model,

# Instead of using the smallest 124 million
# parameter model as before, we load the medium-sized model with 355 million parameters.

# The reason for this choice is that the 124 million parameter model is too limited in capacity
# to achieve qualitatively satisfactory results via instruction finetuning.

# This is done using the same code as in section 5.5 of chapter 5 and section 6.4 of
# the previous chapter, except that we now specify "gpt2-medium (355M)" instead of "gpt2-small
# (124M)".

# Please note that executing the code provided below will initiate the download of
# the medium-sized GPT model, which has a storage requirement of approximately 1.42
# gigabytes.

# This is roughly three times larger than the storage space needed for the small
# model:

from gpt_download import download_and_load_gpt2

BASE_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

CHOOSE_MODEL = "gpt2-medium (355M)"

BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(
    model_size=model_size,
    models_dir="gpt2"
)

model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval();

# Before diving into finetuning the model in the next section, let's take a moment to assess
# the pretrained LLM's performance on one of the validation tasks by comparing its output to
# the expected response.

# This will give us a baseline understanding of how well the model
# performs on an instruction-following task right out of the box, prior to finetuning, and will
# help us appreciate the impact of finetuning later on.

# We use the first example from the
# validation set for this assessment:

torch.manual_seed(123)
input_text = format_input(val_data[0])
print(input_text)

token_ids = generate(
    model=model,
    idx=text_to_token_ids(input_text, tokenizer),
    max_new_tokens=35,
    context_size=BASE_CONFIG["context_length"],
    eos_id=50256,
)
generated_text = token_ids_to_text(token_ids, tokenizer)

# It's important to note that the generate function returns the combined input and output
# text.

# This behavior was convenient in previous chapters since pretrained LLMs are primarily
# designed as text-completion models, where the input and output are concatenated to
# create a coherent and legible text.

# However, when evaluating the model's performance on a
# specific task, we often want to focus solely on the model's generated response.

# To isolate the model's response text, we need to subtract the length of the input
# instruction from the start of the generated_text:

response_text = generated_text[len(input_text):].strip()
print(response_text)

# This code snippet removes the input text from the beginning of the generated_text,
# leaving us with only the model's generated response. The strip() function is then applied
# to remove any leading or trailing whitespace characters. The output is as follows:

# As we can see from the output, the pretrained model is not yet capable of correctly
# following the given instruction.

# While it does create a "Response" section, it simply repeats
# the original input sentence and part of the instruction, failing to convert the active sentence
# to passive voice as requested.


# In the upcoming section, we implement the finetuning process to improve the model's
# ability to comprehend and appropriately respond to such requests.

## STEP 5: FINETUNING THE LLM ON INSTRUCTION DATA

# We already did all the hard work when we implemented the
# instruction dataset processing at the beginning of this chapter.

# For the finetuning process
# itself, we can reuse the loss calculation and training functions implemented in chapter 5
# during the pretraining:

# Before we begin training, let's calculate the initial loss for the training and validation sets:

#### PREVIOUSLY DEFINED FUNCTIONS WHICH WE WILL REQUIRE

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # Calculate loss gradients
            optimizer.step() # Update model weights using loss gradients
            tokens_seen += input_batch.numel() # Returns the total number of elements (or tokens) in the input_batch.
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Print a sample text after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen


model.to(device)

torch.manual_seed(123)

with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)

print("Training loss:", train_loss)
print("Validation loss:", val_loss)

# With the model and data loaders prepared, we can now proceed to train the model.

# The following code sets up the training process, including initializing the optimizer, setting the
# number of epochs, and defining the evaluation frequency and starting context to evaluate
# generated LLM responses during training based on the first validation set instruction
# (val_data[0]) we looked at earlier:

import time

start_time = time.time()

torch.manual_seed(123)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)

num_epochs = 1

train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context=format_input(val_data[0]), tokenizer=tokenizer
)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

# As we can see based on the outputs above, the model trains well, as we can tell based on the decreasing training loss and validation loss values.

# Furthermore, based on the response text printed after each epoch, we can see that the model almost correctly follows the instruction to convert the input sentence 'The chef cooks the meal every day.' into passive voice 'The meal is prepared every day by the chef.' (We will properly format and evaluate the responses in a later section.

# To get better results, we need to finetune the model for more epochs.

# Finally, let's take a look at the training and validation loss curves

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig("loss-plot.pdf")
    plt.show()

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

# As we can see in the loss plot shown above, the model's performance on both the
# training and validation sets improves substantially over the course of training.

# The rapid
# decrease in losses during the initial phase indicates that the model is quickly learning
# meaningful patterns and representations from the data. Then, as training progresses to the
# second epoch, the losses continue to decrease but at a slower rate, suggesting that the
# model is finetuning its learned representations and converging to a stable solution.


# While the loss plot in figure 7.17 indicates that the model is training effectively, the most
# crucial aspect is its performance in terms of response quality and correctness. In the
# remaining sections of this chapter, we will extract the responses and store them in a format
# that allows us to evaluate and quantify the response quality.

## STEP 6: EXTRACTING AND SAVING RESPONSES

# After finetuning the LLM on the training portion of the instruction dataset as described in
# the previous section, we now proceed to evaluate its performance on the held-out test set.

# To accomplish this, we first extract the model-generated responses for each input in the
# test dataset and collect them for manual analysis

# Step 1: Iterate over the first 3 test set samples

# Step 2:  Use the generate function defined earlier

# As mentioned earlier, the generate function returns the combined input and output text, so
# we use slicing and the .replace() method on the generated_text contents to extract the
# model's response.

# The instructions, followed by the given test set response and model
# response are shown below:

torch.manual_seed(123)


for entry in test_data[:3]:

    input_text = format_input(entry)

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    response_text = (
        generated_text[len(input_text):]
        .replace("### Response:", "")
        .strip()
)

    print(input_text)
    print(f"\nCorrect response:\n>> {entry['output']}")
    print(f"\nModel response:\n>> {response_text.strip()}")
    print("-------------------------------------")

# As we can see based on the test set instructions, given responses, and the model's
# responses, the model performs relatively well.

# The answers to the first instruction
# is clearly correct, while the second answer and the third answers are not correct.

# This is because we have done the fine-tuning for only 1 epoch due to hardware limitations. To get better results, we need to increase the epochs to at least 2.

# Most importantly, we can see that model evaluation is not as straightforward as in the
# previous chapter, where we simply calculated the percentage of correct spam/non-spam
# class labels to obtain the classification accuracy.

# In practice, instruction-finetuned LLMs
# such as chatbots are evaluated via multiple approaches:

# 1. Short-answer and multiple choice benchmarks such as MMLU ("Measuring
# Massive Multitask Language Understanding," https://arxiv.org/abs/2009.
# 03300), which test the general knowledge of a model.

# 2. Human preference comparison to other LLMs, such as LMSYS chatbot
# arena (https://arena.lmsys.org).

# 3. Automated conversational benchmarks, where another LLM like GPT-4 is
# used to evaluate the responses, such as AlpacaEval (https://tatsulab.github.io/alpaca_eval/).
# completes the request.

# Considering the scale of the task at hand, we will implement an approach similar to
# method 3, which involves evaluating the responses automatically using another LLM.

# This
# will allow us to efficiently assess the quality of the generated responses without the need
# for extensive human involvement, thereby saving time and resources while still obtaining
# meaningful performance indicators.

# To prepare the responses for this evaluation process, we append the generated model
# responses to the test_set dictionary and save the updated data as an "instructiondata-with-response.json" file for record keeping.

# Additionally, by saving this file, we can
# easily load and analyze the responses in separate Python sessions later on if needed.

# The following code uses the generate method in the same manner as before; however,
# we now iterate over the entire test_set.

# Also, instead of printing the model responses, we
# add them to the test_set dictionary:

from tqdm import tqdm

for i, entry in tqdm(enumerate(test_data), total=len(test_data)):

    input_text = format_input(entry)

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    response_text = generated_text[len(input_text):].replace("### Response:", "").strip()

    test_data[i]["model_response"] = response_text


with open("instruction-data-with-response.json", "w") as file:
    json.dump(test_data, file, indent=4)  # "indent" for pretty-printing

# Let's verify that the responses have been correctly added to the test_set dictionary by
# examining one of the entries:

print(test_data[0])

# Based on the output, we can see that the model_response has been added correctly.

# Finally, we save the model as gpt2-medium355M-sft.pth file to be able to reuse it in future
# projects:

import re


file_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL) }-sft.pth"
torch.save(model.state_dict(), file_name)
print(f"Model saved as {file_name}")

# Load model via
# model.load_state_dict(torch.load("gpt2-medium355M-sft.pth"))

## STEP 7: EVALUATING THE FINE-TUNED LLM

# Previously, we judged the performance of an instruction finetuned model by looking at its
# responses on 3 examples of the test set.

# While this gives us a rough idea of how well the
# model performs, this method does not really scale well to larger amounts of responses.

# So, in this section, we implement a method to automate the response evaluation of the finetuned LLM using another, larger LLM.

# To implement the evaluation step which involves evaluating test set responses in
# an automated fashion, we utilize an existing instruction-finetuned 8 billion parameter Llama
# 3 model developed by Meta AI.

# This model can be run locally using the open-source Ollama
# application (https://ollama.com).

# Ollama is an efficient application for running LLMs on a laptop.

# It serves as a wrapper
# around the open-source llama.cpp library (https://github.com/ggerganov/llama.cpp), which
# implements LLMs in pure C/C++ to maximize efficiency.

# However, note that Ollama is only
# a tool for generating text using LLMs (inference) and does not support training or finetuning
# LLMs.

# The following code verifies that the Ollama session is running properly before we use
# Ollama to evaluate the test set responses generated in the previous section:

import psutil

def check_if_running(process_name):
    running = False
    for proc in psutil.process_iter(["name"]):
        if process_name in proc.info["name"]:
            running = True
            break
    return running

ollama_running = check_if_running("ollama")

if not ollama_running:
    raise RuntimeError("Ollama not running. Launch ollama before proceeding.")
print("Ollama running:", check_if_running("ollama"))

# An alternative to the ollama run command for interacting with the model is through its
# REST API using Python.

# The following query_model function demonstrates how to use the
# API:

# Step 1: Create the data payload as a dictionary

# Step 2: Convert the dictionary to a JSON formatted string and encode it to bytes

# Step 3: Create a request object, setting the method to POST and adding necessary headers

# Step 4: Send the request and capture the response

import urllib.request

def query_model(
    prompt,
    model="llama3",
    url="http://localhost:11434/api/chat"
):
    # Create the data payload as a dictionary
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "options": {     # Settings below are required for deterministic responses
            "seed": 123,
            "temperature": 0,
            "num_ctx": 2048
        }
    }


    # Convert the dictionary to a JSON formatted string and encode it to bytes
    payload = json.dumps(data).encode("utf-8")

    # Create a request object, setting the method to POST and adding necessary headers
    request = urllib.request.Request(
        url,
        data=payload,
        method="POST"
    )
    request.add_header("Content-Type", "application/json")

    # Send the request and capture the response
    response_data = ""
    with urllib.request.urlopen(request) as response:
        # Read and decode the response
        while True:
            line = response.readline().decode("utf-8")
            if not line:
                break
            response_json = json.loads(line)
            response_data += response_json["message"]["content"]

    return response_data

# Before running the subsequent code cells in this notebook, ensure that Ollama is still
# running. The previous code cells should print "Ollama running: True" to confirm that the
# model is active and ready to receive requests.

# Here's an example of how to use the query_llama function we just implemented:

model = "llama3"
result = query_model("What do Llamas eat?", model)
print(result)

# Using the query_model function defined earlier, we can evaluate the responses generated
# by our finetuned model with a prompt that prompts the Llama 3 model to rate our
# finetuned model's responses on a scale from 0 to 100 based on the given test set response
# as reference.

# First, we apply this approach to the first three examples from the test set that we
# examined in a previous section:

for entry in test_data[:3]:
    prompt = (
        f"Given the input `{format_input(entry)}` "
        f"and correct output `{entry['output']}`, "
        f"score the model response `{entry['model_response']}`"
        f" on a scale from 0 to 100, where 100 is the best score. "
    )
    print("\nDataset response:")
    print(">>", entry['output'])
    print("\nModel response:")
    print(">>", entry["model_response"])
    print("\nScore:")
    print(">>", query_model(prompt))
    print("\n-------------------------")

# Based on the generated responses, we can observe that the Llama 3 model provides
# reasonable evaluations and is capable of assigning partial points when a model's answer is
# not entirely correct.

# The previous prompt returns highly detailed evaluations in addition to the score.

# We can
# modify the prompt to just generate integer scores ranging from 0 to 100, where 100
# represents the best possible score.

# This modification allows us to calculate an average score
# for our model, which serves as a more concise and quantitative assessment of its
# performance.

# The following generate_model_scores function uses a modified the prompt telling the
# model to "Respond with the integer number only.":

for entry in test_data[:2]:
    prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry['model_response']}`"
            f" on a scale from 0 to 100, where 100 is the best score. "
            f"Respond with the integer number only."
        )
    score = query_model(prompt, model)
    print("\nDataset response:")
    print(">>", entry['output'])
    print("\nModel response:")
    print(">>", entry["model_response"])
    print("\nScore:")
    print(">>", query_model(prompt, model))
    print("\n-------------------------")

def generate_model_scores(json_data, json_key, model="llama3"):
    scores = []
    for entry in tqdm(json_data, desc="Scoring entries"):
        prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry[json_key]}`"
            f" on a scale from 0 to 100, where 100 is the best score. "
            f"Respond with the integer number only."
        )
        score = query_model(prompt, model)
        try:
            scores.append(int(score))
        except ValueError:
            print(f"Could not convert score: {score}")
            continue

    return scores

# I am not running the above function because of hardware limitations. I am using a Macbook Air 2020.

# It takes about 1 min  on a M3 Macbook Air.

# When you run the above code, you will see that the evaluation output shows that our finetuned model achieves an average score above 50,
# which provides a useful benchmark for comparison against other models or for
# experimenting with different training configurations to improve the model's performance.

# It's worth noting that Ollama is not entirely deterministic at the time of this writing,
# which means that the scores you obtain might slightly vary from the ones presented above.

# To obtain more robust results, you can repeat the evaluation multiple times and average
# the resulting scores.

# To further improve our model's performance, we can explore various strategies, such as:

# (1) Adjusting the hyperparameters during finetuning, such as the learning
# rate, batch size, or number of epochs.
# (2) Increasing the size of the training dataset or diversifying the examples to
# cover a broader range of topics and styles.
# (3) Experimenting with different prompts or instruction formats to guide the
# model's responses more effectively.
# (4) Considering the use of a larger pretrained model, which may have greater
# capacity to capture complex patterns and generate more accurate
# responses.
# (5) We can also use parameter efficient fine-tuning techniques like LoRA.