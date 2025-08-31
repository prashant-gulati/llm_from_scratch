##############################################################################################################################
##############################################################################################################################
## Building Blocks

##############################################################################################################################
## Step 1: Creating Tokens

# Read in a short story as text sample into Python
# Print total number of characters & first 100 characters of this file
import os
import ssl
import certifi
import urllib.request

file_path = "the-verdict.txt"
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

if not os.path.exists(file_path):
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(url, context=ssl_context) as response:
        raw_text = response.read().decode('utf-8')
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(raw_text)
else:
    with open(file_path, "r", encoding="utf-8") as file:
        raw_text = file.read()

print("Total number of character:", len(raw_text))
print(raw_text[:99])

# Goal: tokenize this 20,479-character short story into individual words and special
# characters that we can then turn into embeddings for LLM training. We use a single book to
# understand the text processing steps and to make it possible to run it in reasonable time on consumer hardware.

# Using some simple example text, we can use the re.split to split a text on whitespace characters (\s = space, tab, newline)

import re

text = "Hello, world. This, is a test."
result = re.split(r'(\s)', text)

print(result)

# The result is a list of individual words, whitespaces, and punctuation characters.
# Let's modify the regular expression splits on whitespaces (\s) and commas, and periods ([,.]):

result = re.split(r'([,.]|\s)', text)

print(result)

# A small remaining issue is that the list still includes whitespace characters.
# We can remove these redundant characters safely as follows:

result = [item for item in result if item.strip()]
print(result)

# The tokenization scheme we devised above works well on the simple sample text. Let's
# modify it a bit further so that it can also handle other types of punctuation, such as
# question marks, quotation marks, and the double-dashes we have seen earlier in the first
# 100 characters of the short story, along with additional special characters:

text = "Hello, world. Is this-- a test?"
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
# Strip whitespace (spaces, tabs, newlines) from each item and then filter out any empty strings.
result = [item.strip() for item in result if item.strip()]
print(result)

# REMOVING WHITESPACES OR NOT
# Removing whitespaces reduces the memory and computing requirements. 
# However, keeping whitespaces can be useful if we train models that
# are sensitive to the exact structure of the text (for example, Python code, which is sensitive to indentation and spacing) 
# Here, we remove whitespaces for simplicity and brevity of the tokenized outputs.

# Basic tokenizer working, let's apply it to Edith Wharton's entire short story

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(preprocessed[:30])
print(len(preprocessed))

##############################################################################################################################
## Step 2: Creating Token IDs

# Goal: create a list of all unique tokens and sort them alphabetically to determine the vocabulary size:
# set() drops duplicate values. all_words is a set

all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)

# After determining that the vocabulary size is 1,130 via the above code, we create the
# vocabulary and print its first 51 entries for illustration purposes. vocab is a dict

vocab = {token:integer for integer,token in enumerate(all_words)}

for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break

# The dictionary contains individual tokens associated with unique integer labels.
# For when we want to convert the outputs of an LLM from numbers back into text, we also need the inverse capability

# Complete tokenizer class in Python:
# 1) Encode method that splits
# text into tokens and carries out the string-to-integer mapping to produce token IDs via the vocabulary.

# 2) Decode method that carries out the reverse integer-to-string mapping to convert the token IDs back into text

# Step 1: Store the vocabulary as a class attribute for access in the encode and decode methods
# Step 2: Create an inverse vocabulary that maps token IDs back to the original text tokens
# Step 3: Process input text into token IDs
# Step 4: Convert token IDs back into text
# Step 5: Replace spaces before the specified punctuation

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)

        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    # ids is a list
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

# Let's instantiate a new tokenizer object from the SimpleTokenizerV1 class and tokenize a
# passage from Edith Wharton's short story to try it out in practice:

tokenizer = SimpleTokenizerV1(vocab)

text = """"It's the last he painted, you know,"
           Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)
 
# Next, let's see if we can turn these token IDs back into text using the decode method:

print(tokenizer.decode(ids))

# Based on the output above, we can see that the decode method successfully converted the
# token IDs back into the original text. Let's now apply it to a new text sample that is not contained in the training set:

text = "Hello, do you like tea?"
#print(tokenizer.encode(text))

# The problem is that the word "Hello" was not used in the The Verdict short story and isn't in the vocab

### ADDING SPECIAL CONTEXT TOKENS
# We will modify this tokenizer to handle unknown words

# In particular, we will modify the vocabulary and tokenizer we implemented in the
# previous section, SimpleTokenizerV2, to support two new tokens, <|unk|> and <|endoftext|> for OOV words or between unrelated texts

all_tokens = sorted(set(preprocessed))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])

vocab = {token:integer for integer,token in enumerate(all_tokens)}
len(vocab.items())

# New vocabulary size is 1132. As an additional quick check, let's print the last 5 entries of the updated vocabulary:

for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)

# Class: A simple text tokenizer that handles unknown words
# Step 1: Replace unknown words by <|unk|> tokens
# Step 2: Replace spaces before the specified punctuations

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_to_int
            else "<|unk|>" for item in preprocessed
        ]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text

tokenizer = SimpleTokenizerV2(vocab)

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."

text = " <|endoftext|> ".join((text1, text2))

print(text)

print(tokenizer.encode(text))
print(tokenizer.decode(tokenizer.encode(text)))

# Based on comparing the de-tokenized text above with the original input text, we know that
# the training dataset, Edith Wharton's short story The Verdict, did not contain the words "Hello" and "palace."

# Other additional special tokens:

# [BOS] (beginning of sequence): This token marks the start of a text. It signifies to the LLM where a piece of content begins.

# [EOS] (end of sequence): This token is positioned at the end of a text,
# and is especially useful when concatenating multiple unrelated texts,
# similar to <|endoftext|>

# [PAD] (padding): When training LLMs with batch sizes larger than one,
# the batch might contain texts of varying lengths. To ensure all texts have
# the same length, the shorter texts are extended or "padded" using the
# [PAD] token, up to the length of the longest text in the batch.

# Tokenizer used for GPT models does not need any of these tokens mentioned above but only uses an <|endoftext|> token for simplicity
# Also doesn't use an <|unk|> token for outof-vocabulary words. Instead, GPT models use a byte pair encoding tokenizer

### BYTE PAIR ENCODING (BPE)

# Since implementing BPE is complicated, we will use an existing Python
# open-source library called tiktoken (https://github.com/openai/tiktoken), used to train LLMs such as GPT-2, GPT-3

import importlib
import importlib.metadata
import tiktoken

print("tiktoken version:", importlib.metadata.version("tiktoken"))
tokenizer = tiktoken.get_encoding("gpt2")

# The usage of this tokenizer is similar to SimpleTokenizerV2 we implemented previously via an encode method:

text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."
)

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)

# We can then convert the token IDs back into text using the decode method

strings = tokenizer.decode(integers)
print(strings)

# First, the <|endoftext|> token is assigned a relatively large token ID 50256
# BPE tokenizer has a total vocabulary size of 50,257

# Second, the BPE tokenizer above encodes and decodes unknown words, such as "someunknownPlace" correctly.

# The algorithm underlying BPE breaks down words that aren't in its predefined vocabulary into smaller subword units or even individual characters.
# The enables it to handle out-of-vocabulary words.

integers = tokenizer.encode("Akwirw ier")
print(integers)

strings = tokenizer.decode(integers)
print(strings)

import tiktoken

# Initialize the encodings for GPT-2, GPT-3, and GPT-4
encodings = {
    "gpt2": tiktoken.get_encoding("gpt2"),
    "gpt3": tiktoken.get_encoding("p50k_base"),  # Commonly associated with GPT-3 models
    "gpt4": tiktoken.get_encoding("cl100k_base")  # Used for GPT-4 and later versions
}

# Get the vocabulary size for each encoding
vocab_sizes = {model: encoding.n_vocab for model, encoding in encodings.items()}

# Print the vocabulary sizes
for model, size in vocab_sizes.items():
    print(f"The vocabulary size for {model.upper()} is: {size}")

### CREATING INPUT-TARGET PAIRS

# Data loader that fetches the input-target pairs using a sliding window approach
# First tokenize the whole The Verdict short story with the BPE tokenizer

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)
print(f"Num tokens in the_verdict: {len(enc_text)}")

# Next, we remove the first 50 tokens from the dataset for demonstration purposes as it
# results in a slightly more interesting text passage in the next steps:

enc_sample = enc_text[50:]

# One of the easiest and most intuitive ways to create the input-target pairs for the nextword prediction 
# task is to create two variables, x and y
# where x contains the input tokens and y contains the targets, which are the inputs shifted by 1:
# The context size determines how many tokens are included in the input

context_size = 4 #length of the input

#The context_size of 4 means that the model is trained to look at a sequence of 4 words (or tokens)
#to predict the next word in the sequence.
#The input x is the first 4 tokens [1, 2, 3, 4], and the target y is the next 4 tokens [2, 3, 4, 5]

x = enc_sample[0:context_size]
y = enc_sample[1:context_size+1]

print(f"x: {x}")
print(f"y:      {y}")

# Processing the inputs along with the targets, which are the inputs shifted by one position,
# we can then create the next-word prediction tasks as follows:

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]

    print(context, "---->", desired)

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]

    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))

# Everything left of the arrow (---->) refers to the input an LLM would receive, and the token
# ID on the right side of the arrow represents the target token ID that the LLM is supposed to predict

# To turn the tokens into embeddings: implement an efficient data loader that
# iterates over the input dataset and returns the inputs and targets as PyTorch tensors

# In particular, we are interested in returning two tensors: an input tensor containing the
# text that the LLM sees and a target tensor that includes the targets for the LLM to predict,

### IMPLEMENTING A DATA LOADER

# For efficient data loader implementation, we will use PyTorch's built-in Dataset and DataLoader classes.

# Step 1: Tokenize the entire text
# Step 2: Use a sliding window to chunk the book into overlapping sequences of max_length
# Step 3: Return the total number of rows in the dataset
# Step 4: Return a single row from the dataset

from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

# The GPTDatasetV1 class in listing 2.5 is based on the PyTorch Dataset class.
# It defines how individual rows are fetched from the dataset.

# Each row consists of a number of token IDs (based on a max_length) assigned to an input_chunk tensor.

# The target_chunk tensor contains the corresponding targets.

# The following code will use the GPTDatasetV1 to load the inputs in batches via a PyTorch DataLoader:
# Step 1: Initialize the tokenizer
# Step 2: Create dataset
# Step 3: drop_last=True drops the last batch if it is shorter than the specified batch_size to prevent loss spikes
# during training
# Step 4: The number of CPU processes to use for preprocessing

def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):

    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader

# Let's test the dataloader with a batch size of 1 for an LLM with a context size of 4
# This will develop an intuition of how the GPTDatasetV1 class and the create_dataloader_v1 function work together:

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Convert dataloader into a Python iterator to fetch the next entry via Python's built-in next() function

import torch
print("PyTorch version:", torch.__version__)

dataloader = create_dataloader_v1(
    raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
)

data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)

# The first_batch variable contains two tensors: the first tensor stores the input token IDs,
# and the second tensor stores the target token IDs.

# Since the max_length is set to 4, each of the two tensors contains 4 token IDs.
# It is common to train LLMs with input sizes of at least 256

# The stride setting dictates the number of positions the inputs shift across batches, emulating a sliding
# window approach

second_batch = next(data_iter)
print(second_batch)

# Small batch sizes require less memory during training but lead to more noisy model updates.
# The batch size is a hyperparameter to experiment with when training LLMs.

# Data loader to sample with a batch size greater than 1
# We increase the stride to 4. This is to utilize the data set fully (we don't skip a
# single word) but also avoid any overlap between the batches, since more overlap could lead to increased overfitting.

dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)

### CREATING TOKEN EMBEDDINGS

# Suppose we have the following four input tokens with IDs 2, 3, 5, and 1:

input_ids = torch.tensor([2, 3, 5, 1])

# Suppose we have a small vocabulary of only 6 words and we want to create embeddings of size 3 (GPT-3: 50,257 -> 12,288)
# Using the vocab_size and output_dim, we can instantiate an embedding layer in PyTorch,
# setting the random seed to 123 for reproducibility purposes:

vocab_size = 6
output_dim = 3

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# Embedding layer's underlying weight matrix:
print(embedding_layer.weight)

# We can see that the weight matrix of the embedding layer contains small, random values.
# These values are optimized during LLM training as part of the LLM optimization itself. 
# The weight matrix has six rows and three columns. 
# There is one row for each of the six possible tokens in the vocabulary.
# And there is one column for each of the three embedding dimensions.

# After we instantiated the embedding layer, let's now apply it to a token ID to obtain the embedding vector:

print(embedding_layer(torch.tensor([3])))

# If we compare the embedding vector for token ID 3 to the previous embedding matrix, we
# see that it is identical to the 4th row (Python starts with a zero index, so it's the row
# corresponding to index 3). In other words, the embedding layer is essentially a look-up
# operation that retrieves rows from the embedding layer's weight matrix via a token ID.

print(embedding_layer(input_ids))

# Each row in this output matrix is obtained via a lookup operation from the embedding weight matrix

### POSITIONAL EMBEDDINGS (ENCODING WORD POSITIONS)

# We consider more realistic and useful embedding sizes and encode the input
# tokens into a 256-dimensional vector representation.
# Furthermore, we assume that the token IDs were created by the BPE
# tokenizer that we implemented earlier, which has a vocabulary size of 50,257:

vocab_size = 50257
output_dim = 256

token_embedding_layer_vocab = torch.nn.Embedding(vocab_size, output_dim)

# Using the token_embedding_layer_vocab above, if we sample data from the data loader, we
# embed each token in each batch into a 256-dimensional vector. If we have a batch size of 8
# with four tokens each, the result will be an 8 x 4 x 256 tensor.

max_length = 4
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length,
    stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)

token_embeddings = token_embedding_layer_vocab(inputs)
print(token_embeddings.shape)

# As we can tell based on the 8x4x256-dimensional tensor output, each token ID is now
# embedded as a 256-dimensional vector.

# For a GPT model's absolute embedding approach, we just need to create another
# embedding layer that has the same dimension as the token_embeddings:

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

pos_embeddings = pos_embedding_layer(torch.arange(max_length))
print(pos_embeddings.shape)

# The input to the pos_embeddings is usually a
# placeholder vector torch.arange(context_length), which contains a sequence of
# numbers 0, 1, ..., up to the maximum input length − 1.

# The context_length is a variable that represents the supported input size of the LLM.
# Here, we choose it similar to the maximum length of the input text.
# In practice, input text can be longer than the supported context length, in which case we have to truncate the text.

# As we can see, the positional embedding tensor consists of four 256-dimensional vectors.
# We can now add these directly to the token embeddings, where PyTorch will add the 4x256-
# dimensional pos_embeddings tensor to each 4x256-dimensional token embedding tensor in
# each of the 8 batches (broadcasting):

input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)

##############################################################################################################################
## Step 3: IMPLEMENTING A SIMPLIFIED ATTENTION MECHANISM

# Consider the following input sentence, which has already been embedded into 3-dimensional vectors
# We choose a small embedding dimension for illustration purposes to ensure it fits on the page without line breaks:

import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55] # step     (x^6)
   ]
)

# Corresponding words
words = ['Your', 'journey', 'starts', 'with', 'one', 'step']

# Extract x, y, z coordinates
x_coords = inputs[:, 0].numpy()
y_coords = inputs[:, 1].numpy()
z_coords = inputs[:, 2].numpy()

# Create 3D plot with vectors from origin to each point, using different colors
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define a list of colors for the vectors
colors = ['r', 'g', 'b', 'c', 'm', 'y']

# Plot each vector with a different color and annotate with the corresponding word
for (x, y, z, word, color) in zip(x_coords, y_coords, z_coords, words, colors):
    # Draw vector from origin to the point (x, y, z) with specified color and smaller arrow length ratio
    ax.quiver(0, 0, 0, x, y, z, color=color, arrow_length_ratio=0.05)
    ax.text(x, y, z, word, fontsize=10, color=color)

# Set labels for axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set plot limits to keep arrows within the plot boundaries
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_zlim([0, 1])

plt.title('3D Plot of Word Embeddings with Colored Vectors')
plt.show()

# Each row represents a word, and each column represents an embedding dimension
# The second input token serves as the query

query = inputs[1]  # 2nd input token is the query

# Attention Scores
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query) # dot product (transpose not necessary here since they are 1-dim vectors)

print(attn_scores_2)

# Attention Weights
# Normalize each of the attention scores that we computed previously to obtain attention weights that sum up to 1.
# Useful for interpretation and for maintaining training stability in an LLM.

attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()

print("Attention weights:", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())

# In practice, it's more common and advisable to use the softmax function for normalization.
# This approach is better at managing extreme values and offers more favorable gradient properties during training

def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2_naive = softmax_naive(attn_scores_2)

print("Attention weights:", attn_weights_2_naive)
print("Sum:", attn_weights_2_naive.sum())

# Softmax function ensures that the attention weights are always positive.
# This makes the output interpretable as probabilities or relative importance, where higher
# weights indicate greater importance.

# Note that this naive softmax implementation (softmax_naive) may encounter numerical
# instability problems, such as overflow and underflow, when dealing with large or small input values.
# Therefore, in practice, it's advisable to use the PyTorch implementation of softmax,
# which has been extensively optimized for performance: softmax(x - max(x))

attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Attention weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())

# In this case, we can see that it yields the same results as our previous softmax_naive

# The context vector z(2) is calculated as a weighted sum of all input vectors.
# This involves multiplying each input vector by its corresponding attention weight:

query = inputs[1] # 2nd input token is the query

context_vec_2 = torch.zeros(query.shape)
for i,x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i]*x_i

print(context_vec_2)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

inputs2 = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55], # step     (x^6)
   [0.4419, 0.6515, 0.5683]]
)

# Corresponding words
words2 = ['Your', 'journey', 'starts', 'with', 'one', 'step', 'journey-context']

# Extract x, y, z coordinates
x_coords = inputs2[:, 0].numpy()
y_coords = inputs2[:, 1].numpy()
z_coords = inputs2[:, 2].numpy()

# Create 3D plot with vectors from origin to each point, using different colors
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define a list of colors for the vectors
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'r']

# Plot each vector with a different color and annotate with the corresponding word
for (x, y, z, word, color) in zip(x_coords, y_coords, z_coords, words2, colors):
    # Draw vector from origin to the point (x, y, z) with specified color and smaller arrow length ratio
    ax.quiver(0, 0, 0, x, y, z, color=color, arrow_length_ratio=0.05)
    ax.text(x, y, z, word, fontsize=10, color=color)

# Set labels for axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set plot limits to keep arrows within the plot boundaries
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_zlim([0, 1])

plt.title('3D Plot of Word Embeddings with Colored Vectors + Context Vector')
plt.show()

# Now, we can extend this computation to calculate attention weights and context vectors for all inputs.
# First, we add an additional for-loop to compute the dot products for all pairs of inputs.

attn_scores = torch.empty(6, 6)

for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)

print(attn_scores)

# Each element in the preceding tensor represents an attention score between each pair of inputs.
# When computing the preceding attention score tensor, we used for-loops in Python.
# However, for-loops are generally slow, and we can achieve the same results using matrix multiplication:

attn_scores = inputs @ inputs.T
print(attn_scores)

# We now normalize each row so that the values in each row sum to 1:

attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)

# In the context of using PyTorch, the dim parameter in functions like torch.softmax specifies
# the dimension of the input tensor along which the function will be computed.

# By setting dim=-1, we are instructing the softmax function to apply the normalization along the last
# dimension of the attn_scores tensor.

# If attn_scores is a 2D tensor (for example, with a shape of [rows, columns]), dim=-1 will normalize across the columns so that the values in
# each row (summing over the column dimension) sum up to 1.

# Let's briefly verify that the rows indeed all sum to 1:

row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
print("Row 2 sum:", row_2_sum)
print("All row sums:", attn_weights.sum(dim=-1))

# In the third and last step, we now use these attention weights to compute all context vectors via matrix multiplication:

all_context_vecs = attn_weights @ inputs
print(all_context_vecs)

# We can double-check that the code is correct by comparing the 2nd row with the context vector z(2) calculated previously

print("Previous 2nd context vector:", context_vec_2)

# Based on the result, we can see that the previously calculated context_vec_2 matches the second row in the previous tensor exactly

##############################################################################################################################
## Step 4: IMPLEMENTING SELF ATTENTION WITH TRAINABLE WEIGHTS

import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

# Let's begin by defining a few variables:
# A The second input element
# B The input embedding size, d=3
# C The output embedding size, d_out=2

x_2 = inputs[1] #A
d_in = inputs.shape[1] #B
d_out = 2 #C

# Note that in GPT-like models, the input and output dimensions are usually the same.
# But for illustration purposes, to better follow the computation, we choose different input (d_in=3) and output (d_out=2) dimensions
# Next, we initialize the three weight matrices Wq, Wk and Wv

torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

print(W_query)
print(W_key)
print(W_value)

# Note that we are setting requires_grad=False to reduce clutter in the outputs for illustration purposes.
# If we were to use the weight matrices for model training, we would set requires_grad=True to update these matrices during model training.
# Next, we compute the query, key, and value vectors as shown earlier

query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
print(query_2)

# As we can see based on the output for the query, this results in a 2-dimensional vector.

# This is because: we set the number of columns of the corresponding weight matrix, via d_out, to 2:
# Even though our temporary goal is to only compute the one context vector z(2), we still
# require the key and value vectors for all input elements.
# This is because they are involved in computing the attention weights with respect to the query q(2)

# We can obtain all keys and values via matrix multiplication:

keys = inputs @ W_key
values = inputs @ W_value
queries = inputs @ W_query
print("keys.shape:", keys.shape)
print("values.shape:", values.shape)
print("queries.shape:", queries.shape)

# As we can tell from the outputs, we successfully projected the 6 input tokens from a 3D onto a 2D embedding space:
# First, let's compute the attention score ω22

keys_2 = keys[1] #A
attn_score_22 = query_2.dot(keys_2)
print(attn_score_22)

# Again, we can generalize this computation to all attention scores via matrix multiplication:

attn_scores_2 = query_2 @ keys.T # All attention scores for given query
print(attn_scores_2)

attn_scores = queries @ keys.T # omega
print(attn_scores)

# We compute the attention weights by scaling the attention scores using the softmax function we used earlier.
# The difference to earlier is that we now scale the attention scores by dividing them by the square root of the embedding dimension of the keys.

d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print(attn_weights_2)
print(d_k)

# WHY DIVIDE BY SQRT (DIMENSION)
# Reason 1: For stability in learning
# The softmax function is sensitive to the magnitudes of its inputs. When the inputs are large, the differences between the exponential values of each input become much more pronounced.
# This causes the softmax output to become "peaky," where the highest value receives almost all the probability mass, and the rest receive very little.
# In attention mechanisms, particularly in transformers, if the dot products between query and key vectors become too large (like multiplying by 8 in this example), 
# the attention scores can become very large. This results in a very sharp softmax distribution, making the model overly confident in one particular "key." 
# Such sharp distributions can make learning unstable
# Reason 2: To make the variance of the dot product stable
# The dot product of  Q and K increases the variance because multiplying two random numbers increases the variance.
# The increase in variance grows with the dimension.
# Dividing by sqrt (dimension) keeps the variance close to 1

import torch

# Define the tensor
tensor = torch.tensor([0.1, -0.2, 0.3, -0.2, 0.5])

# Apply softmax without scaling
softmax_result = torch.softmax(tensor, dim=-1)
print("Softmax without scaling:", softmax_result)

# Multiply the tensor by 8 and then apply softmax
scaled_tensor = tensor * 8
softmax_scaled_result = torch.softmax(scaled_tensor, dim=-1)
print("Softmax after scaling (tensor * 8):", softmax_scaled_result)

import numpy as np

# Function to compute variance before and after scaling
def compute_variance(dim, num_trials=1000):
    dot_products = []
    scaled_dot_products = []

    # Generate multiple random vectors and compute dot products
    for _ in range(num_trials):
        q = np.random.randn(dim)
        k = np.random.randn(dim)

        # Compute dot product
        dot_product = np.dot(q, k)
        dot_products.append(dot_product)

        # Scale the dot product by sqrt(dim)
        scaled_dot_product = dot_product / np.sqrt(dim)
        scaled_dot_products.append(scaled_dot_product)

    # Calculate variance of the dot products
    variance_before_scaling = np.var(dot_products)
    variance_after_scaling = np.var(scaled_dot_products)

    return variance_before_scaling, variance_after_scaling

# For dimension 5
variance_before_5, variance_after_5 = compute_variance(5)
print(f"Variance before scaling (dim=5): {variance_before_5}")
print(f"Variance after scaling (dim=5): {variance_after_5}")

# For dimension 20
variance_before_100, variance_after_100 = compute_variance(100)
print(f"Variance before scaling (dim=100): {variance_before_100}")
print(f"Variance after scaling (dim=100): {variance_after_100}")

# We now compute the context vector as a weighted sum over the value vectors.
# Here, the attention weights serve as a weighting factor that weighs the respective importance of each value vector.
# We can use matrix multiplication to obtain the output in one step:

context_vec_2 = attn_weights_2 @ values
print(context_vec_2)

# So far, we only computed a single context vector, z(2).
# In the next section, we will generalize the code to compute all context vectors in the input sequence, z(1)to z (T)

##############################################################################################################################
## Step 5: IMPLEMENTING A COMPACT SELF ATTENTION PYTHON CLASS

# In the previous sections, we have gone through a lot of steps to compute the self-attention outputs.

# This was mainly done for illustration purposes so we could go through one step at a time.
# In practice, with the LLM implementation in the next chapter in mind, it is helpful to organize this code into a Python class as follows:

import torch.nn as nn

class SelfAttention_v1(nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value

        attn_scores = queries @ keys.T # omega
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )

        context_vec = attn_weights @ values
        return context_vec

# In this PyTorch code, SelfAttention_v1 is a class derived from nn.Module, which is a
# fundamental building block of PyTorch models, which provides necessary functionalities for
# model layer creation and management.

# The __init__ method initializes trainable weight matrices (W_query, W_key, and
# W_value) for queries, keys, and values, each transforming the input dimension d_in to an
# output dimension d_out.

# During the forward pass, using the forward method, we compute the attention scores
# by multiplying queries and keys, normalizing these scores using softmax.

# Finally, we create a context vector by weighting the values with these normalized attention weights.

torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print(sa_v1(inputs))

# Since inputs contains six embedding vectors, we get a matrix storing the six
# context vectors, as shown in the above result.

# We can improve the SelfAttention_v1 implementation further by utilizing PyTorch's
# nn.Linear layers, which effectively perform matrix multiplication when the bias units are disabled.

# Additionally, a significant advantage of using nn.Linear instead of manually
# implementing nn.Parameter(torch.rand(...)) is that nn.Linear has an optimized weight
# initialization scheme, contributing to more stable and effective model training.

class SelfAttention_v2(nn.Module):

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        context_vec = attn_weights @ values
        return context_vec

# You can use the SelfAttention_v2 similar to SelfAttention_v1:

torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
print(sa_v2(inputs))

##############################################################################################################################
## Step 6: HIDING FUTURE WORDS WITH CAUSAL ATTENTION

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

queries = sa_v2.W_query(inputs) #A
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=1)
print(attn_weights)

# We can now use PyTorch's tril function to create a mask where the values above the diagonal are zero:

torch.ones(context_length, context_length)

context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
print(mask_simple)

# Now, we can multiply this mask with the attention weights to zero out the values above the diagonal:

masked_simple = attn_weights*mask_simple
print(masked_simple)

# As we can see, the elements above the diagonal are successfully zeroed out
# The third step is to renormalize the attention weights to sum up to 1 again in each row.
# We can achieve this by dividing each element in each row by the sum in each row:

row_sums = masked_simple.sum(dim=1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
print(masked_simple_norm)

# The result is an attention weight matrix where the attention weights above the diagonal are
# zeroed out and where the rows sum to 1.

# While we could be technically done with implementing causal attention at this point, we can
# take advantage of a mathematical property of the softmax function.
# We can implement the computation of the masked attention weights more efficiently in fewer steps.
# The softmax function converts its inputs into a probability distribution.
# When negative infinity values (-∞) are present in a row, the softmax function treats them as zero probability.
# (Mathematically, this is because e-∞ approaches 0.)
# We can implement this more efficient masking "trick" by creating a mask with 1's above
# the diagonal and then replacing these 1's with negative infinity (-inf) values:

print(attn_scores)

torch.triu(torch.ones(context_length, context_length))

mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
print(mask)

mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print(masked)

# Now, all we need to do is apply the softmax function to these masked results, and we are done.

attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1)
print(attn_weights)

# As we can see based on the output, the values in each row sum to 1, and no further normalization is necessary.
# Masking in Transformers sets scores for future tokens to a large negative value, making their influence in the softmax calculation effectively zero.
# The softmax function then recalculates attention weights only among the unmasked tokens.
# This process ensures no information leakage from masked tokens, focusing the model solely on the intended data.
# We could now use the modified attention weights to compute the context vectors via context_vec = attn_weights @ values.

### MASKING ADDITIONAL ATTENTION WEIGHTS WITH DROPOUT

# Dropout rate of 50%, which means masking out half of the attention weights.

# When we train the GPT model in later chapters, we will use a lower dropout rate, such as 0.1 or 0.2.

example = torch.ones(6, 6) #B
print(example)

torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5) #A
example = torch.ones(6, 6) #B
print(dropout(example))

# When applying dropout to an attention weight matrix with a rate of 50%, half of the elements in the matrix are randomly set to zero.
# To compensate for the reduction in active elements, the values of the remaining elements in the matrix are scaled up by a factor of 1/0.5 =2.

# This scaling is crucial to maintain the overall balance of the attention weights,
# ensuring that the average influence of the attention mechanism remains consistent during both the training and inference phases.

# Now, let's apply dropout to the attention weight matrix itself:

torch.manual_seed(123)
print(dropout(attn_weights))

# As we can see above, the resulting attention weight matrix now has additional elements zeroed out and the remaining ones rescaled.

# Having gained an understanding of causal attention and dropout masking, we will develop a concise Python class in the following section.

# This class is designed to facilitate the efficient application of these two techniques.

### IMPLEMENTING A COMPACT CAUSAL ATTENTION CLASS

# In this section, we will now incorporate the causal attention and dropout modifications into
# the SelfAttention Python class we developed in section 3.4.

# This class will then serve as a template for developing multi-head attention in the upcoming section.
# Before we begin, one more thing is to ensure that the code can handle batches consisting of more than one input.

# This will ensure that the CausalAttention class supports the batch outputs produced by the data loader we implemented earlier.

# For simplicity, to simulate such batch inputs, we duplicate the input text example:
# 2 inputs with 6 tokens each, and each token has embedding dimension 3

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)
batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape)

# This results in a 3D tensor consisting of 2 input texts with 6 tokens each, where each token is a 3-dimensional embedding vector.

# The following CausalAttention class is similar to the SelfAttention class we
# implemented earlier, except that we now added the dropout and causal mask components as highlighted in the following code.

# Step 1: Compared to the previous SelfAttention class, we added a dropout layer.
# Step 2: The register_buffer call is also a new addition
# Step 3: We transpose dimensions 1 and 2, keeping the batch dimension at the first position (0).
# Step 4: In PyTorch, operations with a trailing underscore are performed in-place, avoiding unnecessary memory copies

class CausalAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length,
                 dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout) # New
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1)) # New

    def forward(self, x):
        b, num_tokens, d_in = x.shape # New batch dimension b
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2) # Changed transpose
        attn_scores.masked_fill_(  # New, _ ops are in-place
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)  # `:num_tokens` to account for cases where the number of tokens in the batch is smaller than the supported context_size
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights) # New

        context_vec = attn_weights @ values
        return context_vec

# The use of register_buffer in PyTorch is not strictly necessary for all use cases but offers several advantages here.

# For instance, when we use the CausalAttention class in our LLM, buffers are automatically
# moved to the appropriate device (CPU or GPU) along with our model, which will be relevant when training the LLM in future chapters.
# This means we don't need to manually ensure these tensors are on the same device as your model parameters, avoiding device mismatch errors.
# We can use the CausalAttention class as follows, similar to SelfAttention previously:

print(d_in)
print(d_out)

torch.manual_seed(123)
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)
print("context_vecs.shape:", context_vecs.shape)

print(context_vecs)

# As we can see, the resulting context vector is a 3D tensor where each token is now represented by a 2D embedding:

# In the next section, we will expand on this concept and implement a multi-head attention module, that implements several of such causal
# attention mechanisms in parallel.

##############################################################################################################################
## Step 7: EXTENDING SINGLE HEAD ATTENTION TO MULTI-HEAD ATTENTION

# In practical terms, implementing multi-head attention involves creating multiple instances
# of the self-attention mechanism, each with its own weights, and then combining their outputs

# In code, we can achieve this by implementing a simple MultiHeadAttentionWrapper
# class that stacks multiple instances of our previously implemented CausalAttention module:

class MultiHeadAttentionWrapper(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
             for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

# For example, if we use this MultiHeadAttentionWrapper class with two attention heads (via
# num_heads=2) and CausalAttention output dimension d_out=2, this results in a 4-dimensional context vectors (d_out*num_heads=4)

# To illustrate further with a concrete example, we can use the
# MultiHeadAttentionWrapper class similar to the CausalAttention class before:

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)
batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape)

torch.manual_seed(123)
context_length = batch.shape[1] # This is the number of tokens = 6
d_in, d_out = 3, 2
mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)

# The first dimension of the resulting context_vecs tensor is 2 since we have two input texts
# (the input texts are duplicated, which is why the context vectors are exactly the same for those).

# The second dimension refers to the 6 tokens in each input. The third dimension
# refers to the 4-dimensional embedding of each token.

# In this section, we implemented a MultiHeadAttentionWrapper that combined multiple single-head attention modules.

# However, note that these are processed sequentially via [head(x) for head in self.heads] in the forward method.
# We can improve this implementation by processing the heads in parallel.
# One way to achieve this is by computing the outputs for all attention heads simultaneously via matrix multiplication, as
# we will explore in the next section.

### IMPLEMENTING MULTI-HEAD ATTENTION WITH WEIGHT SPLITS

# Instead of maintaining two separate classes, MultiHeadAttentionWrapper and
# CausalAttention, we can combine both of these concepts into a single MultiHeadAttention class.

# In the MultiHeadAttentionWrapper, multiple heads are implemented by creating a list
# of CausalAttention objects (self.heads), each representing a separate attention head.

# The CausalAttention class independently performs the attention mechanism, and the
# results from each head are concatenated.

# In contrast, the following MultiHeadAttention class integrates the multi-head functionality within a single class.

# It splits the input into multiple heads by reshaping the projected query, key, and value tensors and then combines
# the results from these heads after computing attention.

# Let's take a look at the MultiHeadAttention class before we discuss it further:

import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x) # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) # optional projection

        return context_vec

# Step 1: Reduce the projection dim to match desired output dim
# Step 2: Use a Linear layer to combine head outputs
# Step 3: Tensor shape: (b, num_tokens, d_out)
# Step 4: We implicitly split the matrix by adding a `num_heads` dimension. Then we unroll last dim: (b,
# num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
# Step 5: Transpose from shape (b, num_tokens, num_heads, head_dim) to (b, num_heads, num_tokens, head_dim)
# Step 6: Compute dot product for each head
# Step 7: Mask truncated to the number of tokens
# Step 8: Use the mask to fill attention scores
# Step 9: Context tensor shape: (b, num_tokens, n_heads, head_dim)
# Step 10: Combine heads, where self.d_out = self.num_heads * self.head_dim
# Step 11: Add an optional linear projection

# Even though the reshaping (.view) and transposing (.transpose) of tensors inside the
# MultiHeadAttention class looks very complicated, mathematically, the
# MultiHeadAttention class implements the same concept as the
# MultiHeadAttentionWrapper earlier.

# On a big-picture level, in the previous MultiHeadAttentionWrapper, we stacked
# multiple single-head attention layers that we combined into a multi-head attention layer.

# The MultiHeadAttention class takes an integrated approach.

# It starts with a multi-head layer and then internally splits this layer into individual attention heads

#### DETAILED EXPLANATION OF THE MULTI-HEAD ATTENTION CLASS

# The splitting of the query, key, and value tensors, is achieved
# through tensor reshaping and transposing operations using PyTorch's .view and .transpose methods.

# The input is first transformed (via linear layers for queries, keys, and
# values) and then reshaped to represent multiple heads.

# The key operation is to split the d_out dimension into num_heads and head_dim, where head_dim = d_out / num_heads.

# This splitting is then achieved using the .view method: a
# tensor of dimensions (b, num_tokens, d_out) is reshaped to dimension (b, num_tokens, num_heads, head_dim).

# The tensors are then transposed to bring the num_heads dimension before the
# num_tokens dimension, resulting in a shape of (b, num_heads, num_tokens, head_dim).

# This transposition is crucial for correctly aligning the queries, keys, and values across the
# different heads and performing batched matrix multiplications efficiently.

# Continuing with MultiHeadAttention, after computing the attention weights and context
# vectors, the context vectors from all heads are transposed back to the shape (b,
# num_tokens, num_heads, head_dim).

# These vectors are then reshaped (flattened) into the
# shape (b, num_tokens, d_out), effectively combining the outputs from all heads

# Additionally, we added a so-called output projection layer (self.out_proj) to
# MultiHeadAttention after combining the heads, which is not present in the CausalAttention class.

# This output projection layer is not strictly necessary, but it is commonly used in many LLM
# architectures, which is why we added it here for completeness.

# Even though the MultiHeadAttention class looks more complicated than the
# MultiHeadAttentionWrapper due to the additional reshaping and transposition of tensors, it is more efficient.

# The reason is that we only need one matrix multiplication to compute
# the keys, for instance, keys = self.W_key(x) (the same is true for the queries and values).

# In the MultiHeadAttentionWrapper, we needed to repeat this matrix multiplication,
# which is computationally one of the most expensive steps, for each attention head.

# The MultiHeadAttention class can be used similar to the SelfAttention and
# CausalAttention classes we implemented earlier:

torch.manual_seed(123)

# Define the tensor with 3 rows and 6 columns
inputs = torch.tensor(
    [[0.43, 0.15, 0.89, 0.55, 0.87, 0.66],  # Row 1
     [0.57, 0.85, 0.64, 0.22, 0.58, 0.33],  # Row 2
     [0.77, 0.25, 0.10, 0.05, 0.80, 0.55]]  # Row 3
)

batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape)

batch_size, context_length, d_in = batch.shape
d_out = 6
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)

# As we can see based on the results, the output dimension is directly controlled by the d_out argument:

# Note that while the code is fully functional, we used relatively small embedding sizes and numbers of attention heads to keep the outputs readable.
# For comparison, the smallest GPT-2 model (117 million parameters) has 12 attention heads and a context vector embedding size of 768.
# The largest GPT-2 model (1.5 billion parameters) has 25 attention heads and a context vector embedding size of 1600.