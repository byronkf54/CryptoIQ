import pandas as pd
import math
import numpy as np
from sklearn.metrics import classification_report
import torch
from tqdm import tqdm, trange
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
from sklearn.model_selection import train_test_split
from transformers import (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer)
import os

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
torch.cuda.empty_cache()

# ## Load data

# **Load CSV data**
data_file_address = "/app/datasets/Tweets/english_tweets.csv"
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
# d.rename(columns={"target":"labels"}, inplace=True)
# d.drop("id", inplace=True, axis=1)
# d.to_csv("../datasets/training.1600000.processed.noemoticon.csv", index=False)

# negative_data = pd.read_csv(data_file_address,sep=",",encoding="utf-8").head(5000)# .tail(2500)
# positive_data = pd.read_csv(data_file_address,sep=",",encoding="utf-8").tail(5000)# .head(2500)
# df_data = pd.concat([negative_data, positive_data], ignore_index=True)
samples = 15
df_data = pd.read_csv(data_file_address, sep=",", encoding="utf-8")  # .head(samples)
# df_data['labels'] = df_data['compound']
neutral = [0]

# print(df_data.username.unique())
# userids = df_data.username.unique().tolist()
# with open("user_ids.txt", "w") as f:
#     for uid in userids:
#         f.write(str(uid) + "\n")


def add_labels(row):
    # return row['labels'] + 1
    if row['compound'] > 0:  # positive
        return 1
    elif row['compound'] < 0:  # negative
        return 0
    else:
        neutral[0] += 1
        return 2


# df_data = df_data[~df_data['labels'].isin([2])]
# df_data = df_data[~df_data['language'].isin(["ca"])]
# print(df_data.language.unique())
df_data['labels'] = df_data.apply(add_labels, axis=1)  # -1 (neg), 0 (neu), 1 (pos)
df_data = df_data[~df_data['labels'].isin([2])]

df_pos = df_data.loc[df_data['labels'] == 1]
df_neg = df_data.loc[df_data['labels'] == 0]

df_pos_sampled = df_pos.sample(n=len(df_neg))  # balance classes

df_data = pd.concat([df_pos_sampled, df_neg], axis=0, ignore_index=True)

print(df_data.columns)

print(df_data.tail(n=50))

# **Have a look at labels**
print(df_data.labels.unique())

# Analyse the labels distribution
print(df_data.labels.value_counts())
# exit()

# ## Parser data
# **Parser data into document structure**

# Get sentence data
sentences = df_data.decoded_tweet.to_list()
print(sentences[0])

# Get tag labels data
labels = df_data.labels.to_numpy()
print(labels[0])

# **Make TAG name into index for training**
# Set a dict for mapping id to tag name
# tag2idx = {t: i for i, t in enumerate(tags_vals)}

# Recommend to set it by manual define, good for reusing
# 0: negative, 1: neutral, 2:positive
tag2idx = {'0': 0, '1': 1}

print(tag2idx)

# Mapping index to name
tag2name = {tag2idx[key]: key for key in tag2idx.keys()}
print(tag2name)

# ## Make training data

# Make raw data into trainable data for XLNet, including:

# - Set gpu environment
# - Load tokenizer and tokenize
# - Set 3 embedding, token embedding, mask word embedding, segmentation embedding
# - Split data set into train and validate, then send them to dataloader

# **Set up gpu environment**

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
n_gpu = torch.cuda.device_count()

# ### Load tokenizer
# Remember to install sentencepiece with  'pip install sentencepiece'
# Manual define vocabulary address, if you download the model in local
# The vocabulary can download from "https://s3.amazonaws.com/models.huggingface.co/bert/xlnet-base-cased-spiece.model"
vocabulary = '/app/Models/SA_models/xlnet-base-cased-spiece.model'

# Len of the sentence must be the same as the training model
# Tweet max length = 280 (allow for CLS and SEQ tokens)
max_len = 282

# With cased model, set do_lower_case = False
tokenizer = XLNetTokenizer(vocab_file=vocabulary, do_lower_case=True)


# ### Set text input embedding
# - token id embedding
# - mask embedding
# - segment embedding

# The Embedding process was referred to [XLNet official repo](https://github.com/zihangdai/xlnet/blob/master/classifier_utils.py)

# **This process is huge different from BERT**
def text_embedding(text_sentences):
    all_input_ids = []
    all_input_masks = []
    all_segment_ids = []

    SEG_ID_A = 0
    SEG_ID_CLS = 2
    SEG_ID_PAD = 4

    CLS_ID = tokenizer.encode("<cls>")[0]
    SEP_ID = tokenizer.encode("<sep>")[0]

    for i, sentence in enumerate(text_sentences):
        # Tokenize sentence to token id list
        tokens_a = tokenizer.encode(sentence)

        # Trim the len of text
        if len(tokens_a) > max_len - 2:
            tokens_a = tokens_a[:max_len - 2]

        tokens = []
        segment_ids = []

        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(SEG_ID_A)

        # Add <sep> token
        tokens.append(SEP_ID)
        segment_ids.append(SEG_ID_A)

        # Add <cls> token
        tokens.append(CLS_ID)
        segment_ids.append(SEG_ID_CLS)

        input_ids = tokens

        # The mask has 0 for real tokens and 1 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [0] * len(input_ids)

        # Zero-pad up to the sequence length at front
        if len(input_ids) < max_len:
            delta_len = max_len - len(input_ids)
            input_ids = [0] * delta_len + input_ids
            input_mask = [1] * delta_len + input_mask
            segment_ids = [SEG_ID_PAD] * delta_len + segment_ids

        assert len(input_ids) == max_len
        assert len(input_mask) == max_len
        assert len(segment_ids) == max_len

        all_input_ids.append(input_ids)
        all_input_masks.append(input_mask)
        all_segment_ids.append(segment_ids)

    return all_input_ids, all_input_masks, all_segment_ids


all_input_ids, all_input_masks, all_segment_ids = text_embedding(sentences)

# ### Set label embedding
# Make label into id
tags = [tag2idx[str(lab)] for lab in labels]
# print(tags)


# ## Split data into train and validate

# 70% for training, 30% for validation

# **Split all data**
tr_inputs, val_inputs, tr_tags, val_tags, tr_masks, val_masks, tr_segs, val_segs = train_test_split(all_input_ids,
                                                                                                    tags,
                                                                                                    all_input_masks,
                                                                                                    all_segment_ids,
                                                                                                    random_state=4,
                                                                                                    test_size=0.05,
                                                                                                    shuffle=True)

print(len(tr_inputs), len(val_inputs), len(tr_segs), len(val_segs))

# **Set data into tensor**
# Not recommend tensor.to(device) at this process, since it will run out of GPU memory
tr_inputs = torch.tensor(tr_inputs)
tr_tags = torch.tensor(tr_tags)
tr_masks = torch.tensor(tr_masks)
tr_segs = torch.tensor(tr_segs)

val_inputs = torch.tensor(val_inputs)
val_tags = torch.tensor(val_tags)
val_masks = torch.tensor(val_masks)
val_segs = torch.tensor(val_segs)

# **Put data into data loader**
# Set batch num
batch_num = 16

# Set token embedding, attention embedding, segment embedding
train_data = TensorDataset(tr_inputs, tr_masks, tr_segs, tr_tags)
train_sampler = RandomSampler(train_data)
# Drop last can make batch training better for the last one
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_num)  # , drop_last=True

valid_data = TensorDataset(val_inputs, val_masks, val_segs, val_tags)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=batch_num)

# ## Train model
# **Load pre-trained XLNet model**
# In this document, contain config(txt) and weight(bin) files
# The folder must contain: pytorch_model.bin, config.json
# model_file_address = '../Models/SA_Models/'
# xlnet_out_address = '../Models/SA_models/20000samples-3e'
model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels=len(tag2idx))
# print(model)

# Set model to GPU,if you are using GPU machine
model.to(device)
torch.cuda.empty_cache()

# Add multi GPU support
if n_gpu > 1:
    model = torch.nn.DataParallel(model)

# Set epoch and grad max num
epochs = 8
max_grad_norm = 1.0

# Calculate train optimization num
num_train_optimization_steps = int(math.ceil(len(tr_inputs) / batch_num) / 1) * epochs

# ### Set fine tuning method

# **Manual optimizer**
# True: fine tuning all the layers
# False: only fine tuning the classifier layers
# Since XLNet in 'pytorch_transformer' did not contain classifier layers
# all_FINETUNING = True need to set True
all_FINETUNING = True

if all_FINETUNING:
    # Fine tune model all layer parameters
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
else:
    # Only fine tune classifier parameters
    param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
optimizer = Adam(optimizer_grouped_parameters, lr=2e-5)

# ### Fine-tuning model
# TRAIN loop
model.train()


def save_XLNet_model():
    xlnet_out_address = f'/app/Models/SA_models/final_model-{epochs}e'

    # In[ ]:

    # Make dir if not exits
    if not os.path.exists(xlnet_out_address):
        os.makedirs(xlnet_out_address)

    # In[ ]:

    # Save a trained model, configuration and tokenizer
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

    # In[ ]:

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(xlnet_out_address, "pytorch_model.bin")
    output_config_file = os.path.join(xlnet_out_address, "config.json")

    # In[ ]:

    # Save model into file
    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(xlnet_out_address)


save_XLNet_model()

print("***** Running training *****")
print("  Num examples = %d" % (len(tr_inputs)))
print("  Batch size = %d" % batch_num)
print("  Num steps = %d" % num_train_optimization_steps)
for _ in trange(epochs, desc="Epoch"):
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(train_dataloader):
        # add batch to gpu
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_segs, b_labels = batch

        # forward pass
        outputs = model(input_ids=b_input_ids, token_type_ids=b_segs, input_mask=b_input_mask, labels=b_labels)

        loss, logits = outputs[:2]
        if n_gpu > 1:
            # When multi gpu, average it
            loss = loss.mean()

        # backward pass
        loss.backward()

        # track train loss
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)

        # update parameters
        optimizer.step()
        optimizer.zero_grad()

    # print train loss per epoch
    print("Train loss: {}".format(tr_loss/nb_tr_steps))
    save_XLNet_model()


# ## Save model
# save_XLNet_model()


# ## Load model
xlnet_out_address = '/app/Models/SA_models/final_model-8e'
model = XLNetForSequenceClassification.from_pretrained(xlnet_out_address, num_labels=len(tag2idx))

# In[ ]:


# Set model to GPU
model.to(device)

# In[ ]:


if n_gpu > 1:
    model = torch.nn.DataParallel(model)

# ## Eval model

# In[ ]:


# # Testing own text
# all_input_ids, all_input_masks, all_segment_ids = text_embedding(["The food was fantastic but the service was quite bad"])
#
# val_inputs = torch.tensor(all_input_ids)
# val_tags = torch.tensor(tags)
# val_masks = torch.tensor(all_input_masks)
# val_segs = torch.tensor(all_segment_ids)
# batch_num = 32
# valid_data = TensorDataset(val_inputs, val_masks, val_segs, val_tags)
# valid_sampler = SequentialSampler(valid_data)
# valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=batch_num)

# Evaluate loop
model.eval()


# Set acc function
def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


eval_loss, eval_accuracy = 0, 0
nb_eval_steps, nb_eval_examples = 0, 0

y_true = []
y_predict = []
print("***** Running evaluation *****")
print("  Num examples = {}".format(len(val_inputs)))
print("  Batch size = {}".format(batch_num))
for step, batch in enumerate(valid_dataloader):
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_segs, b_labels = batch

    with torch.no_grad():
        outputs = model(input_ids=b_input_ids, token_type_ids=b_segs, input_mask=b_input_mask, labels=b_labels)
        tmp_eval_loss, logits = outputs[:2]

    # Get text classification predict result
    logits = logits.detach().cpu().numpy()  # PREDICTED VAL
    label_ids = b_labels.to('cpu').numpy()
    tmp_eval_accuracy = accuracy(logits, label_ids)
    # print(tmp_eval_accuracy)
    # print(np.argmax(logits, axis=1))
    # print(label_ids)

    # Save predict and real label result for analyze
    for predict in np.argmax(logits, axis=1):
        y_predict.append(predict)

    for real_result in label_ids.tolist():
        y_true.append(real_result)

    eval_loss += tmp_eval_loss.mean().item()
    eval_accuracy += tmp_eval_accuracy

    nb_eval_steps += 1

eval_loss /= nb_eval_steps
eval_accuracy /= len(val_inputs)
# loss = tr_loss/nb_tr_steps
result = {'eval_loss': eval_loss, 'eval_accuracy': eval_accuracy}
# 'loss': loss}
report = classification_report(y_pred=np.array(y_predict), y_true=np.array(y_true))

# Save the report into file
output_eval_file = os.path.join(xlnet_out_address, "eval_results.txt")
with open(output_eval_file, "w") as writer:
    print("***** Eval results *****")
    for key in sorted(result.keys()):
        print("  %s = %s" % (key, str(result[key])))
        writer.write("%s = %s\n" % (key, str(result[key])))

    print(report)
    writer.write("\n\n")
    writer.write(report)
