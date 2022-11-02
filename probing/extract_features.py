"""
reference:
https://github.com/ganeshjawahar/interpret_bert/blob/master/probing/extract_features.py

extract bert features for sentence level probing tasks 
(10 of them as defined by Conneau et al.)
"""

import collections
import argparse
from tqdm import tqdm
import json
import os

import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import RobertaTokenizerFast
from transformers import AutoConfig, AutoModel

class InputExample(object):
  def __init__(self, unique_id, text):
    self.unique_id = unique_id
    self.text = text

class InputFeatures(object):
  """A single set of features of data."""
  def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
    self.unique_id = unique_id
    self.tokens = tokens
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.input_type_ids = input_type_ids

def read_examples(input_file):
  """Read a list of `InputExample`s from an input file."""
  examples = []
  unique_id = 0
  with open(input_file, "r", encoding='utf-8') as reader:
    while True:
      line = reader.readline()
      if not line:
          break
      text = line.strip().split('\t')[-1]
      examples.append(
          InputExample(unique_id=unique_id, text=text))
      unique_id += 1
  return examples

def convert_examples_to_features(examples, seq_length, tokenizer):
  """Loads a data file into a list of `InputBatch`s."""
  features = []
  for (ex_index, example) in enumerate(examples):
    cand_tokens = tokenizer.tokenize(example.text)
    # Account for [CLS] and [SEP] with "- 2"
    if len(cand_tokens) > seq_length - 2:
      cand_tokens = cand_tokens[0:(seq_length - 2)]

    tokens = []
    input_type_ids = []
    tokens.append("[CLS]")
    input_type_ids.append(0)
    for token in cand_tokens:
      tokens.append(token)
      input_type_ids.append(0)
    tokens.append("[SEP]")
    input_type_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < seq_length:
      input_ids.append(0)
      input_mask.append(0)
      input_type_ids.append(0)

    assert len(input_ids) == seq_length
    assert len(input_mask) == seq_length
    assert len(input_type_ids) == seq_length

    features.append(
      InputFeatures(
          unique_id=example.unique_id,
          tokens=tokens,
          input_ids=input_ids,
          input_mask=input_mask,
          input_type_ids=input_type_ids))
  return features

def get_max_seq_length(instances, tokenizer):
  max_seq_len = -1
  for instance in instances:
    cand_tokens = tokenizer.tokenize(' '.join(instance.text))
    cur_len = len(cand_tokens)
    if cur_len > max_seq_len:
      max_seq_len = cur_len
  return max_seq_len

def save(args, model, tokenizer, device):
  # convert data to ids
  examples = read_examples(args.data_file)
  features = convert_examples_to_features(
        examples=examples, seq_length=2+get_max_seq_length(examples, tokenizer), tokenizer=tokenizer)

  # extract and write features
  all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
  all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
  all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
  eval_data = TensorDataset(all_input_ids, all_input_mask, all_example_index)
  eval_sampler = SequentialSampler(eval_data)
  eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size)

  pbar = tqdm(total=len(examples)//args.batch_size)

#  if not os.path.exists(args.output_dir):
#    os.makedirs(args.output_dir)

  with open(os.path.join(args.output_dir, args.output_file), "w", encoding='utf-8') as writer:
  #with open(args.output_file, "w", encoding='utf-8') as writer:
    for input_ids, input_mask, example_indices in eval_dataloader:
      input_ids = input_ids.to(device)
      input_mask = input_mask.to(device)
      #all_encoder_layers, _ = model(input_ids, token_type_ids=None, attention_mask=input_mask)
      all_encoder_layers = model(input_ids, token_type_ids=None, attention_mask=input_mask)[2]
      for b, example_index in enumerate(example_indices):
        feature = features[example_index.item()]
        unique_id = int(feature.unique_id)
        output_json = collections.OrderedDict()
        output_json["linex_index"] = unique_id
        all_out_features = []
        for (i, token) in enumerate(feature.tokens):
          all_layers = []
          for layer_index in range(len(all_encoder_layers)):
            layer_output = all_encoder_layers[int(layer_index)].detach().cpu().numpy()
            layer_output = layer_output[b]
            layers = collections.OrderedDict()
            layers["index"] = layer_index
            layers["values"] = [
                round(x.item(), 6) for x in layer_output[i]
            ]
            all_layers.append(layers)
          out_features = collections.OrderedDict()
          out_features["token"] = token
          out_features["layers"] = all_layers
          all_out_features.append(out_features)
          break
        output_json["features"] = all_out_features
        writer.write(json.dumps(output_json) + "\n")
      pbar.update(1)
  pbar.close()
  print('written features to %s'%(args.output_file))

total = 0
def init_weights(m):
  global total
  if type(m) == torch.nn.Linear:
    torch.nn.init.xavier_uniform(m.weight)
    m.bias.data.fill_(0.01)
    total += m.weight.size(0)*m.weight.size(1)
    total += m.bias.size(0)
  elif type(m) == torch.nn.Embedding:
    torch.nn.init.xavier_uniform(m.weight)
    total += m.weight.size(0)*m.weight.size(1)
  elif hasattr(m, 'weight') and hasattr(m, 'bias'):
    total += m.weight.size(0)
    total += m.bias.size(0)

def load(args):
  print('loading %s model'%args.bert_model)
  device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
  tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

  config = AutoConfig.from_pretrained(
        args.bert_model,
        output_hidden_states=True
    )
  model = AutoModel.from_pretrained(
        args.bert_model,
        from_tf=bool(".ckpt" in args.bert_model),
        config=config
    )

  model.to(device)
  if args.num_gpus > 1:
    model = torch.nn.DataParallel(model)
  if args.untrained_bert:
    model.apply(init_weights)
  model.eval()
  return model, tokenizer, device

def main():
  parser = argparse.ArgumentParser()

  parser.add_argument("--data_file",
                      default=None,
                      type=str,
                      required=True,
                      help="path to the data file for probing task from https://github.com/facebookresearch/SentEval/tree/master/data/probing")
  parser.add_argument("--output_file",
                      default=None,
                      type=str,
                      required=True,
                      help="output file where the features will be written")
  parser.add_argument("--output_dir",
                      default=None,
                      type=str,
                      required=True,
                      help="output dir where the features file will be written")
  parser.add_argument("--cache_dir",
                      default="/tmp",
                      type=str,
                      help="directory to cache bert pre-trained models")
  parser.add_argument("--bert_model", 
                      default="bert-base-uncased", 
                      type=str,
                      help="bert pre-trained model selected in the list: bert-base-uncased, "
                      "bert-large-uncased, bert-base-cased, bert-large-cased")
  parser.add_argument("--no_cuda",
                      action='store_true',
                      help="whether not to use CUDA when available")
  parser.add_argument("--batch_size",
                      default=8,
                      type=int,
                      help="total batch size for inference")
  parser.add_argument("--num_gpus",
                      default=1,
                      type=int,
                      help="no. of gpus to use")
  parser.add_argument("--untrained_bert",
                      action='store_true',
                      help="use untrained version of bert")
  
  args = parser.parse_args()
  print(args)
  model, tokenizer, device = load(args)
  save(args, model, tokenizer, device)

if __name__ == "__main__":
  main()