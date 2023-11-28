How does the pre-training objective affect what large language models learn about linguistic properties?
===

This repo contains the pre-trained models and the implementation code for the ACL 2022 paper [How does the pre-training objective affect what large language models learn about linguistic properties?](https://aclanthology.org/2022.acl-short.16/).

## Pre-trained models
The links to the pre-trained models that are uploaded to Hugging Face: 
| Base models | Medium models | Small models |
| --- | ----------- | ----------- |
| [MLM](https://huggingface.co/aajrami/bert-mlm-base) | [MLM](https://huggingface.co/aajrami/bert-mlm-medium) | [MLM](https://huggingface.co/aajrami/bert-mlm-small) | 
| [Shuffle + Random](https://huggingface.co/aajrami/bert-sr-base) | [Shuffle + Random](https://huggingface.co/aajrami/bert-sr-medium) | [Shuffle + Random](https://huggingface.co/aajrami/bert-sr-small) |
| [First Char](https://huggingface.co/aajrami/bert-fc-base) | [First Char](https://huggingface.co/aajrami/bert-fc-medium) | [First Char](https://huggingface.co/aajrami/bert-fc-small) |
| [ASCII](https://huggingface.co/aajrami/bert-ascii-base) | [ASCII](https://huggingface.co/aajrami/bert-ascii-medium) | [ASCII](https://huggingface.co/aajrami/bert-ascii-small) | 
| [Random](https://huggingface.co/aajrami/bert-rand-base) | [Random](https://huggingface.co/aajrami/bert-rand-medium) | [Random](https://huggingface.co/aajrami/bert-rand-small) | 

## Requirements  
* torch
* transformers
* datasets
* scikit-learn
* tensorflow
* spacy

## Pre-training
### 1. Required packages installation
After cloning this repo, the required packages can be installed using the following command:
```
pip install -r requirements.txt
```  
The `en_core_web_sm` model from spaCy is also needed for preprocessing, and it can be installed by running the follwoing `python -m spacy download en_core_web_sm`.

### 2. Datasets pre-processing
The following command pre-process the BookCorpus and the English Wikipedia datasets required for pre-training:
```
cd ./utils
python preprocess_roberta.py --path=/path/to/save/data/
```   

### 3. Models pre-training
The following is an example to pre-train an MLM base model:
```
cd ../
python pretrainer.py \
--data_dir=/path/to/dataset/ \
--do_train \
--learning_rate=1e-4 \
--hidden_size=768 \
--intermediate_size=3072 \
--num_attention_heads=12 \
--num_hidden_layers=12 \
--weight_decay=0.01 \
--adam_epsilon=1e-8 \
--max_grad_norm=1.0 \
--num_train_epochs=10 \
--warmup_steps=10000 \
--save_steps=10000 \
--save_interval=100000 \
--seed=42 \
--per_device_train_batch_size=16 \
--logging_steps=100 \
--output_dir=/path/to/save/weights/ \
--overwrite_output_dir \
--logging_dir=/path/to/save/log/files/ \
--disable_tqdm=True \
--prediction_loss_only \
--fp16 \
--mlm_prob=0.15 \
--pretrain_model=RobertaForMaskedLM
```
* You can choose the `pretrain_model` from the follwing options:   
    * MLM: `RobertaForMaskedLM` 
    * Shuffle + Random: `RobertaForShuffleRandomThreeWayClassification`
    * First Char: `RobertaForFirstCharPrediction`
    * ASCII: `RobertaForAsciiValuePrediction`
    * Random: `RobertaForRandomValuePrediction`


#### Pre-training progress  
You can use the Tensorboard to monitor the pre-training progress as follows:  
```
tensorboard --logdir=/path/to/log/dir/
```

#### Distributed pre-training  
The pre-training process can be distributed using the following command:  
```
python -m torch.distributed.launch \
--nproc_per_node=8 \
pretrainer.py \
--data_dir=/path/to/dataset/ \
--do_train \
--learning_rate=1e-4 \
--hidden_size=768 \
--intermediate_size=3072 \
--num_attention_heads=12 \
--num_hidden_layers=12 \
--weight_decay=0.01 \
--adam_epsilon=1e-8 \
--max_grad_norm=1.0 \
--num_train_epochs=10 \
--warmup_steps=10000 \
--save_steps=10000 \
--save_interval=100000 \
--seed=42 \
--per_device_train_batch_size=16 \
--logging_steps=100 \
--output_dir=/path/to/save/weights/ \
--overwrite_output_dir \
--logging_dir=/path/to/save/log/files/ \
--disable_tqdm=True \
--prediction_loss_only \
--fp16 \
--mlm_prob=0.15 \
--pretrain_model=RobertaForMaskedLM
```

## Fine-tunining on GLUE   
1. **Download GLUE data**  
    ```
    git clone https://github.com/huggingface/transformers
    python transformers/utils/download_glue_data.py
    ```

2. **Create a json config file**  
You can use `.json` file for the fine-tuning configuration or you can use command line arguments.

    ```json
    {
        "model_name_or_path": "/path/to/pre-trained/weights/",
        "tokenizer_name": "roberta-base",
        "task_name": "<task>",
        "do_train": true,
        "do_eval": true,
        "data_dir": "/path/to/task/dataset/",
        "max_seq_length": 128,
        "learning_rate": 2e-5,
        "num_train_epochs": 3, 
        "per_device_train_batch_size": 32,
        "per_device_eval_batch_size": 128,
        "logging_steps": 500,
        "logging_first_step": true,
        "save_steps": 1000,
        "save_total_limit": 2,
        "evaluate_during_training": true,
        "output_dir": "/path/to/save/models/",
        "overwrite_output_dir": true,
        "logging_dir": "/path/to/save/log/files/",
        "disable_tqdm": true
    }
    ```
    > For `task_name` and `data_dir`, you can choose one of the follwoing: CoLA, SST-2, MRPC, STS-B, QQP, MNLI, QNLI, RTE, and WNLI.  

3. **Fine-tuning**  
    ```
    python run_glue.py /path/to/json/
    ```


## Probing 
1. **Download probing tasks data**
    ```
    git clone https://github.com/facebookresearch/SentEval.git
    ```
    The data files for the probing tasks can be accessed in the `SentEval/data/probing/` directory.

2. **Model feature extraction**

    Run the following code to extract models features for each probing task:
    ```
    cd ./probing
    python extract_features.py \
    --data_file /path/to/probing/task/data/file/ \
    --output_file extracted_features_file_name.json \
    --output_dir /path/to/save/extracted/features/ \
    --bert_model /path/to/pre-trained/weights/
    ```
3. **Train the probing classifier**

    Run the following code to train the classifier for each probing task for a given model layer:
    ```
    cd ../
    python probe.py \
    --labels_file /path/to/probing/task/data/file/ \
    --feats_file /path/to/extracted/features/file.json \
    --layer 1 \
    --seed 9
    ```

## Citation  
```
@inproceedings{alajrami-aletras-2022-pre,
    title = "How does the pre-training objective affect what large language models learn about linguistic properties?",
    author = "Alajrami, Ahmed  and
      Aletras, Nikolaos",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-short.16",
    doi = "10.18653/v1/2022.acl-short.16",
    pages = "131--147",
    abstract = "Several pre-training objectives, such as masked language modeling (MLM), have been proposed to pre-train language models (e.g. BERT) with the aim of learning better language representations. However, to the best of our knowledge, no previous work so far has investigated how different pre-training objectives affect what BERT learns about linguistics properties. We hypothesize that linguistically motivated objectives such as MLM should help BERT to acquire better linguistic knowledge compared to other non-linguistically motivated objectives that are not intuitive or hard for humans to guess the association between the input and the label to be predicted. To this end, we pre-train BERT with two linguistically motivated objectives and three non-linguistically motivated ones. We then probe for linguistic characteristics encoded in the representation of the resulting models. We find strong evidence that there are only small differences in probing performance between the representations learned by the two different types of objectives. These surprising results question the dominant narrative of linguistically informed pre-training.",
}
```

## License
[MIT License](./LICENSE)
