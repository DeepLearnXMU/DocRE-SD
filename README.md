## Requirements
* Python (tested on 3.9.7)
* CUDA (tested on 11.4)
* PyTorch (tested on 1.10.1)
* Transformers (tested on 4.15.0)
* numpy (tested on 1.20.3)
* wandb
* ujson
* tqdm

## Dataset
The [DocRED](https://www.aclweb.org/anthology/P19-1074/) dataset can be downloaded following the instructions at [link](https://github.com/thunlp/DocRED/tree/master/data). 
```
Code
 |-- docred
 |    |-- train_annotated.json        
 |    |-- train_distant.json
 |    |-- dev.json
 |    |-- test.json
 |    |-- rel2id.json
 |    |-- ner2id.json
 |    |-- rel_info.json
 |-- logs
 |-- result
 |-- model
 |    |-- bert        
 |    |-- save_model
```
**Notice:**

1. The code of our reasoning module is located in the `Reasoning_module.py` file.
2. The code of our self-distillation training framework is located in the `Entity_level_predict()` function of `model.py`.

## Training and Evaluation on DocRED
### Training Model
Train the BERT-based model on DocRED with the following command:
```bash
>> sh run_bert.sh 
```
Hyper-parameters Setting
```bash
#! /bin/bash
export CUDA_VISIBLE_DEVICES=1
python3 -u train.py \
  --data_dir ./docred \
  --transformer_type bert \
  --model_name_or_path ./model/bert \
  --save_path ./model/save_model/ \
  --load_path ./model/save_model/ \
  --train_file train_annotated.json \
  -- train_base \
  --dev_file dev.json \
  --test_file test.json \
  --train_batch_size 9 \
  --test_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --num_labels 4 \
  --decoder_layers 1 \
  --learning_rate 1e-4 \
  --encoder_lr 5e-5 \
  --max_grad_norm 1.0 \
  --warmup_ratio 0.06 \
  --num_train_epochs 52 \
  --seed 66 \
  --num_class 97 \
  | tee logs/logs.train.log 2>&1
```
The trained base module is saved in `./model/save_model/`.


### Evaluating Model
First, you can set the model to be loaded on line 346 of `train.py`.

Then, Test our entire model on DocRED with the following command:

```bash
>> sh run_bert.sh 
```
Hyper-parameters Setting
```bash
#! /bin/bash
export CUDA_VISIBLE_DEVICES=1
python3 -u train.py \
  --data_dir ./docred \
  --transformer_type bert \
  --model_name_or_path ./model/bert \
  --save_path ./model/save_model/ \
  --load_path ./model/save_model/ \
  --train_file train_annotated.json \
  --test \
  --dev_file dev.json \
  --test_file test.json \
  --train_batch_size 9 \
  --test_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --num_labels 4 \
  --decoder_layers 1 \
  --learning_rate 1e-4 \
  --encoder_lr 1e-5 \
  --max_grad_norm 1.0 \
  --warmup_ratio 0.06 \
  --num_train_epochs 52 \
  --seed 66 \
  --num_class 97 \
  | tee logs/logs.train.log 2>&1
```

The program will generate a test file `./result/result.json` in the official evaluation format. 
You can compress and submit it to Colab for the official test score.

