# Towards Improving Faithfulness in Abstractive Summarization (NeurIPS 2022)

## 1. How to Install

### Requirements
- `python3`
- `conda create --name env `
- `pip3 install -r requirements.txt`

### Description of Codes

- `run_mybart` -> training and evaluation procedure
- `magic_bart.py` -> main models
- `module.py` -> modules
- `dataset_maker.py` -> data preprocessing

### Workspace
`./log/seq2seqV4/` will be created for storing model checkpoints and scores.

## 2. Preprocessing
Download the dataset from <https://drive.google.com/file/d/1b_NXY_KsMtkTpaftEfPJhJNw5VRmUgSg/view?usp=sharing>.
Download causal language model from <https://drive.google.com/file/d/1_XQ49dh07i6KNw3tE8H_WxXU9H3I94us/view?usp=sharing>.

For data preprocessing, please run

```
CUDA_VISIBLE_DEVICES=0 python3 run_mybart.py --model_name_or_path facebook/bart-base \
											 --do_train --do_eval --train_file [train_file] \
											 --validation_file [valid_file] \
											 --test_file [test_file] --output_dir das\
											 --exp_name cnndm --max_source_length 1024 \
											 --max_target_length 100 --gene_dataset_path tgt_dir
```
The preprocessing precedure will store the processed data as seperate json files in `tgt_dir`.

## 3. How to Run


### Train
```
python3 run_mybart.py --model_name_or_path facebook/bart-large \
                      --do_train --output_dir das \
                      --exp_name exp_name \
                      --max_source_length 1024 --max_target_length 100 \
                      --save_dataset_path tgt_dir\
                      --num_train_epochs 100 \
                      --per_device_train_batch_size 8 --save_strategy epoch \
                      --label_smoothing_factor 0.1 --weight_decay 0.01 \
                      --max_grad_norm 0.1 --warmup_steps 500\
                      --gradient_accumulation_steps 4 \
                      --learning_rate 3e-05 --margin_model True \
                      --lm_path lm_model
```
### Evaluate
```
python3 run_mybart.py  --per_device_eval_batch_size 32 \
									--log_root ./log --save_dataset_path tgt_dir \
									--exp_name exp_name --do_predict \
									--predict_with_generate True \
									--output_dir das \
									--val_max_target_length 120 \
									--model_name_or_path model_path\
									--lm_path lm_model
```

## Citation
We appreciate your citation if you find our work beneficial.

```
@article{chen2022towards,
  title={Towards Improving Faithfulness in Abstractive Summarization},
  author={Chen, Xiuying and Li, Mingzhe and Gao, Xin and Zhang, Xiangliang},
  journal={NeurIPS},
  year={2022}
}
```
