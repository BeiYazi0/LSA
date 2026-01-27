#  LSA

Official PyTorch implementation of [LSA: Layer-wise Sparsity Allocation for Large Language Model Pruning Based on Minimal Linear Reconstruction Error](https://openreview.net/forum?id=xq3lza5IjN).


## Results

Zero-shot results. We produce the Uniform, OWL, DLP and LSA(Ours) with 70% unstructured sparsity on LLaMA3, Qwen2.5, Qwen3 models. The best performance result is indicated in bold.

Table 1. Accuracy (%) of LLaMA3-8B on seven zero-shot tasks at 70% unstructured sparsity.

| LLaMA3-8B | winogrande | hellaswag | boolq     | piqa      | openbookqa | arc_easy  | arc_challenge | avg       |
|-----------|------------|-----------|-----------|-----------|------------|-----------|---------------|-----------|
| Dense     | 72.61      | 60.17     | 81.59     | 79.65     | 34.8       | 80.09     | 50.17         | 65.58     |
| SparseGPT | 57.3       | 33.74     | 66.39     | 62.89     | 15.0       | 44.95     | 22.01         | 43.18     |
| +owl      | 60.93      | 36.67     | 70.58     | 65.4      | 16.2       | 48.27     | 23.81         | 45.98     |
| +dlp      | 61.56      | 37.54     | 70.67     | 65.13     | **19.2**   | 48.19     | **25.6**      | 46.84     |
| +lsa      | **62.12**  | **37.93** | **74.53** | **65.89** | 18.6       | **48.95** | 25.09         | **47.59** |
| Wanda     | 48.22      | 27.28     | 50.43     | 55.6      | 13.6       | 32.15     | 17.66         | 34.99     |
| +owl      | 49.49      | 28.4      | **61.5**  | 57.83     | 13.4       | 35.52     | 17.66         | 37.69     |
| +dlp      | 52.41      | 29.51     | 58.53     | 60.66     | 14.0       | 38.51     | **19.03**     | 38.95     |
| +lsa      | **53.91**  | **30.45** | 60.49     | **60.83** | **15.0**   | **41.04** | **19.03**     | **40.11** |

Table 2. Accuracy (%) of Qwen2.5-7B  on seven zero-shot tasks at 70% unstructured sparsity.

| Qwen2.5-7B    | winogrande | hellaswag | boolq     | piqa      | openbookqa | arc_easy  | arc_challenge | avg       |
|---------------|------------|-----------|-----------|-----------|------------|-----------|---------------|-----------|
| Dense         | 73.01      | 60.04     | 85.11     | 78.78     | 33.20      | 80.47     | 47.78         | 65.48     |
| SparseGPT     | 61.72      | 40.00     | 73.24     | 68.93     | 20.00      | 63.05     | 29.18         | 50.88     |
| +owl          | 61.09      | 38.02     | 64.62     | 67.63     | 19.20      | 59.93     | 27.39         | 48.27     |
| +dlp          | 61.96      | 38.31     | 67.80     | 65.40     | 18.80      | 55.98     | 26.28         | 47.79     |
| +lsa          | 62.67      | 39.04     | **77.40** | 65.78     | 19.00      | 56.10     | 27.65         | 49.66     |
| +lsa(block-w) | **64.33**  | **40.68** | 72.69     | **68.99** | **23.00**  | **63.34** | **29.69**     | **51.82** |
| Wanda         | 53.04      | 30.59     | 62.02     | 61.81     | 15.80      | 45.75     | 20.56         | 41.37     |
| +owl          | 52.17      | 30.68     | 62.17     | 62.02     | 14.60      | 45.66     | 19.37         | 40.95     |
| +dlp          | 56.67      | 33.38     | 62.23     | 63.00     | 16.60      | 45.62     | 21.16         | 42.67     |
| +lsa          | **56.91**  | **33.41** | **62.57** | **63.06** | **16.80**  | **46.68** | **23.29**     | **43.24** |

Table 3. Accuracy (%) of Qwen3-8B  on seven zero-shot tasks at 70% unstructured sparsity.

| Qwen3-8B           | winogrande | hellaswag | boolq     | piqa      | openbookqa | arc_easy  | arc_challenge | avg       |
|--------------------|------------|-----------|-----------|-----------|------------|-----------|---------------|-----------|
| Dense              | 67.64      | 57.12     | 86.64     | 76.88     | 31.0       | 83.54     | 55.89         | 65.53     |
| SparseGPT          | 62.12      | 38.57     | **73.30** | 68.72     | 21.20      | 61.45     | 29.52         | 50.70     |
| +owl               | 60.22      | 37.17     | 66.54     | 67.41     | 21.40      | 58.75     | 27.73         | 48.46     |
| +dlp               | 63.77      | 37.58     | 63.33     | 66.00     | 20.40      | 57.49     | 29.61         | 48.31     |
| +lsa               | **64.72**  | 38.31     | 68.44     | 67.74     | 22.40      | 59.93     | 31.31         | 50.41     |
| +lsa(projection-w) | 64.01      | **39.49** | 71.96     | **69.15** | **23.80**  | **62.92** | **31.48**     | **51.83** |
| Wanda              | 53.51      | 30.53     | 62.32     | 61.15     | 15.00      | 50.04     | 21.25         | 41.97     |
| +owl               | 52.09      | 29.52     | 61.99     | 61.15     | 15.20      | 47.18     | 18.86         | 40.86     |
| +dlp               | 55.49      | 31.90     | 62.20     | 62.02     | 16.20      | 48.48     | 23.38         | 42.81     |
| +lsa               | **57.54**  | **32.84** | **62.39** | **63.76** | **16.40**  | **51.60** | **24.15**     | **44.10** |


## Installation 

Step 1: Create a new conda environment:
```
conda create -n lsa python=3.9
conda activate lsa
```
Step 2: Install relevant packages
```
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install transformers==4.28.0 datasets==2.16.1 wandb sentencepiece
pip install accelerate==0.18.0
```


## Usage

We provide a quick overview of the arguments:  
- `--base_model`: The identifier for the LLaMA model on the Hugging Face model hub.
- `--pruner`: We have implemented three pruning methods, namely [`mag`, `wanda`, `sgpt`].
- `--layer`: We have implemented three sparsity allocation methods, namely [`owl`, `dlp`, `lsa`].
- `--final_s`: Denotes the percentage of weights to be pruned.
- `--tasks`: Eval ppl and zero shot tasks.
- `--num_examples`: Calibration dataset size.
- `--block`: Group size for LSA or GPTQ.

### unstructured sparsity

Below is an example command for pruning LLaMA3-8B with LSA, to achieve unstructured 70% sparsity.

```
python main.py \
--base_model meta-llama/Meta-Llama-3-8B \
-s 0.7 -p sgpt --layer lsa \
--num_examples 128 --block 128 \
--tasks wikitext,ptb,c4,storycloze,rte,openbookqa,arc_easy,winogrande,arc_challenge,piqa,boolq,hellaswag \
--fp16
```    

Below is an example command for pruning Qwen2.5-7B with LSA(block-wise), to achieve unstructured 70% sparsity.

```
python main.py \
--base_model Qwen/Qwen2.5-7B \
-s 0.7 -p sgpt --layer lsab \
--num_examples 128 --block 128 \
--tasks wikitext,ptb,c4,storycloze,rte,openbookqa,arc_easy,winogrande,arc_challenge,piqa,boolq,hellaswag \
--fp16 --Lamda 0.07
```  

Below is an example command for pruning Qwen3-8B with LSA(projection-wise), to achieve unstructured 70% sparsity.

```
python main.py \
--base_model Qwen/Qwen3-8B \
-s 0.7 -p sgpt --layer lsac \
--num_examples 128 --block 128 \
--tasks wikitext,ptb,c4,storycloze,rte,openbookqa,arc_easy,winogrande,arc_challenge,piqa,boolq,hellaswag \
--fp16 --Lamda 0.07
```

### deepsparse

Below is an example command for pruning Llama-2-7b-chat-hf with LSA, to achieve speedup on deepsparse.

```
deepsparse==1.8.0
torch==2.6.0+cu124
onnx==1.16.0
onnxruntime==1.16.0
onnx-ir==0.1.6
onnxscript==0.3.2
onnx-graphsurgeon==0.5.2
protobuf==6.32.0
``` 

``` 
python main.py\
--base_model NousResearch/Llama-2-7b-chat-hf\
-s 0.1 -p wanda --layer lsa\
--num_examples 128 --block 128\
--deepsparse --onnx_export_path ./Llama-2-7B/chat-onnx
``` 


``` 
deepsparse.benchmark ./Llama-2-7B/chat-onnx -b 1 -s sync -nstreams 1
``` 
  
  

### Acknowledgement
The implementation of GPTQ is build upon the [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa) repositories.

Zero-shot tasks is evaluated on the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) repositories.


### Citation
``` 
@inproceedings{
yang2026lsa,
title={{LSA}: Layer-wise Sparsity Allocation for Large Language Model Pruning Based on Minimal Linear Reconstruction Error},
author={Zhiguo Yang and Changjian Deng and Qinke Chen and Zijing Zhou and Jian Cheng},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=xq3lza5IjN}
}
``` 