#  LSA

Official PyTorch implementation of [LSA: Layer-wise Sparsity Allocation for Large Language Model Pruning Based on Minimal Linear Reconstruction Error]().





## Results

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
- `--N`: N:M sprsity.
- `--M`: N:M sprsity.
- `--tasks`: Eval ppl and zero shot tasks.
- `--num_examples`: Calibration dataset size.
- `--block`: Group size for LSA or GPTQ.

Below is an example command for pruning LLaMA-7B with LSA, to achieve unstructured 70% sparsity.

```
python main.py \
--base_model huggyllama/llama-7b \
-s 0.7 -p sgpt --layer lsa \
--num_examples 128 --block 128 \
--tasks wikitext,ptb,c4,storycloze,rte,openbookqa,arc_easy,winogrande,arc_challenge,piqa,boolq,hellaswag \
--fp16
```    

Below is an example command for pruning LLaMA-7B with LSA, to achieve structured 20% sparsity.

```
python main.py \
--base_model huggyllama/llama-7b\
-p llm -s 0.2 --layer lsa\
--num_examples 10 --block 128\
--tasks wikitext,ptb
``` 

Below is an example command for pruning LLaMA-7B with LSA, to achieve N:M 75% sparsity.

```
python main.py\
--base_model huggyllama/llama-7b\
-p wanda -s 0 --layer lsa\
--num_examples 128 --block 128\
--tasks wikitext\
--N 2 --M 8\
--fp16
```    

Below is an example command for pruning LLaMA-7B with LSA, to achieve quantization.

```
python main.py\
--base_model huggyllama/llama-7b\
-p sgpt  -s 0.7 --layer lsa\
--num_examples 128 --block 128\
--tasks wikitext,ptb,c4\
--gptq --wbits 4 --act-order\
--fp16
```   

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
