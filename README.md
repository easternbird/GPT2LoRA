# GPT2LoRA

This repo is a sample to use LoRA method for fine-tuning GPT-2 model. If you want to train on your own, please follow these steps:

Firstly, download requirements which is nessecary to run the code by using command:

```python
pip install -r requirements.txt

```

Then, prepatre the IMDB dataset from [here](https://huggingface.co/datasets/stanfordnlp/imdb). After downloading the datasets, put them into the folder '/IMDB' and rename these data to 'train.parquet' and 'test.parquet'.

Next, download your GPT-2 model from [here](https://huggingface.co/openai-community/gpt2) or by running command:
```python
git clone https://huggingface.co/openai-community/gpt2
```

Finally, train your model by command:
```python
python train.py
```

You can see accuracy and summary report in the output every epoch. Also, the model will be saved after an epoch is finished. Trained models will be saved to path '/outputs'.
