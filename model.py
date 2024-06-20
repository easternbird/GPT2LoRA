import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import GPT2Config, GPT2ForSequenceClassification, GPT2Tokenizer
from transformers.pytorch_utils import Conv1D

from lora import LinearLoRA, Conv1DLoRA
from utils import set_module
import config



class GPT2LoRA(nn.Module):
    def __init__(self, model_path=None):
        super().__init__()
        
        #get model configuration
        model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path="gpt2", num_labels=2)
        #load gpt2 model and tokenizer from pretrained model
        if model_path:
            model = torch.load(model_path)
        else:
            model = GPT2ForSequenceClassification.from_pretrained("gpt2")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2", config=model_config)
        # default to left padding
        self.tokenizer.padding_side = "left"
        # Define PAD Token = EOS Token = 50256
        self.tokenizer.pad_token = self.tokenizer.eos_token

        #resize model to match tokenizer, this step is necessary
        if not model_path:
            model.resize_token_embeddings(len(self.tokenizer))
            #initial model pad_token id to avoid error
            model.config.pad_token_id = model.config.eos_token_id    

        #freeze gpt2 model parameters
        if not config.train_gpt2:
            for name, param in model.named_parameters():
                if name != 'score.weight' or not config.train_score_weight:
                    param.requires_grad = False
            
        for name, param in model.named_parameters():
            print(name, ":", param.requires_grad)
            
        print('-'*20)
        
        #add MLP layer to output
        # self.lin_out = LinearLoRA(nn.Linear(50257, 2), config.rank, config.alpha)
            
        #fine-tuning gpt2 with lora
        if config.use_lora:
            for name, module in model.named_modules():
                if isinstance(module, Conv1D):
                    set_module(model, name, Conv1DLoRA(module, config.rank, config.alpha))
                if isinstance(module, nn.Linear):
                    set_module(model, name, LinearLoRA(module, config.rank, config.alpha))
        
        for name, param in model.named_parameters():
            print(name, ":", param.requires_grad)
            
        self.model = model
        self.loss = nn.CrossEntropyLoss
        
        self.pad_id = self.tokenize('<pad>').input_ids[0]
        
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = self.model.to(self.device)
        
        # self.prefix = "The movie review is:\""
        # self.post_prefix = "\". From this movie review we can see the author's attitude is"
        
        # #get id of token 'positive' and 'negative'
        # self.pos_id = self.tokenize('positive').input_ids[0]
        # self.neg_id = self.tokenize('negative').input_ids[0]

        
    def forward(self, *args, **kwargs):
        x =  self.model(*args, **kwargs)
        # x = self.lin_out(x.logits)
        return x
    
    def tokenize(self, text, return_tensors=None):
        if return_tensors:
            token = self.tokenizer(text, return_tensors=return_tensors)
        else:
            token = self.tokenizer(text)
        return token
        
    def batch_tokenize(self, text_batch):
        input_ids = list()
        attention_masks = list()
        
        for text in text_batch:
            tokenized = self.tokenizer.encode_plus(text, padding=True, truncation=True, return_tensors='pt')
            input_id = tokenized['input_ids']
            attention_mask = tokenized['attention_mask']
            
            # pad input id and mask
            padding_len = config.padding_len - len(input_id[0])
            # assert padding_len >= 0, 'padding_len in config is too small'
            if padding_len <= 0:
                input_id = input_id[:, :config.padding_len]
                attention_mask = attention_mask[:, :config.padding_len]
            else:
                input_id = F.pad(input_id, (0, padding_len), value=self.pad_id)
                attention_mask = F.pad(attention_mask, (0, padding_len), value=0)
            
            input_ids.append(input_id)
            attention_masks.append(attention_mask)
        
        input_ids = torch.stack(input_ids).squeeze().to(self.device)
        attention_masks = torch.stack(attention_masks).squeeze().to(self.device)
            
        return input_ids, attention_masks
        
        # #handle prefix and post prefix
        # pre_tok = self.tokenize(self.prefix, return_tensors='pt')
        # post_tok = self.tokenize(self.post_prefix, 'pt')
        
        # len_prefix = len(pre_tok['input_ids'][0])
        # len_post = len(post_tok['input_ids'][0])
        
        # #tokenize text in batch
        # for text, label in zip(text_batch, labels.tolist()):
        #     tokenized = self.tokenizer.encode_plus(text, padding=True, truncation=True, return_tensors='pt')
        #     input_id = tokenized['input_ids'].squeeze()
        #     attention_mask = tokenized['attention_mask'].squeeze()
        #     #pad input id and mask, 1 is for last word 'positive' or 'negative'
        #     padding_len = config.padding_len - len_prefix - len_post - 1
        #     assert padding_len >= 0, 'padding_len in config is too small'
        #     if len(input_id) >= padding_len:
        #         input_id = input_id[:padding_len]
        #         attention_mask = attention_mask[:padding_len]
        #     else:
        #         input_id = F.pad(input_id, (0, padding_len-len(input_id)), value=self.tokenize('<pad>').input_ids[0])
        #         attention_mask = F.pad(attention_mask, (0, padding_len-len(attention_mask)), value=0)
        #     att = self.pos_id if label else self.neg_id
        #     input_id = torch.cat([pre_tok['input_ids'][0], input_id, post_tok['input_ids'][0], torch.IntTensor([att])], dim=0)
        #     attention_mask = torch.cat([pre_tok['attention_mask'][0], attention_mask, post_tok['attention_mask'][0], torch.IntTensor([1])], dim=0)
        #     input_ids.append(input_id)
        #     attention_masks.append(attention_mask)
        
        # input_ids = torch.stack(input_ids)
        # attention_masks = torch.stack(attention_masks)
        
        # return input_ids, attention_masks
        # return dict(input_ids=input_ids, attention_mask=attention_masks)
        
        
class GPT2ClassificationCollator(object):
    r"""
    Data Collator used for GPT2 in a classificaiton rask. 
    
    It uses a given tokenizer and label encoder to convert any text and labels to numbers that 
    can go straight into a GPT2 model.

    This class is built with reusability in mind: it can be used as is as long
    as the `dataloader` outputs a batch in dictionary format that can be passed 
    straight into the model - `model(**batch)`.

    Arguments:

      use_tokenizer (:obj:`transformers.tokenization_?`):
          Transformer type tokenizer used to process raw text into numbers.

      labels_ids (:obj:`dict`):
          Dictionary to encode any labels names into numbers. Keys map to 
          labels names and Values map to number associated to those labels.

      max_sequence_len (:obj:`int`, `optional`)
          Value to indicate the maximum desired sequence to truncate or pad text
          sequences. If no value is passed it will used maximum sequence size
          supported by the tokenizer and model.

    """

    def __init__(self, tokenizer, max_sequence_len=None):

        # Tokenizer to be used inside the class.
        self.tokenizer = tokenizer
        # Check max sequence length.
        self.max_sequence_len = tokenizer.model_max_length if max_sequence_len is None else max_sequence_len


        return

    def __call__(self, sequences):
        r"""
        This function allowes the class objesct to be used as a function call.
        Sine the PyTorch DataLoader needs a collator function, I can use this 
        class as a function.

        Arguments:

          item (:obj:`list`):
              List of texts and labels.

        Returns:
          :obj:`Dict[str, object]`: Dictionary of inputs that feed into the model.
          It holddes the statement `model(**Returned Dictionary)`.
        """

        # Get all texts from sequences list.
        texts = [sequence['text'] for sequence in sequences]
        # Get all labels from sequences list.
        labels = [sequence['label'] for sequence in sequences]
        # Call tokenizer on all texts to convert into tensors of numbers with 
        # appropriate padding.
        inputs = self.tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True,  max_length=self.max_sequence_len)
        # Update the inputs with the associated encoded labels as tensor.
        inputs.update({'labels':torch.tensor(labels)})

        return inputs
