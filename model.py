import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.pytorch_utils import Conv1D

from lora import LinearLoRA, Conv1DLoRA
from utils import set_module
import config



class GPT2LoRA(nn.Module):
    def __init__(self):
        super().__init__()
        
        #load gpt2 model and tokenizer from pretrained model
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        #resize model to match tokenizer, this step is necessary
        model.resize_token_embeddings(len(tokenizer))
        #initial model pad_token id to avoid error
        model.generation_config.pad_token_id = model.generation_config.eos_token_id    

        #freeze gpt2 model parameters
        for param in model.parameters():
            param.requires_grad = False
        
        #add MLP layer to output
        self.model = nn.Sequential(model, 
                                   LinearLoRA(nn.Linear(50257, 2), config.rank, config.alpha)
                                   )
            
        #fine-tuning gpt2 with lora
        if config.use_lora:
            for name, module in model.named_modules():
                if isinstance(module, Conv1D):
                    set_module(model, name, Conv1DLoRA(module, config.rank, config.alpha))
                if isinstance(module, nn.Linear):
                    set_module(model, name, LinearLoRA(module, config.rank, config.alpha))
        
    def forward(self, x):
        return self.model(x)