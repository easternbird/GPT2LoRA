import torch
import torch.nn.functional as F

def set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def label_to_onehot(label_batch, pos_id, neg_id, num_classes=50257):
    label_batch = torch.where(label_batch, pos_id, neg_id)
    label_onehot = F.one_hot(label_batch, num_classes)
    return label_onehot


def int_to_unitstr(x: int):
    assert 0 <= x < 1e12
    units = ['', 'K', 'M', 'B']
    i = 0
    while x >= 1000:
        x /= 1000
        i += 1
        
    return str(round(x, 2)) + units[i]


def print_param_info(model):
    
    print("The max model length is {} for this model, although the actual embedding size for GPT small is 768".format(model.tokenizer.model_max_length))
    print("The beginning of sequence token {} token has the id {}".format(model.tokenizer.convert_ids_to_tokens(model.tokenizer.bos_token_id), model.tokenizer.bos_token_id))
    print("The end of sequence token {} has the id {}".format(model.tokenizer.convert_ids_to_tokens(model.tokenizer.eos_token_id), model.tokenizer.eos_token_id))
    print("The padding token {} has the id {}".format(model.tokenizer.convert_ids_to_tokens(model.tokenizer.pad_token_id), model.tokenizer.pad_token_id))

    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print('-'*20)
    print("Total params:", int_to_unitstr(total_num))
    print("Trainable params:", int_to_unitstr(trainable_num))
    print("-"*20)

