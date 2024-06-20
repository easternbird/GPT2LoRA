import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
import sys
import os

from model import GPT2LoRA, GPT2ClassificationCollator
from data import IMDBDataset
import config
from utils import print_param_info



def load_model(path):
    
    last_model_path = None
    
    if os.path.exists(path):
        epoch = 0
        while True:
            model_path = os.path.join(path, f'checkpoint_epoch_{epoch}.pt')
            if os.path.exists(model_path):
                last_model_path = model_path
                epoch += 1
            else:
                break
        
        if last_model_path:
            # model = torch.load(last_model_path)
            model = GPT2LoRA(last_model_path)
            last_epoch = epoch
            print("Resume from last epoch %d" % epoch)
        else:
            model = GPT2LoRA()
            last_epoch = 0
            
    else:
        os.mkdir(path)
        print("Output model path created:", path)
        model = GPT2LoRA()
        last_epoch = 0
            
    return model, last_epoch
        

def save_model(model, output_path):
    print("Saving model to:", output_path)
    torch.save(model, output_path) 
    print("Model has been successfully saved.")   

def train(epoch: int, model: GPT2LoRA, optimizer: AdamW, train_dataloader: DataLoader):
    
    true_labels = list()
    predicted_labels = list()
    total_loss = 0
    
    print(f">>> Training epoch {epoch}")
    sys.stdout.flush()
    
    #allow model in train mode
    model.train()
    
    progress = tqdm(total=len(train_dataloader), desc='Epoch %d' % epoch)
    
    for _, batch in enumerate(train_dataloader):
        
        true_labels += batch['labels'].numpy().flatten().tolist()
         # move batch to device
        batch = {k:v.type(torch.long).to(model.device) for k,v in batch.items()}
        # Always clear any previously calculated gradients before performing a backward pass.
        model.zero_grad()
        optimizer.zero_grad()
        outputs = model(**batch)
        # The call to `model` always returns a tuple, so we need to pull the 
        # loss value out of the tuple along with the logits. We will use logits
        # later to calculate training accuracy.
        loss, logits = outputs[:2]

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value 
        # from the tensor.
        total_loss += loss.item()
        
        # Perform a backward pass to calculate the gradients.
        loss.backward()
        optimizer.step()
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()

        # Convert these logits to list of predicted labels values.
        predicted_labels += logits.argmax(axis=-1).flatten().tolist()

        progress.set_postfix({"loss": loss.item()})
        progress.update()
        
    progress.close()
    
    # Calculate the average loss over the training data.
    avg_epoch_loss = total_loss / len(train_dataloader)
    
    #save model
    output_path = f'outputs/checkpoint_epoch_{epoch}.pt'
    save_model(model, output_path)
    
    # Return all true labels and prediction for future evaluations.
    return true_labels, predicted_labels, avg_epoch_loss





def val(epoch: int, model: GPT2LoRA, val_dataloader: DataLoader):
    
    true_labels = list()
    predicted_labels = list()
    total_loss = 0
    
    print(f">>> Evaling epoch {epoch}")
    sys.stdout.flush()
    
    #allow model in eval mode
    model.eval()
    
    progress = tqdm(total=len(val_dataloader), desc='Epoch %d' % epoch)
    
    for _, batch in enumerate(val_dataloader):
        
        true_labels += batch['labels'].numpy().flatten().tolist()
         # move batch to device
        batch = {k:v.type(torch.long).to(model.device) for k,v in batch.items()}
        
        # set no grad to avoid grad calculation
        with torch.no_grad():
        
            outputs = model(**batch)
            # The call to `model` always returns a tuple, so we need to pull the 
            # loss value out of the tuple along with the logits. We will use logits
            # later to calculate training accuracy.
            loss, logits = outputs[:2]

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value 
            # from the tensor.
            total_loss += loss.item()
            
            # Perform a backward pass to calculate the gradients.

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()

            # Convert these logits to list of predicted labels values.
            predicted_labels += logits.argmax(axis=-1).flatten().tolist()
        
        progress.set_postfix({"loss": loss.item()})
        progress.update()
        
    progress.close()
    
    # Calculate the average loss over the training data.
    avg_epoch_loss = total_loss / len(val_dataloader)
    
    # Return all true labels and prediction for future evaluations.
    return true_labels, predicted_labels, avg_epoch_loss






def main():
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model_path = 'outputs'
    
    model, epoch = load_model(model_path)
    
    
    # Create data collator to encode text and labels into numbers.
    gpt2_classificaiton_collator = GPT2ClassificationCollator(tokenizer=model.tokenizer, 
                                                              max_sequence_len=config.padding_len)
    

    # load IMDB data
    train_dataset = IMDBDataset(is_train=True)
    val_dataset = IMDBDataset(is_train=False)
    #tokenize dataset

    # set dataloader
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, 
                                  collate_fn=gpt2_classificaiton_collator, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=config.batch_size, 
                                collate_fn=gpt2_classificaiton_collator, shuffle=True)
    
    #init optimizer
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    
    #print params info
    print_param_info(model)
    print(model.model)
    
    # for name, param in model.model.named_parameters():
    #     print(name, ":", param.requires_grad)
    
    #train and eval model
    while True:
        train_labels, train_predict, train_loss = train(epoch, model, optimizer, train_dataloader)
        train_acc = accuracy_score(train_labels, train_predict)
        
        val_labels, val_predict, val_loss = val(epoch, model, val_dataloader)
        val_acc = accuracy_score(val_labels, val_predict)
        
        print(f"====== Epoch {epoch} results ======")
        print("train_loss: %.5f" % train_loss)
        print("val_loss: %.5f" % val_loss)
        print("train_acc: %.5f" % train_acc)
        print("valid_acc: %.5f" % val_acc)
        
        # show the evaluation report.
        evaluation_report = classification_report(val_labels, val_predict, labels=[0, 1], target_names=['neg', 'pos'])
        print("evaluation report:")
        print("==================================")
        print(evaluation_report)
        print("==================================")
        
        epoch += 1
    

if __name__ == '__main__':
    main()