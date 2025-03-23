import torch
from tqdm.auto import tqdm
import numpy as np

def train(train_loader, val_loader, net, loss_fn, optimizer, dtype, num_epochs, device, steps_per_update=-1):

    # set train flags, initialize step
    net.train() 
    loss_fn.train()
    epoch = 0

    while epoch < num_epochs:
        # reset data loader to get random augmentations
        np.random.seed()
   
        # zero gradients
        total_loss = 0.0  # To track the sum of losses for averaging

        if steps_per_update == -1:
            total_steps = len(train_loader.dataset)
        else:
            total_steps = steps_per_update         

        with tqdm(total=total_steps, desc=f"Epoch {epoch}", leave=True) as pbar:
    
            for feature, label in train_loader:
            
                optimizer.zero_grad()

                label = label.type(dtype)
                label = label.to(device)
                feature = feature.to(device)
                
                # forward
                predicted = net(feature)
                label=torch.squeeze(label,1)
                loss_value = loss_fn(input=predicted, target=label)

                # Accumulate loss for averaging
                total_loss += loss_value.item()

                # pass through loss
                loss_value.backward()
                
                pbar.update(label.shape[0])
            
                optimizer.step()

        # Compute the average loss over all steps
        average_loss = total_loss / total_steps
        pbar.write(f'training loss at epoch {epoch} is {average_loss}')
            
        epoch += 1
       
def test_data_loader(train_loader, val_loader, net, loss_fn, optimizer, dtype, num_epochs, steps_per_update, device, writer):

    # set train flags, initialize step
    net.train() 
    loss_fn.train()
    epoch = 0

    while epoch < num_epochs:
        # reset data loader to get random augmentations
        np.random.seed()
        
        # reset pbar
        #pbar.reset()
        #pbar = tqdm(total=steps_per_update)
        #pbar.set_description(f"Epoch {step}")
    
        # zero gradients

        total_loss = 0.0  # To track the sum of losses for averaging

        total_steps = len(train_loader.dataset)        
        with tqdm(total=total_steps, desc=f"Epoch {epoch}", leave=True) as pbar:
    
            for feature, label in train_loader:
            
                print(feature.shape, feature.dtype, feature.min(), feature.max(), feature.mean())
                print(label.shape, label.dtype, label.min(), label.max())
                pbar.update(label.shape[0])
            
