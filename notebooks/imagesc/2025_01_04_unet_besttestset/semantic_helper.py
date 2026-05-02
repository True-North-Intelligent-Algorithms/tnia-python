import torch
from tqdm.auto import tqdm
import numpy as np
import random

def model_step(model, loss_fn, optimizer, feature, label, activation, train_step=True):
    
    # zero gradients if training
    if train_step:
        optimizer.zero_grad()
     
    # forward
    #logits = model(feature)
    # final activation
    #predicted = activation(logits)

    predicted = model(feature)

    label=torch.squeeze(label,1)
    # pass through loss
    loss_value = loss_fn(input=predicted, target=label)
    
    # backward if training mode
    if train_step:
        loss_value.backward()
        optimizer.step()

    outputs = {
        'pred': predicted,
        'logits': None,
    }
    
    return loss_value, outputs

def train(train_loader, val_loader, net, loss_fn, activation, optimizer, dtype, training_steps, device, writer):

    # set train flags, initialize step
    net.train() 
    loss_fn.train()
    step = 0

    with tqdm(total=training_steps) as pbar:
        while step < training_steps:
            # reset data loader to get random augmentations
            np.random.seed()
            tmp_loader = iter(train_loader)
            for feature, label in tmp_loader:
                label = label.type(dtype)
                label = label.to(device)
                feature = feature.to(device)
                loss_value, pred = model_step(net, loss_fn, optimizer, feature, label, activation)
                writer.add_scalar('loss',loss_value.cpu().detach().numpy(),step)
                step += 1
                pbar.update(1)
                if step % 100 == 0:
                    net.eval()
                    tmp_val_loader = iter(val_loader)
                    acc_loss = []
                    for feature, label in tmp_val_loader:                    
                        label = label.type(dtype)
                        label = label.to(device)
                        feature = feature.to(device)
                        loss_value, _ = model_step(net, loss_fn, optimizer, feature, label, activation, train_step=False)
                        acc_loss.append(loss_value.cpu().detach().numpy())
                    writer.add_scalar('val_loss',np.mean(acc_loss),step)
                    net.train()

                    print(np.mean(acc_loss))

def train2(train_loader, val_loader, net, loss_fn, optimizer, dtype, num_epochs, steps_per_update, device, writer):

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
        optimizer.zero_grad()

        total_loss = 0.0  # To track the sum of losses for averaging
        
        with tqdm(total=steps_per_update, desc=f"Epoch {epoch}", leave=True) as pbar:
    
            for i in range(steps_per_update):
            
                tmp_loader = iter(train_loader)
                
                feature, label = next(tmp_loader)

                #for feature, label in tmp_loader:
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
                
                pbar.update(1)
            
                optimizer.step()

        # Compute the average loss over all steps
        average_loss = total_loss / steps_per_update
            
        writer.add_scalar('loss',average_loss,epoch)
        epoch += 1
        if True: #step % 100 == 0:
            net.eval()
            tmp_val_loader = iter(val_loader)
            acc_loss = []
            for feature, label in tmp_val_loader:                    
                label = label.type(dtype)
                label = label.to(device)
                feature = feature.to(device)
                
                predicted = net(feature)
                label=torch.squeeze(label,1)
                loss_value = loss_fn(input=predicted, target=label)
                #loss_value, _ = model_step(net, loss_fn, optimizer, feature, label, train_step=False)
                
                acc_loss.append(loss_value.cpu().detach().numpy())
            writer.add_scalar('val_loss',np.mean(acc_loss),epoch)
            net.train()

            #print(f'training loss at epoch {step} is {average_loss}')
            pbar.write(f'training loss at epoch {epoch} is {average_loss}')
            #print(np.mean(acc_loss))

def train3(train_loader, val_loader, net, loss_fn, optimizer, dtype, num_epochs, device, steps_per_update=-1):

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
            
        epoch += 1
        
        # Todo: work more on eval step
        if False: #step % 100 == 0:
            net.eval()
            tmp_val_loader = iter(val_loader)
            acc_loss = []
            for feature, label in tmp_val_loader:                    
                label = label.type(dtype)
                label = label.to(device)
                feature = feature.to(device)
                
                predicted = net(feature)
                label=torch.squeeze(label,1)
                loss_value = loss_fn(input=predicted, target=label)
                #loss_value, _ = model_step(net, loss_fn, optimizer, feature, label, train_step=False)
                
                acc_loss.append(loss_value.cpu().detach().numpy())
            net.train()

            #print(f'training loss at epoch {step} is {average_loss}')
            pbar.write(f'training loss at epoch {epoch} is {average_loss}')
            #print(np.mean(acc_loss))


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
            
