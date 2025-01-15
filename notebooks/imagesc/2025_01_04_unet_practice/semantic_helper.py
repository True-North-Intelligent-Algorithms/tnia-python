import torch
from tqdm.auto import tqdm
import numpy as np

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