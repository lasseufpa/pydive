import torch
import os
from pydive.utils.output_feed import progress_bar

def train(
        epoch=None, model=None, loader=None, 
        optimizer=None, criterion=None, scheduler=False,
        swa=False, swa_model=None, swa_start=None,
        device='cpu'
        ):
    model.train()
    loader_size = len(loader.dataset)
    nb_batches = len(loader)

    running_loss = .0
    correct = .0

    for batch, (X, y) in enumerate(loader):
        # Zero os garientes guardados na iteração anterior
        optimizer.zero_grad()
        if device != 'cpu':
            X = X.to(device)
            y = y.to(device)
        # Calcula as predições
        output = model(X)
        # Calcula a perda do lote atual
        loss = criterion(output, y)
        # Multiplica a perda pelo tamanho de 1 lote
        running_loss += loss.item()               
        # Verifica o número de acertos no lote atual
        correct += (output.argmax(1) == y).type(torch.float).sum().item()
        # Calcula os gradientes
        loss.backward()
        # Atualiza os parâmetros
        optimizer.step()

    avg_loss = running_loss / nb_batches
    accuracy = (correct / loader_size)*100

    return avg_loss, accuracy

def validate(model=None, loader=None, criterion=None, device='cpu'):
    model.eval()

    size_loader = len(loader.dataset)
    nb_batches = len(loader)

    total_loss = .0
    correct = .0 

    with torch.no_grad():
        for  X, y in loader:
            if device != 'cpu': 
                X = X.to(device)
                y = y.to(device)
            output = model(X)
            total_loss += criterion(output, y).item()
            correct += (output.argmax(1) == y).type(torch.float).sum().item()

    avg_loss = total_loss / nb_batches
    accuracy = (correct / size_loader)*100

    return avg_loss, accuracy

def training_epoch(loader, 
                   model,
                   loss_fn, 
                   optimizer,
                   device='cpu',
                   feed=None,
                   ):

    model.train()
    # Tamanho do dataset (número de amostras) e do loader (número de lotes)
    loader_size = len(loader.dataset)
    num_batches = len(loader)

    running_loss = .0
    correct = .0
    total_instances = 0

    for batch_idx, (X, y) in enumerate(loader):
        X, y = X.to(device), y.to(device)
        # reset gradients
        optimizer.zero_grad()
        # compute the logits
        output = model(X)
        # compute the loss
        loss = loss_fn(output, y)
        # compute the gradients
        loss.backward()
        # update model parameters
        optimizer.step()

        # update current loss
        running_loss += loss.item()               
        # update the correct count
        total_instances += y.size(0)
        correct += (output.argmax(1) == y).type(torch.float).sum().item()

        if feed == 'bar':
            progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (running_loss/(batch_idx+1), 100.*correct/total_instances, correct, total_instances))

    avg_loss = running_loss / num_batches
    accuracy = (correct / total_instances)*100.

    return {
        'loss': avg_loss,
        'accuracy':  accuracy
    }

def test_evaluation(loader, model, loss_fn, device='cpu', feed=None):
    model.eval()

    loader_size = len(loader.dataset)
    num_batches = len(loader)

    running_loss = .0
    correct = .0
    total_instances = 0

    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(loader):
            X, y = X.to(device), y.to(device)
            output = model(X)
            loss = loss_fn(output, y)
            running_loss += loss.item()
            total_instances += y.size(0)
            correct += (output.argmax(1) == y).type(torch.float).sum().item()

        if feed == 'bar':
            progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (running_loss/(batch_idx+1), 100.*correct/total_instances, correct, total_instances))
            
    avg_loss = running_loss / num_batches
    accuracy = (correct / total_instances)*100.
        
    return {
        'loss': avg_loss,
        'accuracy': accuracy
    }

def save_checkpoint(dir, file, epoch, **kwargs):
    state = {
        'epoch': epoch,
    }
    state.update(kwargs)
    filepath = os.path.join(dir, file)
    torch.save(state, filepath)
