from torch import optim

def get_optimizer(model, optimizer_type, optimizer_params):
    if optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=optimizer_params['lr'], betas=(optimizer_params['beta1'],optimizer_params['beta2']))
    return optimizer