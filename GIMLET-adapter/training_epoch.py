"""Sample-weighted gradient accumulation for the independent labeled protocol."""
import torch


def train_labeled_epoch(model, loader, optimizer, accumulation_steps, progress=lambda x: x):
    if accumulation_steps < 1 or len(loader) == 0:
        raise ValueError('Nonempty training data and positive accumulation steps required')
    model.train()
    optimizer.zero_grad()
    loss_sum, samples, window_samples = 0., 0, 0
    for step, batch in progress(enumerate(loader)):
        batch = {key: value.to(model.device) for key, value in batch.items()}
        count = batch['input_ids'].shape[0]
        loss = model(**batch)['loss']
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss')
        (loss * count).backward()
        loss_sum += loss.detach().item() * count
        samples += count
        window_samples += count
        if (step + 1) % accumulation_steps == 0 or step + 1 == len(loader):
            for group in optimizer.param_groups:
                for parameter in group['params']:
                    if parameter.grad is not None:
                        parameter.grad.div_(window_samples)
            optimizer.step()
            optimizer.zero_grad()
            window_samples = 0
    return loss_sum / samples
