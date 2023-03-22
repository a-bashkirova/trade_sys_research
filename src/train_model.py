def train_model(model, loss, optimizer, num_epochs, cnn_flg=False):
    history = {'train': [],
               'val': []}
    epochs = tqdm(range(num_epochs))
    for epoch in epochs:
        for mode in ['train', 'val']:
            if mode == 'train':
                dataloader = train_dataloader
                model.train()
            else:
                dataloader = test_dataloader
                model.eval()

            running_loss = 0.
            running_acc = 0.

            for input, labels in dataloader:
                if cnn_flg:
                    input = input[None, :]
                    input = input.permute(1, 0, 2)
                input = input.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(mode == 'train'):
                    preds = model(input)
                    losses = loss(preds, labels)
                    pred_class = model.infer(input)

                    if mode == 'train':
                        losses.backward()
                        optimizer.step()

                running_loss += losses.item()
                running_acc += (pred_class == labels.data).float().mean()

            epoch_loss = running_loss / len(dataloader)
            epoch_acc = running_acc / len(dataloader)
            history[mode].append(epoch_loss)

            if mode == 'val':
                epochs.set_description('Val loss: {:.4f} Val acc: {:.4f}'.format(epoch_loss, epoch_acc))

    return history