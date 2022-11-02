from tqdm.auto import tqdm
import numpy as np
from config import *
from model import Conv3DNet
import torch.optim.lr_scheduler
import dataloader
import loss


loss_hist = []
val_loss_hist = []
patience_counter = 0
best_val_loss = np.inf

model3d = Conv3DNet().to(device)

# Adam optimiser
optimiser = torch.optim.AdamW(params=model3d.parameters(), lr=LEARNING_RATE)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=N_EPOCHS)

# Loop over epochs
for epoch in tqdm(range(N_EPOCHS)):
    loss_acc = 0
    val_loss_acc = 0
    train_count = 0
    valid_count = 0

    # Loop over batches
    for imgs, labels in dataloader.train_loader:
        # Send to device
        imgs = imgs.to(device)
        labels = labels.to(device)

        # Forward pass
        preds = model3d(imgs)
        L = loss.competition_loss_row_norm(preds, labels)

        # Backprop
        L.backward()

        # Update parameters
        optimiser.step()

        # Zero gradients
        optimiser.zero_grad()

        # Track loss
        loss_acc += L.detach().item()
        train_count += 1

    # Update learning rate
    scheduler.step()

    # Don't update weights
    with torch.no_grad():
        # Validate
        for val_imgs, val_labels in dataloader.valid_loader:
            # Reshape
            val_imgs = val_imgs.to(device)
            val_labels = val_labels.to(device)

            # Forward pass
            val_preds = model3d(val_imgs)
            val_L = loss.competition_loss_row_norm(val_preds, val_labels)

            # Track loss
            val_loss_acc += val_L.item()
            valid_count += 1

    # Save loss history
    loss_hist.append(loss_acc / train_count)
    val_loss_hist.append(val_loss_acc / valid_count)

    # Print loss
    if (epoch + 1) % 1 == 0:
        print(
            f'Epoch {epoch + 1}/{N_EPOCHS}, loss {loss_acc / train_count:.5f}, val_loss {val_loss_acc / valid_count:.5f}')

    # Save model (& early stopping)
    if (val_loss_acc / valid_count) < best_val_loss:
        best_val_loss = val_loss_acc / valid_count
        patience_counter = 0
        print('Valid loss improved --> saving model')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model3d.state_dict(),
            'optimiser_state_dict': optimiser.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss_acc / train_count,
            'val_loss': val_loss_acc / valid_count,
        }, "Conv3DNet.pt")
    else:
        patience_counter += 1

        if patience_counter == PATIENCE:
            break

print('')
print('Training complete!')