import time
import torch
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
def train_model(model, loaders, optimizer, criterion, device, max_epochs=50, patience=15, save_path="best_model.pt"):
    train_loader, val_loader = loaders
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    scaler = GradScaler()
    best_acc = 0.0
    patience_counter = 0
    train_losses, val_losses = [], []
    train_acc, val_acc = [], []
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        all_preds, all_targets = [], []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device).view(-1, 1)
            optimizer.zero_grad()
            with autocast():
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

            logits = model(x)
            y = y.view(-1).long()
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

        train_loss = epoch_loss / len(train_loader)
        train_accuracy = accuracy_score(all_targets, all_preds)
        train_losses.append(train_loss)
        train_acc.append(train_accuracy)

        model.eval()
        val_loss = 0
        val_labels, val_probs = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device).view(-1, 1)
                with autocast():
                    logits = model(x)
                    loss = criterion(logits, y)
                val_loss += loss.item()
                logits = model(x)
                y = y.view(-1).long()
                val_loss += criterion(logits, y).item()
                preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
                val_probs.extend(preds.cpu().numpy())
                val_labels.extend(y.cpu().numpy())  # shape: [N]
        val_losses.append(val_loss / len(val_loader))
        val_accuracy = accuracy_score(val_labels, val_probs)
        val_acc.append(val_accuracy)
        scheduler.step(val_accuracy)

        print(f"Epoch {epoch + 1}, "
              f"Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_acc[-1]:.4f}, "
              f"Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_acc[-1]:.4f}")
        log_file = save_path.replace(".pt", ".out")
        with open(log_file, "a") as f:
            f.write(f"Epoch {epoch + 1}, "
                    f"Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_acc[-1]:.4f}, "
                    f"Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_acc[-1]:.4f}\n")
        torch.save(model.state_dict(), save_path)
        # # --- Early Stopping ---
        if val_acc[-1] > best_acc:
            best_acc= val_acc[-1]
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⏹️ Early stopping at epoch {epoch+1}")
                break