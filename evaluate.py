import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score,f1_score, confusion_matrix
import torch
import json
import torch.nn.functional as F
def evaluate_model(model, loader, device):
    model.eval()
    y_true, y_probs ,y_preds= [], [],[]
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).view(-1, 1)
            logits = model(x)
            y = y.view(-1).long()
            probs = F.softmax(logits, dim=1).cpu().numpy()
            pred = np.argmax(probs, axis=1)
            y_probs.extend(probs[:, 1])
            y_preds.extend(pred)
            y_true.extend(y.cpu().numpy())

    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    y_preds = np.array(y_preds)
    print(f"AUC: {roc_auc_score(y_true, y_probs):.4f}")
    print(f"Accuracy: {accuracy_score(y_true, y_preds):.4f}")
    print(f"Precision: {precision_score(y_true, y_preds, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_true, y_preds, average='weighted'):.4f}")
    print(f"F1 Score: {f1_score(y_true, y_preds, average='weighted'):.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_preds))