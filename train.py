def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, fold_idx, num_classes):
    model.to(device)
    
    best_val_accuracy = 0.0
    early_stopping_patience = 5 
    epochs_no_improve = 0
    best_val_loss = float('inf') 

    print(f"\n--- Fold {fold_idx+1} Training ---")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_preds = []
        train_labels = []
        train_logits = [] 

        # 训练阶段
        for xray_images_list, mri_images_list, kl_scores in train_loader:
            xray_images_list = [img.to(device) for img in xray_images_list]
            mri_images_list = [img.to(device) for img in mri_images_list]
            kl_scores = kl_scores.to(device)

            optimizer.zero_grad()
            outputs = model(xray_images_list, mri_images_list)
            loss = criterion(outputs, kl_scores)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * kl_scores.size(0) 
            
            _, predicted = torch.max(outputs.data, 1)
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(kl_scores.cpu().numpy())
            train_logits.extend(outputs.detach().cpu().numpy())

        epoch_loss = running_loss / len(train_loader.dataset) 
        
        train_accuracy = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average='weighted', zero_division=0)
        train_precision = precision_score(train_labels, train_preds, average='weighted', zero_division=0)
        train_recall = recall_score(train_labels, train_preds, average='weighted', zero_division=0)
        train_mae = mean_absolute_error(train_labels, train_preds)
        train_kappa = cohen_kappa_score(train_labels, train_preds)
        # 计算训练集AUC
        try:
            train_probas = torch.softmax(torch.tensor(train_logits), dim=1).numpy()
            train_auc = roc_auc_score(train_labels, train_probas, multi_class='ovr', average='weighted')
        except ValueError as e:
            train_auc = np.nan # If only one class is present in y_true, AUC is undefined
            print(f"Warning: Could not calculate train AUC for fold {fold_idx+1} epoch {epoch+1}: {e}")


        print(f"Fold {fold_idx+1} | Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {epoch_loss:.4f}, Acc: {train_accuracy:.4f}, F1: {train_f1:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, MAE: {train_mae:.4f}, Kappa: {train_kappa:.4f}, AUC: {train_auc:.4f}")


        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        val_logits = [] 

        with torch.no_grad():
            for xray_images_list, mri_images_list, kl_scores in val_loader:
                xray_images_list = [img.to(device) for img in xray_images_list]
                mri_images_list = [img.to(device) for img in mri_images_list]
                kl_scores = kl_scores.to(device)
                
                outputs = model(xray_images_list, mri_images_list)
                loss = criterion(outputs, kl_scores)
                val_loss += loss.item() * kl_scores.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(kl_scores.cpu().numpy())
                val_logits.extend(outputs.cpu().numpy()) 

        val_epoch_loss = val_loss / len(val_loader.dataset) 
        
        val_accuracy = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='weighted', zero_division=0)
        val_precision = precision_score(val_labels, val_preds, average='weighted', zero_division=0)
        val_recall = recall_score(val_labels, val_preds, average='weighted', zero_division=0)
        val_mae = mean_absolute_error(val_labels, val_preds)
        val_kappa = cohen_kappa_score(val_labels, val_preds)
        try:
            val_probas = torch.softmax(torch.tensor(val_logits), dim=1).numpy()
            val_auc = roc_auc_score(val_labels, val_probas, multi_class='ovr', average='weighted')
        except ValueError as e:
            val_auc = np.nan 
            print(f"Warning: Could not calculate validation AUC for fold {fold_idx+1} epoch {epoch+1}: {e}")


        print(f"  Val Loss: {val_epoch_loss:.4f}, Acc: {val_accuracy:.4f}, F1: {val_f1:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, MAE: {val_mae:.4f}, Kappa: {val_kappa:.4f}, AUC: {val_auc:.4f}")

        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            epochs_no_improve = 0
            best_val_accuracy = val_accuracy 
        else:
            epochs_no_improve += 1
            if epochs_no_improve == early_stopping_patience:
                print(f"Early stopping triggered for Fold {fold_idx+1} at Epoch {epoch+1}.")
                break

    cm = confusion_matrix(val_labels, val_preds, labels=range(num_classes))
    print(f"\nConfusion Matrix for Fold {fold_idx+1} (Validation Set):")
    cm_df = pd.DataFrame(cm, index=[f'True {i}' for i in range(num_classes)], columns=[f'Pred {i}' for i in range(num_classes)])
    print(cm_df)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(num_classes), yticklabels=range(num_classes))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - Fold {fold_idx+1}')
    plt.show()

    if num_classes > 1 and len(np.unique(val_labels)) > 1:
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        # 转换为one-hot编码的标签
        from sklearn.preprocessing import label_binarize
        binarized_val_labels = label_binarize(val_labels, classes=range(num_classes)) 
        val_probas = torch.softmax(torch.tensor(val_logits), dim=1).numpy()
        
        plt.figure(figsize=(10, 8)) 
        if num_classes > 2: 
            for i in range(num_classes):
                if np.sum(binarized_val_labels[:, i]) > 0:
                    fpri, tpri, _ = roc_curve(binarized_val_labels[:, i], val_probas[:, i])
                    roc_auci = auc(fpri, tpri)
                    fpr[f"class_{i}"] = fpri
                    tpr[f"class_{i}"] = tpri
                    roc_auc[f"class_{i}"] = roc_auci
                    plt.plot(fpri, tpri, lw=2,
                             label=f'ROC curve of class {i} (area = {roc_auci:.2f})')
            
            all_fpr = np.unique(np.concatenate([fpr[f"class_{i}"] for i in range(num_classes) if f"class_{i}" in fpr]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(num_classes):
                if f"class_{i}" in fpr:
                    mean_tpr += np.interp(all_fpr, fpr[f"class_{i}"], tpr[f"class_{i}"])
            mean_tpr /= len([f"class_{i}" for i in range(num_classes) if f"class_{i}" in fpr]) 
            fpr["micro"] = all_fpr
            tpr["micro"] = mean_tpr
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            plt.plot(fpr["micro"], tpr["micro"],
                     label='Micro-average ROC curve (area = {0:0.2f})'
                           ''.format(roc_auc["micro"]),
                     color='deeppink', linestyle=':', linewidth=4)
            
        else: 
            fpri, tpri, _ = roc_curve(val_labels, val_probas[:, 1])
            roc_auci = auc(fpri, tpri)
            plt.plot(fpri, tpri, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auci:.2f})')
            
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves - Fold {fold_idx+1} (Validation Set)')
        plt.legend(loc="lower right")
        plt.show()
    else:
        print(f"Skipping ROC curve plot for fold {fold_idx+1}: Not enough classes or only one class in validation labels.")

    return {
        'val_accuracy': val_accuracy,
        'val_f1': val_f1,
        'val_precision': val_precision,
        'val_recall': val_recall,
        'val_mae': val_mae,
        'val_kappa': val_kappa,
        'val_auc': val_auc 
    }