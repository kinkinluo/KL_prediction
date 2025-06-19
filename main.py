if __name__ == "__main__":
    xray_base_data_dir = 'autodl-tmp/X线104' 
    mri_base_data_dir = 'autodl-tmp/MRI104'   
    kl_scores_csv = 'autodl-tmp/kl_socres.csv' 

    batch_size = 8
    num_epochs = 20
    num_folds = 5 
    learning_rate = 1e-4
    num_classes = 5 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected_xray_knee_sides = ["1_JPG", "2_JPG"]
    selected_mri_sequences = ["PDWI-SAG", "PDWI-COR", "T2WI-SAG"] 
    num_xray_inputs = len(selected_xray_knee_sides)
    num_mri_inputs = len(selected_mri_sequences)
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.RandomHorizontalFlip(), 
        transforms.RandomRotation(degrees=15), 
        transforms.ColorJitter(brightness=0.1, contrast=0.1), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

    full_df = pd.read_csv(kl_scores_csv)
    
    full_dataset = KneeMultiModalDataset(full_df, xray_base_data_dir, mri_base_data_dir,
                                         mri_sequence_types=selected_mri_sequences, 
                                         xray_knee_sides=selected_xray_knee_sides,   
                                         transform=image_transform)

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_results = []

    print(f"\n--- Starting {num_folds}-Fold Cross-Validation ---")
    for fold_idx, (train_indices, val_indices) in enumerate(kf.split(full_dataset)):
        train_sub_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        val_sub_dataset = torch.utils.data.Subset(full_dataset, val_indices)
        train_loader = DataLoader(train_sub_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_sub_dataset, batch_size=batch_size, shuffle=False)
        model = MultiModalKneeModel(num_classes=num_classes, 
                                    num_xray_inputs=num_xray_inputs, 
                                    num_mri_inputs=num_mri_inputs,
                                    dropout_rate=0.5) 
        criterion = nn.CrossEntropyLoss() 
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5) 
        current_fold_metrics = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, fold_idx, num_classes)
        fold_results.append(current_fold_metrics)

    print("\n--- Cross-Validation Training Complete ---")
    print("Aggregated Results Across Folds:")
    avg_metrics = {metric: np.nanmean([res[metric] for res in fold_results if res[metric] is not np.nan]) for metric in fold_results[0]} 
    for metric, value in avg_metrics.items():
        print(f"Average {metric}: {value:.4f}")
    print("\n训练完成！")