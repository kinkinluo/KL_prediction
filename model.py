class MultiModalKneeModel(nn.Module):
    def __init__(self, num_classes, num_xray_inputs, num_mri_inputs, feature_dim=512, num_heads=8, num_encoder_layers=2, dropout_rate=0.5):
        super(MultiModalKneeModel, self).__init__()
        # Xray的CNN特征提取(ResNet-50)
        self.xray_cnn = models.resnet50(pretrained=True)
        self.xray_cnn.fc = nn.Identity() 

        # MRI的CNN特征提取 (ResNet-50)
        self.mri_cnn = models.resnet50(pretrained=True)
        self.mri_cnn.fc = nn.Identity() 
        #CNN映射到tranformer
        self.xray_proj = nn.Linear(2048, feature_dim) 
        self.mri_proj = nn.Linear(2048, feature_dim)

        self.dropout = nn.Dropout(dropout_rate)

        # Transformer
        total_tokens = num_xray_inputs + num_mri_inputs
        encoder_layer = TransformerEncoderLayer(d_model=feature_dim, nhead=num_heads, batch_first=True, dropout=dropout_rate) 
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.classifier = nn.Linear(feature_dim * total_tokens, num_classes) 

    def forward(self, xray_inputs, mri_inputs):
        all_projected_features = []
        for xray_input in xray_inputs:
            xray_features = self.xray_cnn(xray_input) 
            xray_features_proj = self.dropout(self.xray_proj(xray_features)) 
            all_projected_features.append(xray_features_proj)
        for mri_input in mri_inputs:
            mri_features = self.mri_cnn(mri_input) # (batch_size, 2048)
            mri_features_proj = self.dropout(self.mri_proj(mri_features)) 
            all_projected_features.append(mri_features_proj)

        combined_features = torch.stack(all_projected_features, dim=1)
        transformer_output = self.transformer_encoder(combined_features) 
        combined_features_flat = self.dropout(transformer_output.view(transformer_output.size(0), -1))
        output = self.classifier(combined_features_flat)
        return output