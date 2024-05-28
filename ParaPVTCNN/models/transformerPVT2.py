import torch
import torch.nn as nn
import timm



class PVTTransformer(nn.Module):
    def __init__(self, model_name='pvt_v2_b2_li', pretrained=True, **kwargs):
        super().__init__()
        # Create the PVT model using timm
        model_args = dict(
        depths=(3, 3, 3, 3), embed_dims=(320, 2*320, 4*320, 8*320), num_heads=(1, 2, 5, 8), linear=True)
        self.transformer = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            **dict(model_args, **kwargs)
        )
        
        # Get the number of feature maps (channels) from the last stage of PVT
        self.feature_info = self.transformer.feature_info
        self.num_features = self.feature_info[-1]['num_chs']

        # Define a final linear layer for classification or other tasks
        self.classifier = nn.Linear(self.num_features, 9)  # Example for classification with 1000 classes

    def forward(self, x):
        features = self.transformer(x)
        # Assuming you want to use the last feature map
        x = features[-1]
        # Global average pooling
        x = x.mean(dim=[2, 3])
        x = self.classifier(x)
        return x

'''# Example usage
model = PVTTransformer()
input_tensor = torch.randn(1, 3, 224, 224)  # Example input
output = model(input_tensor)
print(output.shape)  # Should print torch.Size([1, 1000])'''

