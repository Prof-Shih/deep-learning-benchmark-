from torchvision import models


# EfficientNetV2-S
model_s = models.efficientnet_v2_s(
    weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
)

# EfficientNetV2-M
model_m = models.efficientnet_v2_m(
    weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1
)

# EfficientNetV2-L
model_l = models.efficientnet_v2_l(
    weights=models.EfficientNet_V2_L_Weights.IMAGENET1K_V1
)

model_s.eval()
model_m.eval()
model_l.eval()