import torch.nn as nn
import torchvision.models as models


class MultiTaskModel(nn.Module):
    """共通バックボーンに回帰ヘッド（含水率）と分類ヘッド（樹種）を持つマルチタスクモデル。"""

    def __init__(
        self,
        pretrained_model_name: str,
        num_species: int,
        freeze_backbone: bool = True,
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        # 事前学習済みバックボーンをロードし、最終層を Identity に置換（特徴抽出器として使用）
        if pretrained_model_name == "resnet18":
            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            num_ftrs = backbone.fc.in_features
            backbone.fc = nn.Identity()
        elif pretrained_model_name == "vit_b_16":
            backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
            num_ftrs = backbone.heads.head.in_features
            backbone.heads = nn.Identity()
        elif pretrained_model_name == "swin_t":
            backbone = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
            num_ftrs = backbone.head.in_features
            backbone.head = nn.Identity()
        elif pretrained_model_name == "efficientnet_b1":
            backbone = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)
            num_ftrs = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"model {pretrained_model_name} not defined.")

        self.backbone = backbone

        # バックボーンの凍結
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # 共有 Dropout
        self.dropout = nn.Dropout(dropout_rate)

        # タスク固有ヘッド
        self.regression_head = nn.Linear(num_ftrs, 1)  # 含水率
        self.classification_head = nn.Linear(num_ftrs, num_species)  # 樹種

    def forward(self, x):
        features = self.backbone(x)
        features = self.dropout(features)
        reg_out = self.regression_head(features)
        cls_out = self.classification_head(features)
        return reg_out, cls_out
