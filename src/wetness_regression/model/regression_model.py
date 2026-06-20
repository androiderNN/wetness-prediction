import torch
import torch.nn as nn
import torchvision.models as models # 一般的な事前学習済みモデルのためにtorchvisionを想定


def get_model_input_size(model_name: str) -> int:
    """モデル名から想定する入力解像度（一辺）を返す。"""
    model_input_size = {
        "resnet18": 224,
        "vit_b_16": 224,
        "swin_t": 224,
        "efficientnet_b1": 240,
    }

    if model_name not in model_input_size:
        raise ValueError(
            f"supported models are ['resnet18'、'vit_b_16'、'swin_t'、'efficientnet_b1']"
        )

    return model_input_size[model_name]

def _build_regression_head(
    num_ftrs: int,
    num_output_features: int,
    dropout_rate: float,
    bottleneck_dim: int,
) -> nn.Sequential:
    """回帰ヘッドを構築する。bottleneck_dim > 0 の場合は中間層を挟む。"""
    layers: list[nn.Module] = [nn.Dropout(dropout_rate)]

    if bottleneck_dim > 0:
        layers.append(nn.Linear(num_ftrs, bottleneck_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(bottleneck_dim, num_output_features))
    else:
        layers.append(nn.Linear(num_ftrs, num_output_features))

    return nn.Sequential(*layers)


class RegressionModel(nn.Module):
    def __init__(
        self,
        pretrained_model_name,
        freeze_backbone=True,
        num_output_features=1,
        dropout_rate=0.0,
        bottleneck_dim=0,
    ):
        super().__init__()

        # 事前学習済みモデルをロード
        if pretrained_model_name == 'resnet18':
            self.base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif pretrained_model_name == 'vit_b_16':
            self.base_model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        elif pretrained_model_name == 'swin_t':
            self.base_model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        elif pretrained_model_name == 'efficientnet_b1':
            self.base_model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)
        else:
            raise ValueError(f"model {pretrained_model_name} not defined.")

        # ベースモデルの全てのパラメータを凍結（必要時のみ）
        if freeze_backbone:
            for param in self.base_model.parameters():
                param.requires_grad = False

        # 最終層を回帰ヘッドに置き換え
        if pretrained_model_name == 'resnet18':
            num_ftrs = self.base_model.fc.in_features
            self.base_model.fc = _build_regression_head(
                num_ftrs, num_output_features, dropout_rate, bottleneck_dim,
            )
        elif pretrained_model_name == 'vit_b_16':
            num_ftrs = self.base_model.heads.head.in_features
            self.base_model.heads.head = _build_regression_head(
                num_ftrs, num_output_features, dropout_rate, bottleneck_dim,
            )
        elif pretrained_model_name == 'swin_t':
            num_ftrs = self.base_model.head.in_features
            self.base_model.head = _build_regression_head(
                num_ftrs, num_output_features, dropout_rate, bottleneck_dim,
            )
        elif pretrained_model_name == 'efficientnet_b1':
            num_ftrs = self.base_model.classifier[1].in_features
            self.base_model.classifier[1] = _build_regression_head(
                num_ftrs, num_output_features, dropout_rate, bottleneck_dim,
            )

    def forward(self, x):
        return self.base_model(x)


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, x, y):
        loss = torch.sqrt(self.criterion(x, y))
        return loss
