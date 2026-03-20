import torch
import torch.nn as nn
import torchvision.models as models # 一般的な事前学習済みモデルのためにtorchvisionを想定


def get_model_input_size(model_name: str) -> int:
    """モデル名から想定する入力解像度（一辺）を返す。"""
    model_input_size = {
        "resnet18": 224,
        "vit_b_16": 224,
        "swin_t": 224,
    }

    if model_name not in model_input_size:
        raise ValueError(
            f"supported models are ['resnet18'、'vit_b_16'、'swin_t']"
        )

    return model_input_size[model_name]

class RegressionModel(nn.Module):
    def __init__(self, pretrained_model_name, freeze_backbone=True, num_output_features=1):
        super().__init__()

        # 事前学習済みモデルをロード
        if pretrained_model_name == 'resnet18':
            self.base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif pretrained_model_name == 'vit_b_16':
            self.base_model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        elif pretrained_model_name == 'swin_t':
            self.base_model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        else:
            raise ValueError(f"model {pretrained_model_name} not defined.")

        # ベースモデルの全てのパラメータを凍結（必要時のみ）
        if freeze_backbone:
            for param in self.base_model.parameters():
                param.requires_grad = False

        # 最終層を回帰ヘッドに置き換え
        if pretrained_model_name == 'resnet18':
            num_ftrs = self.base_model.fc.in_features
            self.base_model.fc = nn.Linear(num_ftrs, num_output_features)
        elif pretrained_model_name == 'vit_b_16':
            num_ftrs = self.base_model.heads.head.in_features
            self.base_model.heads.head = nn.Linear(num_ftrs, num_output_features)
        elif pretrained_model_name == 'swin_t':
            num_ftrs = self.base_model.head.in_features
            self.base_model.head = nn.Linear(num_ftrs, num_output_features)

    def forward(self, x):
        raw_pred = self.base_model(x)
        return torch.sigmoid(raw_pred) * 100


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()

    def forward(self,x,y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss
