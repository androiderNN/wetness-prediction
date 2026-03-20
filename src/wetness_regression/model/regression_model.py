import torch
import torch.nn as nn
import torchvision.models as models # 一般的な事前学習済みモデルのためにtorchvisionを想定

class RegressionModel(nn.Module):
    def __init__(self, num_output_features=1, pretrained_model_name='resnet18'):
        super().__init__()
        # 事前学習済みモデル（例: ResNet18）をロード
        if pretrained_model_name == 'resnet18':
            # 最新の利用可能な重みを使用
            self.base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            # ベースモデルの全てのパラメータを凍結
            for param in self.base_model.parameters():
                param.requires_grad = False
            # 最終的な分類層を回帰ヘッドに置き換え
            # ResNetの最終層は`fc`
            num_ftrs = self.base_model.fc.in_features
            self.base_model.fc = nn.Linear(num_ftrs, num_output_features)
        elif pretrained_model_name == 'vit_b_16':
            self.base_model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
            for param in self.base_model.parameters():
                param.requires_grad = False
            num_ftrs = self.base_model.heads.head.in_features
            self.base_model.heads.head = nn.Linear(num_ftrs, num_output_features)
        elif pretrained_model_name == 'swin_t':
            self.base_model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
            for param in self.base_model.parameters():
                param.requires_grad = False
            num_ftrs = self.base_model.head.in_features
            self.base_model.head = nn.Linear(num_ftrs, num_output_features)
        else:
            raise ValueError(f"事前学習済みモデル '{pretrained_model_name}' はサポートされていません。'resnet18'、'vit_b_16'、'swin_t'から選択してください。")

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
