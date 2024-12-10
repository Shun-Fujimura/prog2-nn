import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
import models

# モデルをインスタンス化する
model = models.MyModel()
print(model)

# データセットのロード
ds_train = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),  # PIL画像をTensorに変換
        transforms.Lambda(lambda x: x.to(torch.float32))  # float32に変換
    ])
)

# imageは Tensor に変換済み
image, target = ds_train[0]

# (1, H, W) から (1, 1, H, W) に次元を上げる
image = image.unsqueeze(dim=0)

# モデルに入れて結果(logits)を出す
model.eval()
with torch.no_grad():
    logits = model(image)

# クラス確率を計算
probs = logits.softmax(dim=1)

# 予測クラスを取得
predicted_class = logits.argmax(dim=1).item()

# クラス名のリスト
class_names = datasets.FashionMNIST.classes

# 入力画像とクラス確率を並べて表示
plt.figure(figsize=(12, 5))

# 入力画像
plt.subplot(1, 2, 1)
plt.imshow(image.squeeze(0).squeeze(0), cmap='gray_r')  # (1, 1, H, W) -> (H, W)
plt.title(f"class: {target} {datasets.FashionMNIST.classes[target]}")
#plt.axis('off')

# クラス確率グラフ
plt.subplot(1, 2, 2)
plt.bar(range(len(probs[0])), probs[0].numpy())
#plt.xticks(range(len(class_names)), class_names)
plt.ylim(0, 1)
plt.title(f"predicted class:{probs[0].argmax()}")
#plt.xlabel("Classes")
#plt.ylabel("Probability")
#plt.axvline(predicted_class, color='red', linestyle='--', label="Predicted Class")

plt.legend()

plt.tight_layout()
plt.show()
