import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2


# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
)

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

print(f"Using {device} device")

if device.type == "cuda":
    print(torch.cuda.get_device_name(device))

# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
# print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)



def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):   # batch: 0,1,2,3...
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    """在测试数据集上评估模型的准确率和平均损失。"""

    # 获取测试数据集中的样本总数。
    size = len(dataloader.dataset)

    # 获取 DataLoader 包含的批次数量。
    # 后面用它计算所有批次的平均损失。
    num_batches = len(dataloader)

    # 将模型切换到评估模式。
    # 这会关闭 Dropout 的随机丢弃行为，并让 BatchNorm 使用已保存的统计量。
    model.eval()

    # 初始化累计测试损失和预测正确的样本数量。
    test_loss, correct = 0, 0

    # 关闭自动微分。
    # 测试阶段不需要反向传播，因此不记录计算图可以减少内存占用并提高速度。
    with torch.no_grad():

        # 逐批次读取测试图片 X 和对应标签 y。
        for X, y in dataloader:

            # 将输入和标签移动到模型所在的 CPU 或 GPU。
            X, y = X.to(device), y.to(device)

            # 前向传播，得到每个样本对各个类别的预测分数。
            # 对 FashionMNIST 来说，pred 的形状通常是 [batch_size, 10]。
            pred = model(X)

            # 计算当前批次的损失。
            # .item() 将单元素张量转换为 Python 浮点数。
            # += 将各批次的损失累加起来。
            test_loss += loss_fn(pred, y).item()

            # pred.argmax(1) 获取每个样本预测分数最高的类别编号。
            # == y 将预测类别和真实类别逐个比较，得到布尔张量。
            # type(torch.float) 将 True/False 转为 1.0/0.0。
            # sum() 统计当前批次预测正确的样本数。
            # item() 将结果转换为普通 Python 数值。
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    # 用累计损失除以批次数，得到每个批次的平均损失。
    test_loss /= num_batches

    # 用预测正确的样本数除以样本总数，得到 0～1 之间的准确率。
    correct /= size

    # 输出百分比准确率和平均损失。
    # >0.1f：保留一位小数。
    # >8f：以浮点数显示，字段宽度至少为 8。
    print(
        f"Test Error: \n"
        f" Accuracy: {(100 * correct):>0.1f}%, "
        f"Avg loss: {test_loss:>8f} \n"
    )



epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

# save model
torch.save(model.state_dict(), "checkpoints\model.pth")
print("Saved PyTorch Model State to model.pth")

# loading model
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("checkpoints\model.pth", weights_only=True))


# for pred
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')






