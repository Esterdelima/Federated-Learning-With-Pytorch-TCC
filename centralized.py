# - Importa bibliotecas necessárias: torch e torchvision para construção e treinamento de modelos de deep learning.
# - Compose, ToTensor, Normalize são usados para transformar e normalizar os dados.
# - DataLoader é usado para carregar dados de maneira eficiente.
# - CIFAR10 é o dataset de imagens usado.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

# Definição do dispositivo (GPU se disponível, caso contrário, CPU).
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Definição da rede neural
# - Contém duas camadas convolucionais (conv1 e conv2), uma camada de pooling (pool) e três camadas totalmente conectadas (fc1, fc2, fc3).
# - forward define a passagem direta da rede.
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Função de treinamento
# - Usa CrossEntropyLoss como função de perda.
# - Usa SGD (Gradiente Descendente Estocástico) como otimizador.
# - Treina a rede para o número especificado de épocas.
def train(net, trainloader, epochs):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

# Teste
# - Avaliação de desempenho da rede no conjunto de teste
# - Calcula a perda e a precisão
def test(net, testloader):
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return loss / len(testloader.dataset), correct / total

# Carregar os dados
# - Normaliza as imagens para ter valores de pixel entre -1 e 1
# - Retorna "DataLoader" para conjuntos de treino e teste
def load_data():
    trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = CIFAR10("./data", train=True, download=True, transform=trf)
    testset = CIFAR10("./data", train=False, download=True, transform=trf)
    return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset, batch_size=32, shuffle=False)

# def load_data(dataset_name='CIFAR10'):
#     if dataset_name == 'CIFAR10':
#         trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#         trainset = CIFAR10(root="./data", train=True, download=True, transform=trf)
#         testset = CIFAR10(root="./data", train=False, download=True, transform=trf)
    
#     elif dataset_name == 'MNIST':
#         trf = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
#         trainset = MNIST(root="./data", train=True, download=True, transform=trf)
#         testset = MNIST(root="./data", train=False, download=True, transform=trf)
    
#     elif dataset_name == 'FashionMNIST':
#         trf = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
#         trainset = FashionMNIST(root="./data", train=True, download=True, transform=trf)
#         testset = FashionMNIST(root="./data", train=False, download=True, transform=trf)
    
#     elif dataset_name == 'COCO':
#         trf = Compose([Resize((256, 256)), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#         trainset = CocoDetection(root="./data/COCO/train2017", annFile='./data/COCO/annotations/instances_train2017.json', transform=trf)
#         testset = CocoDetection(root="./data/COCO/val2017", annFile='./data/COCO/annotations/instances_val2017.json', transform=trf)
    
#     elif dataset_name == 'Caltech101':
#         trf = Compose([Resize((256, 256)), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#         trainset = Caltech101(root="./data", download=True, transform=trf)
#         testset = Caltech101(root="./data", download=True, transform=trf)

#     else:
#         raise ValueError(f"Dataset {dataset_name} is not supported")

#     return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset, batch_size=32, shuffle=False)

# Carregando o modelo da rede neural e movendo para o dispositivo
def load_model():
    return Net().to(DEVICE)

# Execução
if __name__ == "__main__":
    net = load_model()
    trainloader, testloader = load_data()
    train(net, trainloader, 5)
    loss, accuracy = test(net, testloader)
    print(f"Loss: {loss:.5f}, Accuracy: {accuracy:.3f}")
