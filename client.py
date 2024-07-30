from collections import OrderedDict

import flwr as fl
import torch

from centralized import load_data, load_model, train, test

#Atualiza os parâmetros do modelo com os parâmetros fornecidos
def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


# Carrega o modelo e os dados de treinamento e teste
net = load_model()
trainloader, testloader = load_data()

# - FlowerClient herda de fl.client.NumPyClient.
# - get_parameters: Retorna os parâmetros do modelo como arrays NumPy.
# - fit: Atualiza os parâmetros do modelo, treina localmente e retorna os novos parâmetros.
# - evaluate: Atualiza os parâmetros do modelo, avalia no conjunto de teste e retorna a perda e a acurácia
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]
    
    def fit(self, parameters, config):
        set_parameters(net, parameters)
        train(net, trainloader, epochs=1)
        return self.get_parameters({}), len(trainloader.dataset), {}
    
    def evaluate(self, parameters, config):
        set_parameters(net, parameters)
        loss, accuracy = test(net, testloader)
        return float(loss), len(testloader.dataset), {"accuracy": accuracy}
    
# Inicia o cliente e conecta ao servidor federado
fl.client.start_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient().to_client(),
)