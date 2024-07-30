import flwr as fl
import csv
from typing import Dict, List, Tuple, Optional

# Função de agregação personalizada
# - accuracies: Lista das acurácias ponderadas pelo número de exemplos em cada cliente.
# - examples: Lista do número de exemplos em cada cliente.
# - Retorna a acurácia média ponderada.
# def weighted_average(metrics):
#     accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
#     examples = [num_examples for num_examples, _ in metrics]

#     return {"accuracy:": sum(accuracies) /  sum(examples)}


# Função de agregação personalizada
def weighted_average(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

# Função para criar a estratégia com base no nome fornecido
def get_strategy(strategy_name: str):
    if strategy_name == "FedAvg":
        return fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average)
    elif strategy_name == "FedProx":
        return fl.server.strategy.FedProx(mu=0.1, evaluate_metrics_aggregation_fn=weighted_average)
    elif strategy_name == "SCAFFOLD":
        return fl.server.strategy.SCAFFOLD(evaluate_metrics_aggregation_fn=weighted_average)
    else:
        raise ValueError(f"Estratégia desconhecida: {strategy_name}")

# Função para armazenar métricas em um arquivo CSV
def save_metrics(strategy_name: str, round_num: int, metrics: Dict[str, float]):
    filename = f"{strategy_name}_metrics.csv"
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([round_num, metrics["accuracy"]])

# - server_address="0.0.0.0:8080": Define o endereço IP e a porta onde o servidor estará escutando as conexões dos clientes. 0.0.0.0 permite que o servidor aceite conexões de qualquer interface de rede.
# - config=fl.server.ServerConfig(num_rounds=3): Configura o servidor para executar 3 rodadas de aprendizado federado.
# - strategy=fl.server.strategy.FedAvg(): Define a estratégia de agregação dos parâmetros do modelo. FedAvg (Federated Averaging) é a estratégia padrão onde os parâmetros dos modelos treinados localmente são agregados pela média ponderada.
def start_server(strategy_name: str):
    strategy = get_strategy(strategy_name)
    
    # Callback para salvar métricas após cada rodada
    def on_evaluate(server_round: int, results: List[Tuple[int, Dict[str, float]]]):
        loss, metrics = results[0]
        save_metrics(strategy_name, server_round, loss, metrics["accuracy"])
    
    # Inicia o servidor Flower com a estratégia especificada
    client_manager = fl.server.client_manager.SimpleClientManager()
    server = fl.server.Server(client_manager=client_manager, strategy=strategy)
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        server=server
    )

if __name__ == "__main__":
    import sys
    strategy_name = sys.argv[1] if len(sys.argv) > 1 else "FedAvg"
    start_server(strategy_name)