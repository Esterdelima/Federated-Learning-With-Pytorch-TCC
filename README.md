 # Federated-Learning-With-Pytorch-TCC

## Necessário criar ambiente conda:
conda create --name <nome-do-ambiente> python=3.8
conda activate <nome-do-ambiente>

## Instalando Dependências 
 pip install -r requirements.txt


 ### Em caso de manutenção, comandando para gerar o requirements atualizado: 
 pip freeze > requirements.txt


 ## Ordem de execução:
 1- python centralized.py (vai pegar os dados)
 
 ### Simultaneamente
 2- python server.py
 3- python client.py (Em um terminal)
 4- python client.py (Em outro terminal)


