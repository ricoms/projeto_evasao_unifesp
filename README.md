# Análise de Evasão da UNIFESP

Projeto de análise de evasão na UNIFESP com licença [MIT](https://github.com/ricoms/projeto_evasao_unifesp/blob/master/LICENSE). A base de dados não está neste repositório pois é de propriedade da Profa. Dra. Juliana Cespedes do Instituto de Ciência e Tecnologia.

## Projeto

Os arquivos principais são [Previsão de desistentes.ipynb](https://github.com/ricoms/projeto_evasao_unifesp/blob/master/Previs%C3%A3o%20de%20desistentes.ipynb) que contém o código e análises feitas para treinar modelos para realizar a previsão entre alunos da classe SIM e NÃO para o atributo DESISTENTE, e o arquivo [Análiselise da base.ipynb](https://github.com/ricoms/projeto_evasao_unifesp/blob/master/An%C3%A1lise%20da%20base.ipynb).


## Requisistos para o projeto
#### python 3.5

#### lista extensa do environment

Os pacotes estarão listados no arquivo [requirements.txt](https://github.com/ricoms/projeto_evasao_unifesp/blob/master/requirements.txt) conforme o exemplo a seguir:

```python
ipython==5.1.0
jupyter==1.0.0
matplotlib==1.5.3
notebook==4.2.3
pandas==0.18.1
xlrd==1.0.0
scikit-learn==0.18.1
```


Daí, para instalar os pacotes você irá precisar de um ambiente virtual ativo em sua máquina.
Para instalar um gestor de ambientes virtuais em sua máquina (Ubuntu e similares) utilize os seguintes comandos no terminal:

```sh
sudo apt-get install python-pip python-dev
sudo pip install virtualenv virtualenvwrapper
echo "export WORKON_HOME=~/envs" >> ~/.bashrc
echo "source /usr/local/bin/virtualenvwrapper.sh" >> ~/.bashrc
echo "export PIP_REQUIRE_VIRTUALENV=true" >> ~/.bashrc
source ~/.bashrc
mkvirtualenv <env_name> -p /usr/bin/python3
```
Observação: substitua \<env_name\> pelo nome de preferência para seu ambiente virtual. Assim que tiver o ambiente virtual instalado, dentro dele, digite o comando abaixo:

```sh
pip install -r requirements.txt
```

Para sair do ambiente virtual utilize:

```sh
deactivate
```
Para entrar em um ambiente virtual já criado utilize:
```sh
workon <env_name>
```
onde \<env_name\> é o nome do seu ambiente virtual já existente.
