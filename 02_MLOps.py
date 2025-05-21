#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

## Desafio:
### Atuando como consultor para uma empresa, a mesma lhe forneceu um código legado de um projeto que não foi para frente com o time de analytics deles.
### A empresa é da área de óleo e gás e trabalha mapeando áreas com potencial para explorar.
### O projeto deles trata de tentar aumentar a granularidade(resolução) de um conjunto de dados inicial para um conjunto de dados final com "melhor resolução" que permita um mapeamento melhor.
### A empresa trabalha com um software comercial que produz resultados razoáveis, mas que é uma caixa preta e o time de negócios da empresa agora resolveu criar suas próprias soluções para ter mais controle e não precisar pagar mais a licença desse software e automatizar os processos.
### A empresa lhe forneceu os dados de treino, e os dados de inferência. Ambos em estrutura numpy array com coordenadas X,Y,Z,Propriedade(target).
### A empresa também lhe forneceu os dados do resultado gerado por eles com o software comercial, com o mesmo tipo de estrutura dos dados de treino e de inferencia, para que você compare com a solução criada por você.
### Cabe a você realizar experimentos novos que melhorem (em relação à solução do software comercial).
### Repare que a solução atual que já consta no código legado claramente apresenta artefatos estranhos, explore isso.


# In[2]:


#pip install segyio 
#from google.colab import drive
#drive.mount('/content/drive')


# In[3]:


### Importando Bicliotecas
import numpy as np
#import simtoseis_library as sts
from modulo import simtoseis_library as sts
import mlflow
import sys 
sys.path.append('simtoseis_library.py')


# In[4]:


#Dados Treino
dados_treino = np.load("sim_slice.npy")
dados_treino


# In[5]:


#Dados para Inferência ou Teste
dados_inferencia = np.load("seismic_slice.npy")
dados_inferencia


# In[6]:


#Dados de Referência para a modelagem(Software Comnercial)
dados_referencia_comercial = np.load("seismic_slice_GT.npy")
dados_referencia_comercial


# # ### Tratamento dos dados

# In[7]:


#Checando a quantidade original de dados
original_slice_shape = dados_treino.shape[0]
print(f"Original number of samples in simulation model: {original_slice_shape}")


# In[8]:


# Filtrando os dados
# Dados com valor =-99 devem ser eliminados do conjunto. Isto é um requisito do time de negocios.
filtered_slice = dados_treino[ dados_treino[:, -1] != -99.0 ]
print(f"Final number of samples after cleaning: {filtered_slice.shape[0]}")


# In[9]:


# Calculate and report the percentage of data removed
percentage_loss = ((original_slice_shape - filtered_slice.shape[0]) / original_slice_shape) * 100
print(f"Percentage loss: {round(percentage_loss, 2)}%")


# In[10]:


dados_treino = sts.simulation_data_cleaning(simulation_data = dados_treino, value_to_clean = -99.0)


# In[11]:


dados_treino = sts.simulation_nan_treatment(simulation = dados_treino, value = 0, method = 'replace')


# # ### Conversão de sinais

# In[12]:


dados_treino, dados_inferencia = sts.depth_signal_checking(
    simulation_data=dados_treino, 
    seismic_data=dados_inferencia)


# In[13]:


# ### Plotar os dados de treino
sts.plot_simulation_distribution(
    dados_treino, bins=35, 
    title="Distribuição dos Dados de Treino");


# # ### Treinamento/Validação do Modelo de ML

# In[14]:


proporcao_treino=0.75
dados_validacao, y, nrms_teste, r2_teste, mape_teste, modelo, dict_params, ET = sts.ML_model_evaluation(
    dados_simulacao=dados_treino, 
    proporcao_treino=proporcao_treino, 
    modelo="extratrees")


# # ### Inferência do Modelo ML treinado

# In[15]:


dados_estimados_prop_vector, dados_estimados = sts.transfer_to_seismic_scale(dados_sismicos=dados_inferencia)


# # ### Histograma dos dados de inferência

# In[16]:


sts.plot_simulation_distribution(dados_estimados, bins=35, title="Distribuição dos Dados de Inferência")


# # ### Calculo dos Residuos: Dados de Referencia(software comercial) - Dados da Inferência ML
# 

# In[17]:


dados_estimados_residual_final = sts.calcular_residuos(dados_estimados, dados_referencia_comercial)


# # ### Plotando resultados dos Resíduos

# In[18]:


sts.plot_simulation_distribution(dados_estimados_residual_final, bins=35, title="Distribuição dos Residuos")


# In[19]:


sts.plot_seismic_slice(dados_treino, title="Slice a ~5000m dos dados de treino")


# In[20]:


sts.plot_seismic_slice(dados_referencia_comercial, title="Slice a ~5000m do Resultado-Referência(software comercial)")


# In[21]:


sts.plot_seismic_slice(dados_estimados, title="Slice a ~5000m da Inferência ML")


# In[22]:


sts.plot_seismic_slice(dados_estimados_residual_final, title = "Slice a ~5000m - Residuo da Inferência")


# # ### MLFLOW Tracking

# In[23]:


# #No prompt do Anaconda, no Terminal do VSCode, ou Terminal Python, digitar o comando abaixo para pegar a URL que gerencia a conexão com o MLFLOw(vide aula2):


# In[24]:


# mlflow ui


# In[25]:


# ### Criando uma lista com as métricas


# In[26]:


import mlflow
from mlflow.exceptions import MlflowException


# In[27]:


tuple_1 = ["nrms_teste", "r2_teste", "mape_teste"]


# In[28]:


metrics_tuple = [64.2, 0.58, 71.8]


# In[29]:


dict_metrics = dict(zip(tuple_1, metrics_tuple))
dict_metrics


# In[30]:


mlflow.set_tracking_uri("http://127.0.0.1:5000")


# In[31]:


# Configuração do experimento com tratamento de erro
try:
    mlflow.set_experiment("Aula02 Experimento1")
except MlflowException:
    # Se o experimento foi deletado, criamos um novo com o mesmo nome
    mlflow.create_experiment("Aula02 Experimento1")
    mlflow.set_experiment("Aula02 Experimento1")


# In[32]:


with mlflow.start_run(run_name="Aula2 MLOps"): #ID do Experimento
    mlflow.log_params(dict_params) # Registro de parametros
    mlflow.log_metrics(dict_metrics) # Métricas do experimento
    mlflow.sklearn.log_model(ET,"Model ExtraTrees") # Modelo ML utilizado


# In[33]:


#with mlflow.start_run(run_name="SimToSeis_ET_Model"):
#    mlflow.log_param("train_ratio", proporcao_treino)
#    modelo, x, y, nrms, r2, mape = sts.treinar_modelo (sim_data, proporcao_treino)
#    dados_validacao, y, nrms_teste, r2_teste, mape_teste, modelo, dict_params, ET


# In[34]:


# ### FIM

