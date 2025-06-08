import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from modulo import aula3_modulo_mlops as sts
import mlflow
import joblib


#Dados Treino pata Inferencia
seismic_slice_clean = np.load("outputs/seismic_slice_clean.npy")
X = np.load ("outputs/X.npy")
y = np.load("outputs/y.npy")

# Carregando modelo treinado
ET= joblib.load("outputs/model.pkl")
sts.ET = ET

### Aplicar o modelo treinado e fazer inferencia
seis_prop_vector, seis_estimated = sts.transfer_to_seismic_scale(dados_sismicos=seismic_slice_clean, modelo = ET, X=X, y=y)

np.save("outputs/seis_estimated.npy", seis_estimated)
print ("Inferencia Pronta")


