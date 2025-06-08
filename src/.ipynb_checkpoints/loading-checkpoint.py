import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from modulo import aula3_modulo_mlops as sts
import mlflow


#Dados Treino
sim_slice = np.load("data/sim_slice.npy")

#Dados para Inferência
seismic_slice = np.load("data/seismic_slice.npy")

#Dados de Referência para a modelagem(Software Comnercial)
seismic_slice_GT = np.load("data/seismic_slice_GT.npy")


print ("Pronto")
 