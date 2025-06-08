import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from modulo import aula3_modulo_mlops as sts
import mlflow
import joblib

sim_clean = np.load ("outputs/sim_clean.npy")

sim_estimado, y, nrms_teste, r2_teste, mape_teste, dict_params, ET, X = sts.ML_model_evaluation(dados_simulacao=sim_clean, proporcao_treino=0.75)

joblib.dump (ET, "outputs/model.pkl")

np.save("outputs/X.npy",X)
np.save("outputs/y.npy",y)

print("Treino do ML pronto")
