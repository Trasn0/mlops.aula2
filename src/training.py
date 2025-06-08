import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from modulo import aula3_modulo_mlops as sts
import mlflow
import mlflow.sklearn
import joblib

sim_clean = np.load ("outputs/sim_clean.npy")

sim_estimado, y, nrms_teste, r2_teste, mape_teste, dict_params, ET, X = sts.ML_model_evaluation(dados_simulacao=sim_clean, proporcao_treino=0.75)

joblib.dump (ET, "outputs/model.pkl")

np.save("outputs/X.npy",X)
np.save("outputs/y.npy",y)

mlflow.set_tracking_uri("http://127.0.0.1:5000")

with mlflow.start_run(run_name="training"):
    mlflow.log_params(dict_params)
    mlflow.log_metric("NRMS",nrms_teste)
    mlflow.log_metric("R2",r2_teste)
    mlflow.log_metric("MAPE",mape_teste)
    mlflow.sklearn.log_model(ET,"model")
    print ("MLflow tracking realizado")

print("Treino do ML pronto")
