import pandas as pd
import numpy as np
import joblib
import os
import logging
import json
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, List
from sklearn.metrics import mean_absolute_percentage_error
from fastapi.middleware.cors import CORSMiddleware  

# --- Configuração de Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIGURAÇÕES DE MLOPS ---
FEEDBACK_PATH = 'data/feedback_local'
MODEL_PATH = 'models/model.joblib'
METRICS_PATH = 'models/model_metrics.json' # <-- NOVO
ERROR_THRESHOLD_MAPE = 0.15 
# ------------------------------

# --- Importar Funções de Outros Scripts ---
try:
    from src.features.build_features import engineer_features
    from src.models.train_model import train as retrain_model 
    logger.info("Módulos de 'engineer_features' e 'retrain_model' importados.")
except ImportError as e:
    logger.error(f"ERRO CRÍTICO ao importar módulos do SRC: {e}")
    engineer_features = None
    retrain_model = None

# --- Carregamento do Modelo (Global e Recarregável) ---
model = None
model_features = None
model_metrics = {} # <-- NOVO: Guardar métricas na memória

def load_model_safely():
    """Tenta carregar o modelo E as métricas."""
    global model, model_features, model_metrics # <-- Adiciona métricas
    try:
        model = joblib.load(MODEL_PATH)
        model_features = model.feature_names_in_ 
        logger.info(f"Modelo {MODEL_PATH} carregado com sucesso.")
        
        # Tenta carregar as métricas
        with open(METRICS_PATH, 'r') as f:
            model_metrics = json.load(f)
        logger.info(f"Métricas {METRICS_PATH} carregadas: MAE {model_metrics.get('mae_usd_formatted')}")
        
        return True
    except Exception as e:
        logger.error(f"ERRO CRÍTICO ao carregar artefatos: {e}")
        # Se as métricas falharem, usa um padrão
        model_metrics = {"mae_usd_formatted": "$ (Métricas não encontradas)"}
        if model is None: # Se o modelo falhou, é crítico
            return False
        return True # Se só as métricas falharam, ainda podemos prever

# --- Definir o App FastAPI e Pydantic ---
app = FastAPI(
    title="API de Previsão de Preços com MLOps Inteligente",
    description="Implementação do ciclo de predição e retreino baseado em performance.",
    version="1.1.0"
)

# --- 2. ADICIONAR O MIDDLEWARE DE CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todas as origens (ex: file://, http://localhost)
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos os métodos (POST, GET, OPTIONS)
    allow_headers=["*"],  # Permite todos os cabeçalhos
)
# ----------------------------------------

# Carrega o modelo na inicialização
if not load_model_safely():
    logger.error("Servidor iniciando com modelo indisponível. Rode /retrain para corrigir.")

# (Resto do seu código... Classes Pydantic)
class HouseData(BaseModel):
    id: int
    date: str
    bedrooms: int
    bathrooms: float
    sqft_living: int
    sqft_lot: int
    floors: float
    waterfront: int
    view: int
    condition: int
    grade: int
    sqft_above: int
    sqft_basement: int
    yr_built: int
    yr_renovated: int
    zipcode: int
    lat: float
    long: float
    Mean_Income: float
    Education_Bachelors_or_Higher: float
    Population_Density: float

class FeedbackData(HouseData):
    ground_truth_price: float 

# --- Endpoint de Predição (/predict) ---
@app.post("/predict")
def predict_price(data: HouseData):
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo não carregado. Tente /retrain.")
    try:
        input_data_dict = data.dict()
        df = pd.DataFrame([input_data_dict])
        df_processed = engineer_features(df)
        df_final = df_processed.reindex(columns=model_features).fillna(0)
        prediction_log = model.predict(df_final)
        prediction_real = np.expm1(prediction_log[0])
        prediction_real_float = float(prediction_real) 
        
        # --- ATUALIZAÇÃO ---
        # Puxa a margem de confiança dinâmica do modelo carregado
        confidence_margin = model_metrics.get("mae_usd_formatted", "$ (Erro N/A)")
        
        return {
            "message": "Predição realizada com sucesso.",
            "predicted_price": prediction_real_float,
            "predicted_price_usd": f"${prediction_real_float:,.2f}",
            "confidence_margin_usd": confidence_margin # <-- RETORNO DINÂMICO
        }
    except Exception as e:
        logger.error(f"Erro na predição: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro interno no servidor: {e}")

# --- Endpoint de Feedback (/feedback) ---
@app.post("/feedback")
def receive_feedback(data: FeedbackData):
    os.makedirs(FEEDBACK_PATH, exist_ok=True)
    feedback_data = data.dict()
    timestamp = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
    file_name = os.path.join(FEEDBACK_PATH, f"feedback_{timestamp}_{data.id}.json")
    try:
        with open(file_name, 'w') as f:
            json.dump(feedback_data, f, indent=4)
        logger.info(f"Feedback salvo com sucesso em {file_name}")
        return {"message": "Feedback recebido com sucesso!", "file": file_name}
    except Exception as e:
        logger.error(f"Erro ao salvar feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro ao salvar feedback: {e}")

# --- Endpoint de Verificação de Performance ---
@app.post("/check-performance")
def check_model_performance(background_tasks: BackgroundTasks): # MANTEMOS O NOME, MAS MUDAMOS A LÓGICA
    logger.info("--- INICIANDO VERIFICAÇÃO DE PERFORMANCE ---")
    if not os.path.exists(FEEDBACK_PATH):
        return {"retrain_triggered": False, "message": "Nenhum dado de feedback para checar."}
    feedback_files = [f for f in os.listdir(FEEDBACK_PATH) if f.endswith('.json')]
    if not feedback_files:
        return {"retrain_triggered": False, "message": "Nenhum dado de feedback para checar."}

    feedback_data = []
    for filename in feedback_files:
        try:
            with open(os.path.join(FEEDBACK_PATH, filename), 'r') as f:
                feedback_data.append(json.load(f))
        except Exception as e:
            logger.warning(f"Erro ao ler arquivo de feedback {filename}: {e}")
    if not feedback_data:
        return {"retrain_triggered": False, "message": "Dados de feedback vazios."}

    try:
        y_true, y_pred = [], []
        for data in feedback_data:
            y_true.append(data['ground_truth_price'])
            df = pd.DataFrame([data])
            df_processed = engineer_features(df)
            df_final = df_processed.reindex(columns=model_features).fillna(0)
            prediction_log = model.predict(df_final)
            y_pred.append(np.expm1(prediction_log[0]))

        current_mape = mean_absolute_percentage_error(y_true, y_pred)
        logger.info(f"MAPE ATUAL (nos {len(y_true)} novos dados): {current_mape:.4f}")
        logger.info(f"LIMITE DE ERRO (Threshold): {ERROR_THRESHOLD_MAPE:.4f}")

        if current_mape > ERROR_THRESHOLD_MAPE:
            logger.warning(f"GATILHO ATIVADO! Erro atual ({current_mape:.4f}) > Limite ({ERROR_THRESHOLD_MAPE:.4f})")
            
            # --- ATUALIZAÇÃO: REMOVE BACKGROUND TASK ---
            # background_tasks.add_task(run_full_retrain_cycle) # <-- REMOVIDO
            
            # Executa o retreino AGORA (síncrono)
            # A API vai travar aqui por ~30-60s, o que é BOM para a simulação
            new_metrics = run_full_retrain_cycle()
            
            return {
                "retrain_triggered": True,
                "message": f"Erro (MAPE) de {current_mape:.2%} excedeu o limite. Retreino CONCLUÍDO!",
                "current_mape_pct": f"{current_mape:.2%}",
                "new_metrics": new_metrics # <-- RETORNA AS NOVAS MÉTRICAS
            }
        else:
            logger.info("Performance do modelo está estável.")
            return {
                "retrain_triggered": False,
                "message": f"Erro (MAPE) de {current_mape:.2%} está dentro do limite. Performance estável.",
                "current_mape_pct": f"{current_mape:.2%}"
            }
    except Exception as e:
        logger.error(f"Erro ao calcular performance: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro ao calcular performance: {e}")

# --- Função de Retreino (agora chamada pelo /check-performance) ---
def run_full_retrain_cycle():
    """
    Executa o ciclo MLOps: coleta, treina, recarrega e RETORNA as novas métricas.
    """
    global model, model_features, model_metrics # <-- Garante que vamos atualizar globais
    logger.info("--- INICIANDO CICLO DE RETREINO ---")
    try:
        original_df = pd.read_csv('data/interim/merged_data.csv')
    except Exception as e:
        logger.error(f"Erro ao carregar dados originais para retreino: {e}. Abortando.")
        return {"error": "Falha ao carregar dados originais."}

    feedback_files = [f for f in os.listdir(FEEDBACK_PATH) if f.endswith('.json')]
    if not feedback_files:
        logger.info("Retreino chamado, mas sem novos dados de feedback. Abortando.")
        return {"error": "Nenhum feedback para treinar."}

    feedback_data = []
    for filename in feedback_files:
        try:
            with open(os.path.join(FEEDBACK_PATH, filename), 'r') as f:
                feedback_data.append(json.load(f))
        except Exception as e:
            logger.warning(f"Erro ao ler arquivo de feedback {filename}: {e}")

    logger.info(f"Coletados {len(feedback_data)} novos pontos de feedback para o treino.")
    new_data_df = pd.DataFrame(feedback_data).rename(columns={'ground_truth_price': 'price'})
    updated_df = pd.concat([original_df, new_data_df], ignore_index=True, sort=False)
    temp_path = 'data/interim/temp_updated_merged_data.csv'
    updated_df.to_csv(temp_path, index=False)

    try:
        os.replace(temp_path, 'data/interim/merged_data.csv')
        
        # --- TREINAR NOVO MODELO ---
        # A função 'train()' agora retorna as métricas
        new_metrics = retrain_model() 
        
        # --- RECARREGAR O MODELO NA API ---
        if load_model_safely():
            logger.warning("--- SUCESSO: NOVO MODELO CARREGADO NA API! ---")
            archive_path = os.path.join(FEEDBACK_PATH, 'archive')
            os.makedirs(archive_path, exist_ok=True)
            for filename in feedback_files:
                os.rename(os.path.join(FEEDBACK_PATH, filename), os.path.join(archive_path, filename))
            logger.info(f"Arquivos de feedback processados movidos para {archive_path}")
            
            return new_metrics # Retorna as métricas do novo modelo
        else:
            logger.error("ERRO: Falha ao carregar novo modelo. Usando o modelo anterior.")
            return {"error": "Falha ao recarregar o modelo."}

    except Exception as e:
        logger.error(f"ERRO CRÍTICO no ciclo de retreino: {e}", exc_info=True)
        return {"error": str(e)}
    finally:
        # Restaura o arquivo original
        original_df.to_csv('data/interim/merged_data.csv', index=False)
        if os.path.exists(temp_path):
            os.remove(temp_path)
    logger.info("--- CICLO DE RETREINO CONCLUÍDO (COM RESTAURAÇÃO) ---")