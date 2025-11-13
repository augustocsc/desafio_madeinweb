import pandas as pd
import numpy as np
import joblib
import os
import logging
import time

# --- Configuração de Logging (Padrão da API) ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
# -------------------------------------------------

# --- 1. IMPORTAR a função de engenharia de features ---
# Isso garante que a lógica de predição e treino é idêntica.
try:
    from src.features.build_features import engineer_features
    logger.info("Função 'engineer_features' importada com sucesso.")
except ImportError:
    logger.error("ERRO CRÍTICO: Não foi possível importar 'engineer_features'.")
    logger.error("Verifique se 'src/features/build_features.py' existe.")
    engineer_features = None

# --- 2. Carregamento do Modelo (Simula o "início" da API) ---
MODEL_PATH = 'models'
MODEL_FILE = os.path.join(MODEL_PATH, 'model.joblib')

model = None
model_features = None

try:
    logger.info(f"Carregando modelo de {MODEL_FILE}...")
    model = joblib.load(MODEL_FILE)
    model_features = model.feature_names_in_ # Salva as features que o modelo espera
    logger.info("Modelo carregado com sucesso.")
except FileNotFoundError:
    logger.error(f"ERRO CRÍTICO: 'model.joblib' não encontrado em {MODEL_PATH}.")
    logger.error("Execute 'python src/models/train_model.py' primeiro.")
except Exception as e:
    logger.error(f"Erro inesperado ao carregar o modelo: {e}")
# -------------------------------------------------------------

def make_prediction(input_data_dict):
    """
    Recebe um DICIONÁRIO com os dados brutos de entrada (do usuário/API),
    aplica a engenharia de features e retorna a predição.
    """
    if model is None or engineer_features is None:
        logger.error("Tentativa de predição falhou: Modelo ou função de features não carregado.")
        return {"error": "Serviço de predição não está pronto."}
        
    try:
        start_time = time.time()
        
        # 1. Converter dicionário para DataFrame
        df = pd.DataFrame([input_data_dict])
        
        # 2. Aplicar a MESMA engenharia de features do treino
        # (Isso cria 'house_age', 'total_rooms', etc.)
        df_processed = engineer_features(df)
        
        # 3. Alinhar colunas: Garantir que o DataFrame tenha as mesmas
        # colunas, na mesma ordem, que o modelo foi treinado.
        # .reindex() preenche colunas faltantes com NaN
        df_final = df_processed.reindex(columns=model_features).fillna(0)

        # 4. Fazer a predição (no log_price)
        prediction_log = model.predict(df_final[model_features])
        
        # 5. Reverter para preço real
        prediction_real = np.expm1(prediction_log[0])
        
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        
        logger.info(f"Predição realizada com sucesso em {duration_ms:.2f} ms.")
        
        return {
            "predicted_price_usd": f"${prediction_real:,.2f}",
            "predicted_price": prediction_real
        }

    except Exception as e:
        logger.error(f"Erro durante a predição: {e}", exc_info=True)
        return {"error": str(e)}


if __name__ == "__main__":
    # Este é um EXEMPLO de como a API receberia os dados
    # Pegue uma linha do seu 'merged_data.csv' (antes do processamento) para testar
    sample_data = {
        'id': 7129300520, 'date': '20141013T000000', 'bedrooms': 3, 
        'bathrooms': 1.0, 'sqft_living': 1180, 'sqft_lot': 5650, 
        'floors': 1.0, 'waterfront': 0, 'view': 0, 'condition': 3, 
        'grade': 7, 'sqft_above': 1180, 'sqft_basement': 0, 
        'yr_built': 1955, 'yr_renovated': 0, 'zipcode': 98178, 
        'lat': 47.5112, 'long': -122.257, 
        # Dados demográficos do merge (o usuário teria que enviar o zipcode,
        # e a API faria o merge, mas para o teste do script, 
        # podemos passar os dados demográficos direto)
        'Mean_Income': 57321, 'Education_Bachelors_or_Higher': 21.0, 
        'Population_Density': 3774 
        # ... (adicione todas as colunas demográficas que seu merge gera)
    }
    
    logger.info("\n" + "---" * 10)
    logger.info("--- TESTE DE PREDIÇÃO LOCAL ---")
    
    prediction = make_prediction(sample_data)
    
    logger.info(f"Dados de entrada (Zipcode): {sample_data.get('zipcode')}")
    logger.info(f"Preço Previsto: {prediction.get('predicted_price_usd')}")
    logger.info("--- FIM DO TESTE ---")