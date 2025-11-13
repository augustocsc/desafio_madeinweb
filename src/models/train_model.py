import pandas as pd
import numpy as np
import joblib
import os
import logging
import time
import json  # <-- Importar JSON
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from xgboost import XGBRegressor
from scipy.stats import uniform, randint

# --- Configuração de Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
# ---------------------------------

def train():
    logger.info("Iniciando pipeline de treinamento ROBUSTO...")
    start_time = time.time()
    
    # --- 1. Caminhos e Carregamento ---
    PROCESSED_PATH = 'data/processed'
    MODEL_PATH = 'models'
    os.makedirs(MODEL_PATH, exist_ok=True)

    try:
        train_df = pd.read_csv(os.path.join(PROCESSED_PATH, 'train_processed.csv'))
        test_df = pd.read_csv(os.path.join(PROCESSED_PATH, 'test_processed.csv'))
    except FileNotFoundError:
        logger.error(f"Erro: Arquivos de treino/teste não encontrados em {PROCESSED_PATH}.")
        logger.error("Execute 'python src/features/build_features.py' primeiro.")
        return

    logger.info(f"Dados de treino carregados: {train_df.shape}")
    logger.info(f"Dados de teste carregados: {test_df.shape}")

    # --- 2. Definir Target e Features ---
    TARGET = 'log_price'
    
    # --- 2. Definir Target e Features ---
    TARGET = 'log_price'
    
    y_train = train_df[TARGET]
    X_train = train_df.drop(columns=TARGET)
    
    y_test = test_df[TARGET]
    X_test = test_df.drop(columns=TARGET)

    # --- 3. TUNING (Busca de Hiperparâmetros) ---
    logger.info("Iniciando busca de hiperparâmetros (RandomizedSearchCV)...")

    xgb = XGBRegressor(random_state=42, objective='reg:squarederror', n_jobs=-1)

    param_dist = {
        'n_estimators': randint(100, 500),
        'learning_rate': uniform(0.01, 0.1),
        'max_depth': randint(3, 8),
        'subsample': uniform(0.7, 0.3),
        'colsample_bytree': uniform(0.7, 0.3)
    }

    random_search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_dist,
        n_iter=10,  # Aumente para 20+ para melhores resultados
        cv=5,       
        scoring='neg_root_mean_squared_error',
        verbose=2, 
        random_state=42,
        n_jobs=-1 
    )

    random_search.fit(X_train, y_train)

    logger.info("Busca de hiperparâmetros concluída.")
    logger.info(f"Melhores parâmetros encontrados: {random_search.best_params_}")

    best_model = random_search.best_estimator_

    # --- 4. Relatório de Performance Detalhado ---
    logger.info("Gerando relatório de performance...")
    
    # Métricas de VALIDAÇÃO (Cross-Validation)
    # Este é o "erro médio" que o RandomizedSearch viu durante o tuning
    best_cv_score_rmse = -random_search.best_score_
    logger.info(f"Melhor RMSE (Validação CV, em log_price): {best_cv_score_rmse:.4f}")

    # Métricas de TESTE (No conjunto isolado)
    y_pred_log = best_model.predict(X_test)
    
    y_test_price = np.expm1(y_test)
    y_pred_price = np.expm1(y_pred_log)
    
    # Calcular métricas
    r2_final = r2_score(y_test_price, y_pred_price)
    rmse_final_usd = np.sqrt(mean_squared_error(y_test_price, y_pred_price))
    mae_final_usd = mean_absolute_error(y_test_price, y_pred_price)
    mape_final_pct = mean_absolute_percentage_error(y_test_price, y_pred_price) * 100

    # O Relatório Profissional
    logger.info("\n\n" + "="*50)
    logger.info("--- RELATÓRIO DE PERFORMANCE DO NOVO MODELO ---")
    logger.info("="*50)
    logger.info(f"Fonte dos Dados de Teste: {PROCESSED_PATH}/test_processed.csv")
    logger.info(f"Tamanho do Conjunto de Teste: {len(X_test)} amostras")
    logger.info("\n--- Métricas de Negócio (em Dólares) ---")
    logger.info(f"R² (R-squared):           {r2_final:.4f}")
    logger.info(f"MAE (Erro Médio em $):    ${mae_final_usd:,.2f}")
    logger.info(f"RMSE (Erro Padrão em $):  ${rmse_final_usd:,.2f}")
    logger.info(f"MAPE (Erro Percentual):   {mape_final_pct:.2f}%")
    logger.info("\n" + "-"*50 + "\n")


    # --- 5. Salvar o Modelo Final ---
    model_output_path = os.path.join(MODEL_PATH, 'model.joblib')
    joblib.dump(best_model, model_output_path)
    logger.info(f"Modelo final (otimizado) salvo com sucesso em: {model_output_path}")

    # --- 6. NOVO: Salvar Métricas ---
    metrics_output_path = os.path.join(MODEL_PATH, 'model_metrics.json')
    metrics = {
        "r2": r2_final,
        "rmse_usd": rmse_final_usd,
        "mae_usd": mae_final_usd,
        "mape_pct": mape_final_pct,
        "mae_usd_formatted": f"${mae_final_usd:,.2f}" # Salva o valor formatado
    }
    with open(metrics_output_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Métricas do modelo salvas com sucesso em: {metrics_output_path}")
    # --------------------------------

    elapsed = time.time() - start_time
    logger.info(f"Pipeline de treinamento concluído em {elapsed:.2f} segundos.")
    
    return metrics # Retorna as métricas para o ciclo de retreino

if __name__ == "__main__":
    train()