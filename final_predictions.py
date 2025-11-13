import pandas as pd
import numpy as np
import joblib
import os
import logging
import time
import json # Necessário para ler o arquivo de métricas

# --- Importar a função de engenharia de features ---
# (Assumindo que src/features/build_features.py existe no seu projeto)
try:
    from src.features.build_features import engineer_features
except ImportError:
    logging.error("ERRO CRÍTICO: Não foi possível importar 'engineer_features'. Verifique o caminho.")
    engineer_features = None

# --- Configuração de Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
# --------------------------------

def run_batch_predictions():
    """
    Executa o pipeline de predição em lote no arquivo 'future_unseen_examples.csv'.
    Adiciona a Margem de Confiança (MAE) à saída.
    
    RETORNA:
        (pd.DataFrame, dict): Tupla contendo (DataFrame de resultados, dicionário de métricas).
    """
    if engineer_features is None:
        logger.error("Pipeline abortado. 'engineer_features' não foi importado.")
        return None, None

    logger.info("Iniciando pipeline de predição em LOTE...")
    start_time = time.time()

    # --- 1. Caminhos ---
    MODEL_PATH = 'models/model.joblib'
    METRICS_PATH = 'models/model_metrics.json' 
    UNSEEN_DATA_PATH = 'data/raw/future_unseen_examples.csv'
    DEMOGRAPHICS_PATH = 'data/raw/zipcode_demographics.csv'
    OUTPUT_DIR = 'data/predictions'
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'final_predictions.csv')

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 2. Carregar Modelo e Métricas ---
    mae_usd = 0
    metrics = {}
    
    try:
        logger.info(f"Carregando modelo de {MODEL_PATH}...")
        model = joblib.load(MODEL_PATH)
        model_features = model.feature_names_in_
        logger.info("Modelo carregado com sucesso.")
        
        # Carregar MAE (Erro Absoluto Médio) para a margem de confiança
        with open(METRICS_PATH, 'r') as f:
            metrics = json.load(f)
        mae_usd = metrics.get('mae_usd', 0)
        logger.info(f"Métrica MAE carregada: ${mae_usd:,.2f}")
        
    except Exception as e:
        logger.error(f"Erro ao carregar modelo ou métricas: {e}")
        mae_usd = 0 # Valor padrão em caso de erro
        metrics = {'erro': f'Falha ao carregar {METRICS_PATH}'}
        
    # Definir o valor da margem de confiança formatado para o CSV
    confidence_margin = f"+/- ${mae_usd:,.2f} (MAE)"

    # --- 3. Carregar Dados de Entrada ---
    try:
        logger.info(f"Carregando dados não vistos de {UNSEEN_DATA_PATH}...")
        unseen_df = pd.read_csv(UNSEEN_DATA_PATH)
        demographics_df = pd.read_csv(DEMOGRAPHICS_PATH)
    except FileNotFoundError as e:
        logger.error(f"Erro: Arquivo de dados não encontrado. {e}")
        return None, None
    
    # --- 4. TRATAMENTO DE ERROS DE COLUNAS FALTANTES ---
    
    # TRATAMENTO DE ID 
    if 'id' in unseen_df.columns:
        original_ids = unseen_df['id']
    else:
        logger.warning("Coluna 'id' não encontrada. Usando índice como ID temporário.")
        original_ids = unseen_df.index
        unseen_df['id'] = original_ids 
        
    # TRATAMENTO DE DATE
    if 'date' not in unseen_df.columns:
        logger.warning("Coluna 'date' não encontrada. Usando data fictícia '20150501T000000'.")
        unseen_df['date'] = '20150501T000000' 
        
    # --- 5. Processar Dados (Merge + Feature Engineering) ---
    logger.info("Fazendo merge com dados demográficos e engenharia de features...")
    if 'ZIP' in demographics_df.columns:
        demographics_df = demographics_df.rename(columns={'ZIP': 'zipcode'})
    
    merged_unseen_df = pd.merge(unseen_df, demographics_df, on='zipcode', how='left')

    # Aplicar a mesma engenharia de features do treino
    processed_unseen_df = engineer_features(merged_unseen_df)

    # --- 6. Alinhar Colunas e Prever ---
    final_unseen_df = processed_unseen_df.reindex(columns=model_features).fillna(0)

    logger.info(f"Realizando predições em {len(final_unseen_df)} registros...")
    predictions_log = model.predict(final_unseen_df)
    predictions_usd = np.expm1(predictions_log)

    # --- 7. Salvar Resultados (Com Margem de Confiança) ---
    logger.info("Salvando resultados...")
    
    # Criar o DataFrame final
    output_df = pd.DataFrame({
        'id': original_ids,
        'predicted_price_usd': predictions_usd,
        'confidence_margin': confidence_margin # Coluna de precisão adicionada!
    })
    
    # Formatar o preço para duas casas decimais
    output_df['predicted_price_usd'] = output_df['predicted_price_usd'].apply(lambda x: round(x, 2))
    
    output_df.to_csv(OUTPUT_FILE, index=False)
    
    elapsed = time.time() - start_time
    logger.info("="*50)
    logger.info(f"PREDIÇÕES EM LOTE CONCLUÍDAS (em {elapsed:.2f}s)")
    logger.info(f"Resultados salvos em: {OUTPUT_FILE}")
    logger.info("="*50)
    
    return output_df, metrics

if __name__ == "__main__":
    # A função agora retorna tanto as previsões quanto as métricas carregadas
    results_df, model_metrics = run_batch_predictions()
    
    # --- Exibir Resultados no Console ---
    if results_df is not None:
        
        # 1. Imprimir as primeiras 10 predições
        print("\n--- PRIMEIRAS 10 PREDIÇÕES (de future_unseen_examples.csv) ---")
        # Usando to_string() para evitar o erro 'tabulate'
        print(results_df.head(10).to_string(index=False))
        print("----------------------------------------------------------------\n")

        # 2. Imprimir as métricas do modelo (carregadas do arquivo JSON)
        print("\n--- MÉTRICAS DE PERFORMANCE DO MODELO (do conjunto de teste) ---")
        # Usar json.dumps para formatar o dicionário de forma legível
        print(json.dumps(model_metrics, indent=4))
        print("----------------------------------------------------------------\n")
        
        # 3. Informar sobre o gráfico
        print("\n--- GRÁFICO DE PERFORMANCE ---")
        print("Não é possível gerar um gráfico de performance para dados 'não vistos' (pois não temos o 'preço' real).")
        print("O gráfico relevante é o 'prediction_vs_actual.png' do seu relatório, que foi gerado no conjunto de teste.")
        print("--------------------------------\n")