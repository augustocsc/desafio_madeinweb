import pandas as pd
import numpy as np
import joblib
import os
import logging
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# --- Configuração ---
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Caminhos ---
MODEL_PATH = 'models/model.joblib'
TEST_DATA_PATH = 'data/processed/test_processed.csv'
OUTPUT_IMAGE_DIR = 'docs/images' # Pasta para salvar os gráficos

# --- Função Auxiliar para Encontrar Nomes de Features (AGORA USADA) ---
def find_feature_name(df, partial_name):
    """Tenta encontrar o nome exato da coluna usando uma busca parcial (case-insensitive)."""
    # A prioridade é encontrar a coluna exata que o modelo usou
    if partial_name in df.columns:
        return partial_name
        
    for col in df.columns:
        if partial_name.lower() in col.lower():
            # Retorna o nome da coluna no DataFrame (o que o SHAP precisa)
            return col 
    return partial_name # Retorna o nome original se não for encontrado

def generate_analysis():
    """
    Carrega o modelo final e os dados de teste para
    gerar e salvar todos os gráficos e métricas
    necessários para os relatórios.
    """
    logger.info("Iniciando a geração de artefatos de análise...")
    
    # Criar pasta de saída
    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)

    # --- 1. Carregar Modelo e Dados ---
    try:
        model = joblib.load(MODEL_PATH)
        logger.info(f"Modelo {MODEL_PATH} carregado.")
        
        test_df = pd.read_csv(TEST_DATA_PATH)
        logger.info(f"Dados de teste {TEST_DATA_PATH} carregados.")
    except FileNotFoundError as e:
        logger.error(f"Erro: Arquivo não encontrado. {e}")
        logger.error("Certifique-se de que o modelo foi treinado ('train_model.py') e os dados processados ('build_features.py').")
        return

    # Separar X e y
    TARGET = 'log_price'
    y_test = test_df[TARGET]
    X_test = test_df.drop(columns=TARGET)
    
    # Garantir que as colunas do X_test correspondem ao modelo
    try:
        X_test = X_test[model.feature_names_in_]
    except AttributeError:
        logger.warning("Atributo 'feature_names_in_' não encontrado. O modelo pode ser de uma versão antiga do scikit-learn.")
    except KeyError as e:
        logger.error(f"Erro de coluna: {e}. O modelo foi treinado com features diferentes das do 'test_processed.csv'.")
        return


    # --- 2. Gerar Predições e Métricas ---
    logger.info("Calculando métricas finais...")
    y_pred_log = model.predict(X_test)
    
    # Reverter para preço real (dólares)
    y_test_price = np.expm1(y_test)
    y_pred_price = np.expm1(y_pred_log)

    # Métricas
    r2 = r2_score(y_test_price, y_pred_price)
    mae = mean_absolute_error(y_test_price, y_pred_price)
    rmse = np.sqrt(mean_squared_error(y_test_price, y_pred_price))
    mape = mean_absolute_percentage_error(y_test_price, y_pred_price) * 100

    # Imprimir métricas para o usuário copiar
    print("\n\n" + "="*50)
    print("--- MÉTRICAS FINAIS PARA OS RELATÓRIOS ---")
    print("="*50)
    print(f"R² (R-squared):           {r2:.4f}")
    print(f"MAE (Erro Médio em $):    {mae:,.2f}")
    print(f"RMSE (Erro Padrão em $):  {rmse:,.2f}")
    print(f"MAPE (Erro Percentual):   {mape:.2f}%")
    print("\nCOPIE OS VALORES ACIMA E COLE NOS ARQUIVOS .md\n" + "="*50 + "\n")

    # --- 3. Gerar Gráfico: Previsto vs. Real ---
    logger.info("Gerando gráfico: Previsto vs. Real...")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test_price, y=y_pred_price, alpha=0.3, s=15, color='#3b82f6') # Azul
    plt.plot([y_test_price.min(), y_test_price.max()], [y_test_price.min(), y_test_price.max()], 'r--', lw=2)
    plt.title('Valor Real vs. Valor Previsto (em $)', fontsize=16)
    plt.xlabel('Valor Real ($)', fontsize=12)
    plt.ylabel('Valor Previsto ($)', fontsize=12)
    plt.gca().ticklabel_format(style='plain', axis='both')
    
    plot_path = os.path.join(OUTPUT_IMAGE_DIR, 'prediction_vs_actual.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Gráfico salvo em: {plot_path}")

    # --- 4. Gerar Gráfico: Histograma de Erros ---
    logger.info("Gerando gráfico: Histograma de Erros...")
    errors = y_test_price - y_pred_price # Erro em dólares
    
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True, bins=50, color='#3b82f6')
    plt.axvline(x=0, color='red', linestyle='--')
    plt.title('Distribuição dos Erros de Predição (em $)', fontsize=16)
    plt.xlabel('Erro (Valor Real - Valor Previsto)', fontsize=12)
    plt.ylabel('Contagem', fontsize=12)
    plt.gca().ticklabel_format(style='plain', axis='x')
    
    plot_path = os.path.join(OUTPUT_IMAGE_DIR, 'error_distribution.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Gráfico salvo em: {plot_path}")

    # --- 5. Gerar Gráfico: Análise de Resíduos ---
    logger.info("Gerando gráfico: Análise de Resíduos...")
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test_price, y=errors, alpha=0.3, s=15, color='#3b82f6')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title('Análise de Resíduos (Erro vs. Preço Real)', fontsize=16)
    plt.xlabel('Valor Real ($)', fontsize=12)
    plt.ylabel('Erro (Resíduo em $)', fontsize=12)
    plt.gca().ticklabel_format(style='plain', axis='both')
    
    plot_path = os.path.join(OUTPUT_IMAGE_DIR, 'residuals_plot.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Gráfico salvo em: {plot_path}")

    # --- 6. Gerar Gráfico: SHAP Summary Plot ---
    logger.info("Calculando valores SHAP... (Isso pode demorar)")
    
    sample_size = min(1000, len(X_test))
    X_test_sample = X_test.sample(n=sample_size, random_state=1)
    
    explainer = shap.TreeExplainer(model)
    
    shap_values_raw = explainer.shap_values(X_test_sample) 

    logger.info("Gerando gráfico: SHAP Summary Plot...")
    plt.figure()
    
    shap.summary_plot(shap_values_raw, X_test_sample, show=False, max_display=15)
    
    fig = plt.gcf()
    fig.set_size_inches(10, 7)
    plot_path = os.path.join(OUTPUT_IMAGE_DIR, 'shap_summary.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Gráfico salvo em: {plot_path}")
    
    logger.info("Análise concluída. Todos os gráficos foram salvos em 'docs/images/'.")

if __name__ == "__main__":
    generate_analysis()