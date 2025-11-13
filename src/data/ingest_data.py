# src/data/ingest_data.py
import pandas as pd
import os

def run_ingestion():
    """
    Carrega os dados brutos, faz o merge e salva em data/interim.
    """
    print("Iniciando ingestão de dados...")
    
    # Caminhos
    RAW_PATH = 'data/raw'
    INTERIM_PATH = 'data/interim'
    
    # Certifique-se de que a pasta de destino exista
    os.makedirs(INTERIM_PATH, exist_ok=True)
    
    # Carregar dados brutos
    try:
        df_house = pd.read_csv(os.path.join(RAW_PATH, 'kc_house_data.csv'))
        df_demo = pd.read_csv(os.path.join(RAW_PATH, 'zipcode_demographics.csv'))
    except FileNotFoundError as e:
        print(f"Erro: Arquivo não encontrado. {e}")
        print("Certifique-se que 'kc_house_data.csv' e 'zipcode_demographics.csv' estão em data/raw/")
        return

    # Renomear coluna de merge em df_demo (ajuste se o nome for outro)
    if 'ZIP' in df_demo.columns:
        df_demo = df_demo.rename(columns={'ZIP': 'zipcode'})
    
    # Fazer o merge
    df_merged = pd.merge(df_house, df_demo, on='zipcode', how='inner')
    
    # Salvar dados merjados
    output_path = os.path.join(INTERIM_PATH, 'merged_data.csv')
    df_merged.to_csv(output_path, index=False)
    
    print(f"Dados merjados salvos com sucesso em: {output_path}")
    print(f"Formato dos dados: {df_merged.shape}")

if __name__ == "__main__":
    # Permite executar este script diretamente do terminal
    run_ingestion()