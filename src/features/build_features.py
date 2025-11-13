# src/features/build_features.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def engineer_features(df):
    """
    Recebe um DataFrame (bruto ou merjado) e aplica 
    toda a engenharia de features SEGURA (row-wise).
    
    Esta função NÃO deve conter features que "aprendem" com os dados,
    como Scalers ou KMeans, para evitar a necessidade de um preprocessor.
    
    Retorna o DataFrame com as novas colunas.
    """
    print("Aplicando engenharia de features...")
    # Copia para evitar SettingWithCopyWarning
    df = df.copy()

    # --- 1. Criar Alvo ---
    # Isso é seguro, pois é o nosso alvo (y)
    if 'price' in df.columns:
        df['log_price'] = np.log1p(df['price'])

    # --- 2. Features de Data ---
    # Tenta encontrar o ano de análise. Se 'date' não existir (ex: na predição), 
    # usamos 2015 como padrão (ano máximo do dataset).
    if 'date' in df.columns:
        # Converte 'date' para datetime (pegando só a data)
        try:
            df['date_dt'] = pd.to_datetime(df['date'].str.split('T').str[0])
            analysis_year = df['date_dt'].dt.year.max()
        except:
            analysis_year = 2015 # Fallback
            
        if pd.isna(analysis_year):
            analysis_year = 2015
    else:
        analysis_year = 2015

    df['house_age'] = analysis_year - df['yr_built']
    
    df['time_since_renovation'] = np.where(
        df['yr_renovated'] == 0,
        df['house_age'], # Se nunca reformou
        analysis_year - df['yr_renovated'] # Se reformou
    )
    
    df['was_renovated'] = (df['yr_renovated'] > 0).astype(int)

    # --- 3. Features de Cômodos (do notebook 02) ---
    df['total_rooms'] = df['bedrooms'] + df['bathrooms']
    
    # Evitar divisão por zero se total_rooms for 0
    df['sqft_per_room'] = df['sqft_living'] / (df['total_rooms'] + 1e-6)
    
    
    print("Engenharia de features concluída.")
    return df

def run_build_features():
    """
    Script principal:
    1. Carrega dados 
    2. Chama engineer_features
    3. Define colunas finais (sem leakage)
    4. Faz o split
    5. Salva train_processed.csv e test_processed.csv
    """
    print("\nIniciando processo 'build_features'...")
    
    # --- 1. Caminhos ---
    INTERIM_PATH = 'data/interim'
    PROCESSED_PATH = 'data/processed'
    os.makedirs(PROCESSED_PATH, exist_ok=True)

    # --- 2. Carregar Dados Merjados ---
    try:
        df = pd.read_csv(os.path.join(INTERIM_PATH, 'merged_data.csv'))
    except FileNotFoundError:
        print(f"Erro: 'merged_data.csv' não encontrado em {INTERIM_PATH}.")
        print("Execute 'python src/data/ingest_data.py' primeiro.")
        return
    
    # --- 3. Aplicar Engenharia de Features ---
    df_processed = engineer_features(df)

    # --- 4. Definir Features Finais ---
    TARGET = 'log_price'
    
    # Colunas para excluir:
    # - 'price', 'log_price': Nossos alvos (não podem ser features)
    # - 'id', 'date': Identificadores ou dados brutos
    # - 'yr_built', 'yr_renovated': Substituídas por 'house_age' etc.
    excluded_cols = ['price', 'log_price', 'id', 'date', 'date_dt', 'yr_built', 'yr_renovated']
    
    # Garantir que apenas colunas existentes sejam "dropadas"
    cols_to_drop = [col for col in excluded_cols if col in df_processed.columns]
    
    # Nossas features são TUDO, exceto as excluídas
    FEATURES = [col for col in df_processed.columns if col not in cols_to_drop]

    # Manter apenas as colunas de features e o alvo
    final_cols = FEATURES + [TARGET]
    df_final = df_processed[final_cols].fillna(0) # Preencher NaNs (ex: demográficos)

    print(f"Número de features finais: {len(FEATURES)}")

    # --- 5. Split Train/Test ---
    # Fazemos o split APÓS criar todas as features seguras
    train_df, test_df = train_test_split(df_final, test_size=0.2, random_state=42)

    # --- 6. Salvar os dois arquivos ---
    train_path = os.path.join(PROCESSED_PATH, 'train_processed.csv')
    test_path = os.path.join(PROCESSED_PATH, 'test_processed.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\nProcesso 'build_features' concluído.")
    print(f"Conjunto de treino salvo em: {train_path} ({train_df.shape})")
    print(f"Conjunto de teste salvo em: {test_path} ({test_df.shape})")

if __name__ == "__main__":
    run_build_features()