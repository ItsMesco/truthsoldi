import pandas as pd

# 1. Carichiamo i due file
print("Caricamento archivi...")
df_truth = pd.read_csv('trump_truths_completo.csv')
df_twitter = pd.read_csv('tweets_01-08-2021.csv')

# 2. Uniformiamo le colonne (Il file Twitter usa spesso 'text' e 'date')
# Adattiamo i nomi per farli combaciare con quelli di Truth
df_twitter = df_twitter.rename(columns={
    'text': 'testo',
    'date': 'timestamp_utc'
})

# 3. Pulizia Date
# Assicuriamoci che entrambi siano in formato datetime UTC
df_truth['timestamp'] = pd.to_datetime(df_truth['timestamp_utc'], utc=True)
df_twitter['timestamp'] = pd.to_datetime(df_twitter['timestamp_utc'], utc=True)

# 4. Unione e ordinamento
df_totale = pd.concat([df_truth, df_twitter], ignore_index=True)
df_totale = df_totale[['timestamp', 'testo']].sort_values('timestamp').dropna()

print(f"Unione completata! Post totali: {len(df_totale)}")
print(f"Periodo coperto: dal {df_totale['timestamp'].min()} al {df_totale['timestamp'].max()}")

df_totale.to_csv('tutti_i_post_trump_uniti.csv', index=False)