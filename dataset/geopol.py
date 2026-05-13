# ============================================================
# SCRIPT STANDALONE — Arricchimento Geopolitico
# Prende il dataset già pronto e aggiunge le feature geo
# senza riscaricare nulla
# ============================================================
import pandas as pd
import numpy as np
import os

# ── 1. CARICA IL TUO DATASET ESISTENTE ───────────────────
print("📂 Caricamento dataset esistente...")

# Prova prima il parquet, poi il csv come fallback
if os.path.exists('dataset_IA_multi_ticker_completo.parquet'):
    df = pd.read_parquet('dataset_IA_multi_ticker_completo.parquet')
    print("   ✅ Caricato da Parquet")
elif os.path.exists('dataset_IA_multi_ticker_completo.csv'):
    df = pd.read_csv('dataset_IA_multi_ticker_completo.csv')
    print("   ✅ Caricato da CSV")
else:
    raise FileNotFoundError("❌ Nessun dataset trovato! Assicurati che il file sia nella stessa cartella.")

df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
print(f"   Righe totali: {len(df):,}")
print(f"   Periodo: {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")

# ── 2. LISTA EVENTI GEOPOLITICI ───────────────────────────
EVENTI_STORICI = [
    # (data_inizio, data_fine, nome, categoria, intensità 0-1)
    # ── Crisi Finanziarie ─────────────────────────────────
    ("2009-01-01", "2009-06-30", "Post-Lehman recovery",         "financial_crisis", 0.90),
    ("2010-05-06", "2010-05-06", "Flash Crash 2010",             "financial_crisis", 0.85),
    ("2011-08-01", "2011-08-31", "Crisi debito USA/Europa",      "financial_crisis", 0.80),
    ("2015-08-17", "2015-09-15", "Svalutazione Yuan/crash Cina", "financial_crisis", 0.75),
    ("2018-02-01", "2018-04-30", "Correzione mercati 2018",      "financial_crisis", 0.65),
    ("2020-02-20", "2020-04-30", "Crash COVID-19",               "financial_crisis", 1.00),
    ("2022-01-01", "2022-12-31", "Bear market 2022",             "financial_crisis", 0.70),
    ("2023-03-09", "2023-03-31", "Crollo SVB/Credit Suisse",     "financial_crisis", 0.65),
    # ── Trade War ─────────────────────────────────────────
    ("2018-03-01", "2018-12-31", "Trade War round 1",            "trade_war",        0.85),
    ("2019-05-05", "2019-06-30", "Escalation dazi Cina",         "trade_war",        0.90),
    ("2025-01-20", "2026-12-31", "Trade War mandato 2 Trump",    "trade_war",        0.95),
    # ── Guerre e Conflitti ────────────────────────────────
    ("2014-02-20", "2014-03-31", "Annessione Crimea",            "war_conflict",     0.70),
    ("2017-04-06", "2017-04-06", "Attacco missilistico Siria",   "war_conflict",     0.60),
    ("2019-09-14", "2019-09-14", "Attacco droni Arabia Saudita", "war_conflict",     0.65),
    ("2020-01-03", "2020-01-07", "Uccisione Soleimani",          "war_conflict",     0.80),
    ("2022-02-24", "2023-12-31", "Invasione Russia-Ucraina",     "war_conflict",     0.95),
    ("2023-10-07", "2024-06-30", "Conflitto Israele-Hamas",      "war_conflict",     0.80),
    # ── Terrorismo ────────────────────────────────────────
    ("2013-04-15", "2013-04-15", "Attentato Boston Marathon",    "terrorism",        0.60),
    ("2015-11-13", "2015-11-13", "Attentati Parigi",             "terrorism",        0.75),
    ("2016-07-14", "2016-07-14", "Attentato Nizza",              "terrorism",        0.55),
    # ── Elezioni ──────────────────────────────────────────
    ("2016-11-08", "2016-11-09", "Elezione Trump 2016",          "election",         0.90),
    ("2020-11-03", "2020-11-07", "Elezione Biden 2020",          "election",         0.85),
    ("2024-11-05", "2024-11-06", "Elezione Trump 2024",          "election",         0.90),
    # ── Pandemia ──────────────────────────────────────────
    ("2020-01-20", "2020-03-01", "Emergenza COVID dichiarata",   "pandemic",         0.95),
    # ── Shock Monetari ────────────────────────────────────
    ("2013-05-22", "2013-06-30", "Taper Tantrum Fed",            "monetary_shock",   0.70),
    ("2022-03-16", "2023-07-26", "Ciclo rialzi Fed post-COVID",  "monetary_shock",   0.85),
]

CATEGORIA_PESO = {
    'financial_crisis': 1.0,
    'trade_war':        0.9,
    'monetary_shock':   0.85,
    'war_conflict':     0.75,
    'pandemic':         1.0,
    'election':         0.80,
    'terrorism':        0.55,
}

df_eventi = pd.DataFrame([
    {
        'evento_start':     pd.Timestamp(s, tz='UTC'),
        'evento_end':       pd.Timestamp(e, tz='UTC'),
        'evento_nome':      nome,
        'evento_categoria': cat,
        'evento_intensita': intensita,
        'evento_peso_base': CATEGORIA_PESO[cat],
    }
    for s, e, nome, cat, intensita in EVENTI_STORICI
])
print(f"\n🌍 {len(df_eventi)} eventi geopolitici caricati")

# ── 3. CALCOLO CONTESTO (su timestamp unici = velocissimo) ──
print("\n⚙️  Calcolo contesto geopolitico (solo sui timestamp unici)...")

GEO_COLS = [
    'geo_financial_crisis',
    'geo_trade_war',
    'geo_war_conflict',
    'geo_monetary_shock',
    'geo_pandemic',
    'geo_election',
    'geo_terrorism',
    'geo_tensione_totale',
]

def assegna_contesto(ts):
    contesto = {col: 0.0 for col in GEO_COLS}
    for _, ev in df_eventi.iterrows():
        if ev['evento_start'] <= ts <= ev['evento_end']:
            cat   = ev['evento_categoria']
            valore = ev['evento_intensita'] * ev['evento_peso_base']
            col   = f'geo_{cat}'
            if col in contesto:
                contesto[col] = max(contesto[col], valore)
            contesto['geo_tensione_totale'] += valore
    contesto['geo_tensione_totale'] = min(1.0, contesto['geo_tensione_totale'])
    return contesto

# Lavora solo sui timestamp unici → poi merge
# Con 45k post unici invece di 5M righe: da ore a secondi
ts_unici = df[['timestamp']].drop_duplicates().copy().reset_index(drop=True)
print(f"   Timestamp unici da processare: {len(ts_unici):,}  (invece di {len(df):,} righe)")

contesti = ts_unici['timestamp'].apply(assegna_contesto)
ts_unici = pd.concat([ts_unici, pd.DataFrame(contesti.tolist())], axis=1)

# ── 4. MERGE SUL DATASET COMPLETO ─────────────────────────
print("\n🔗 Merge sul dataset completo...")
df = df.merge(ts_unici, on='timestamp', how='left')
df[GEO_COLS] = df[GEO_COLS].fillna(0.0)

# ── 5. RIEPILOGO ──────────────────────────────────────────
print(f"\n📊 Riepilogo colonne geopolitiche aggiunte:")
for col in GEO_COLS:
    n_attivi = (df[col] > 0).sum()
    print(f"   {col:<28} → {n_attivi:>8,} righe attive  ({n_attivi/len(df)*100:.1f}%)")

# ── 6. SALVATAGGIO (sovrascrive il file con le nuove colonne) ──
print(f"\n💾 Salvataggio...")
out_parquet = 'dataset_IA_multi_ticker_completo.parquet'
out_csv     = 'dataset_IA_multi_ticker_geo.csv'  # CSV di backup con nome diverso

df.to_parquet(out_parquet, index=False, compression='snappy')
print(f"   ✅ Parquet aggiornato: {out_parquet}")
print(f"      Dimensione: {os.path.getsize(out_parquet)/1e6:.1f} MB")

print(f"\n✅ FATTO! Il tuo dataset ora ha {len(df.columns)} colonne totali.")
print(f"   Colonne geo aggiunte: {GEO_COLS}")
print(f"\n🔍 Anteprima di un post durante la trade war:")
sample = df[df['geo_trade_war'] > 0].iloc[0][['timestamp','testo','geo_trade_war','geo_tensione_totale']]
print(sample.to_string())