import cloudscraper
import json
import time
import pandas as pd
import os
from bs4 import BeautifulSoup

# Inizializziamo lo scraper
scraper = cloudscraper.create_scraper()

USERNAME = "realDonaldTrump"

# ==========================================
# INCOLLA QUI IL TUO TOKEN COPIATO DAL BROWSER
# ==========================================
TOKEN = os.getenv("TRUTHKEY")

HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Authorization": TOKEN 
}

def ottieni_id_account(username):
    print(f"Ricerca ID per @{username}...")
    url_lookup = f"https://truthsocial.com/api/v1/accounts/lookup?acct={username}"
    risposta = scraper.get(url_lookup, headers=HEADERS)
    if risposta.status_code == 200:
        return risposta.json().get("id")
    print(f"Errore ricerca utente: {risposta.status_code}")
    return None

def scarica_tutto_lo_storico(account_id):
    tutti_i_post = []
    max_id = None 
    url_statuses = f"https://truthsocial.com/api/v1/accounts/{account_id}/statuses"
    
    # ==========================================
    # NOVITÀ: CONTROLLO DEL BACKUP (RESUME)
    # ==========================================
    if os.path.exists('backup_emergenza.json'):
        print("Trovato file di backup precedente! Tento di riprendere da dove avevamo lasciato...")
        try:
            with open('backup_emergenza.json', 'r') as f:
                tutti_i_post = json.load(f)
            
            if len(tutti_i_post) > 0:
                # Prende l'ID dell'ULTIMO post salvato nel backup per ripartire da lì
                max_id = tutti_i_post[-1]["id"]
                data_ripresa = tutti_i_post[-1]['created_at'][:10]
                print(f"✅ Ripresa confermata! Ripartiamo dal post del: {data_ripresa}")
                print(f"Post già in cassaforte: {len(tutti_i_post)}\n")
        except Exception as e:
            print(f"Errore nella lettura del backup. Ricomincio da zero. Errore: {e}")
            tutti_i_post = []
            max_id = None
    # ==========================================

    print("\n--- INIZIO DOWNLOAD MASSIVO ---")
    print("Premi CTRL+C in qualsiasi momento per fermare in sicurezza e salvare i dati raccolti.\n")

    tentativi_falliti = 0

    try:
        while True:
            params = {
                "limit": 40,
                "exclude_replies": "true"
            }
            if max_id:
                params["max_id"] = max_id
                
            risposta = scraper.get(url_statuses, headers=HEADERS, params=params)
            
            if risposta.status_code == 429:
                print("⚠️ Rate Limit raggiunto! Pausa di 60 secondi...")
                time.sleep(60)
                continue
            elif risposta.status_code != 200:
                print(f"❌ Errore {risposta.status_code}. Riprovo tra 5 sec...")
                tentativi_falliti += 1
                if tentativi_falliti > 5:
                    print("Troppi errori consecutivi. Interruzione.")
                    break
                time.sleep(5)
                continue
                
            tentativi_falliti = 0
            post_scaricati = risposta.json()
            
            if not post_scaricati:
                print("\n✅ Raggiunto l'inizio del profilo! Tutti i post scaricati.")
                break
                
            tutti_i_post.extend(post_scaricati)
            max_id = post_scaricati[-1]["id"]
            data_corrente = post_scaricati[-1]['created_at'][:10]
            
            # Salva il checkpoint aggiornando il file
            if len(tutti_i_post) % 1000 < 40:
                with open('backup_emergenza.json', 'w') as f:
                    json.dump(tutti_i_post, f)
                print(f"💾 Checkpoint salvato! Totale: {len(tutti_i_post)} post (Siamo al: {data_corrente})")
            else:
                print(f"Scaricati {len(tutti_i_post)} post... (Siamo al: {data_corrente})")
            
            time.sleep(1.2) 
            
    except KeyboardInterrupt:
        print("\n🛑 Download interrotto manualmente. Ultimo salvataggio in corso...")
        with open('backup_emergenza.json', 'w') as f:
            json.dump(tutti_i_post, f)
        print("Dati messi in sicurezza nel file JSON.")

    return tutti_i_post
# ==========================================
# ESECUZIONE E CONVERSIONE IN CSV
# ==========================================
trump_id = ottieni_id_account(USERNAME)

if trump_id:
    # 1. Scarica tutti i post grezzi
    dati_grezzi = scarica_tutto_lo_storico(trump_id)

    if dati_grezzi:
        print(f"\nPulizia e formattazione di {len(dati_grezzi)} post...")
        dati_puliti = []
        
        for post in dati_grezzi:
            testo_html = post.get("content", "")
            # Gestione di post vuoti o solo media
            if not testo_html:
                testo_pulito = ""
            else:
                testo_pulito = BeautifulSoup(testo_html, "html.parser").get_text(separator=' ')
            
            dati_puliti.append({
                "id": post["id"],
                "timestamp_utc": post["created_at"],
                "testo": testo_pulito.replace('\n', ' ').strip(), # Rimuove ritorni a capo per un CSV più pulito
                "retruths": post.get("reblogs_count", 0),
                "likes": post.get("favourites_count", 0)
            })

        # 2. Crea il CSV Finale
        df = pd.DataFrame(dati_puliti)
        df.to_csv('trump_truths_completo.csv', index=False, encoding='utf-8')
        print(f"\n🎉 SUCCESSO! Salvato il file 'trump_truths_completo.csv' con {len(df)} post pronti per l'IA.")
else:
    print("Controlla il TOKEN, il login è fallito.")