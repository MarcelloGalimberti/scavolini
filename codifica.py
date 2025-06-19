# env neuralprophet

# import necessary libraries
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import plotly.express as px
warnings.filterwarnings('ignore')
from io import BytesIO
import streamlit as st
import plotly.graph_objects as go
import calendar

####### Impaginazione

st.set_page_config(layout="wide")

url_immagine = 'https://github.com/MarcelloGalimberti/scavolini/blob/main/Scavolini_logo.png?raw=true'

col_1, col_2 = st.columns([1, 5])

with col_1:
    st.image(url_immagine, width=200)

with col_2:
    st.title('Dashboard Ufficio Codifica')

###### Caricamento del file Excel
# Caricamento del file PPP 2026
st.header('Caricamento database codifica', divider='red')
uploaded_raw = st.file_uploader(
    "Carica file Impegni ufficio",
    #accept_multiple_files=True # caricarli tutti e poi selezionare 2026 (in un secondo momento, 2030 è diverso)
)
if not uploaded_raw:
    st.stop()

# Caricamento del file, specificando il nome del foglio e le righe da saltare
df_raw = pd.read_excel(uploaded_raw, sheet_name='database', skiprows=[0], header=0, parse_dates=True)
# Filtra per Data arrivo MDP > '2023-01-01'
df_raw = df_raw[df_raw['Arrivo Mdp'] >= '2023-01-01'].reset_index(drop=True)
# Colonne rilevanti
df_raw = df_raw[['AZ','Argomento','G.M.P.',	'Tempo','Stato','Priorità','Brenda','Gregori','Biondi','Peloso',
                 'Santucci','Marchionni','Nucci','Maccioni','Bellucci',	'Eusebi','Semprini',	
                 'Bravi','Puzzi','Ugoccioni','Arrivo Mdp','Avanzamento','Classificazione']]
# Filtra per Stato != 'C'
df_raw = df_raw[df_raw['Stato'] != 'C'].reset_index(drop=True)

df_ore_GMP = pd.read_excel(uploaded_raw, sheet_name='ore_GMP')


st.write('### Anteprima del database caricato')
st.dataframe(df_raw.head(10))

st.write('### Anteprima del database ore GMP')
st.dataframe(df_ore_GMP.head(10))


# Grafico percentuale tempo GMP per operatore

df_ore_GMP["% GMP"] = df_ore_GMP["% GMP"] * 100

fig = px.bar(
    df_ore_GMP,
    x="Op",
    y="% GMP",
    title="Percentuale del tempo per GMP per Operatore",
    labels={"Op": "Operatore", "% GMP": "Percentuale GMP"},
    color_discrete_sequence=["red"]
)

# Formatta l'asse y come percentuale
fig.update_layout(
    yaxis_tickformat=".0f%",
)

# Mostra in Streamlit
st.plotly_chart(fig, use_container_width=True)



# Sostituzione di ore/giorno con il valore corrispondente in df_raw

# Crea un dizionario: Op -> ore/giorno
op_to_hours = dict(zip(df_ore_GMP['Op'], df_ore_GMP['ore/giorno']))

# Trova le colonne di df_raw che corrispondono a Op
op_cols = [col for col in df_raw.columns if col in op_to_hours]

# Sostituisci 'x' con il valore corrispondente
for col in op_cols:
    df_raw[col] = df_raw[col].replace('X', op_to_hours[col])


df_raw['numero_op_assegnati'] = df_raw[op_cols].count(axis=1)


for col in op_cols:
    df_raw[col] = df_raw['Tempo']*(1-df_raw['Avanzamento'])/df_raw['numero_op_assegnati']/df_raw[col]


# Ora df_raw ha le sostituzioni richieste
#st.write('### Database elaborato con giorni di workload per operatore, considerndo il tempo e l\'avanzamento')
#st.dataframe(df_raw)


# Database GMP assegnate
df_A = df_raw[df_raw['Stato'] == 'A'].reset_index(drop=True)
st.write('### Database GMP assegnate e workload residuo in giorni')
st.dataframe(df_A)
# Database GMP in backlog
df_N = df_raw[df_raw['Stato'] == 'N'].reset_index(drop=True)
df_N = df_N[['AZ', 'Argomento', 'G.M.P.', 'Tempo', 'Stato', 'Priorità','Arrivo Mdp', 'Classificazione']]
st.write('### Database GMP in backlog')
st.dataframe(df_N)

# Grafico workload in giorni per operatore


# 1. Somma delle colonne op_cols
somme = df_A[op_cols].sum()

# 2. Crea un DataFrame per il grafico
df_plot = somme.reset_index()
df_plot.columns = ['Operatore', 'Giorni di lavoro']

# 3. Grafico a barre rosse
fig = px.bar(
    df_plot,
    x='Operatore',
    y='Giorni di lavoro',
    title='Giorni di lavoro per Operatore',
    color_discrete_sequence=['red']
)

#st.plotly_chart(fig, use_container_width=True)


# secondo grafico



# 1. Melt del dataframe
df_plot = df_A.melt(
    id_vars=['Classificazione', 'AZ', 'G.M.P.', 'Argomento'],
    value_vars=op_cols,
    var_name='Operatore',
    value_name='Giorni di lavoro'
)

# 2. Crea colonna per il tooltip personalizzato
df_plot['Tooltip'] = df_plot['AZ'].astype(str) + " | " + df_plot['G.M.P.'].astype(str) + " | " + df_plot['Argomento'].astype(str)

# 3. Grafico a barre impilate
fig = px.bar(
    df_plot,
    x='Operatore',
    y='Giorni di lavoro',
    color='Classificazione',
    title='Giorni di lavoro per Operatore (impilato per Classificazione)',
    custom_data=['Tooltip']  # per mostrare info extra nel tooltip
)

# Modifica il tooltip in modo che mostri solo la stringa desiderata
fig.update_traces(
    hovertemplate='%{customdata[0]}<br>Giorni di lavoro: %{y}<extra></extra>'
)

# Visualizza in Streamlit
st.plotly_chart(fig, use_container_width=True)

# Analisi backlog

# 1. Conta il numero di righe per ogni combinazione di Classificazione e AZ
df_count = df_N.groupby(['Classificazione', 'AZ']).size().reset_index(name='Conteggio')

# 2. Crea il grafico a barre
fig = px.bar(
    df_count,
    x='Classificazione',
    y='Conteggio',
    color='AZ',
    title='Numero di righe per Classificazione (colorato per AZ)',
    labels={'Classificazione': 'Classificazione', 'Conteggio': 'Numero righe', 'AZ': 'AZ'}
)

# 3. Visualizza in Streamlit
st.plotly_chart(fig, use_container_width=True)



# Crea la colonna per il tooltip personalizzato
df_N['Tooltip'] = df_N['G.M.P.'].astype(str) + " | " + df_N['Argomento'].astype(str)

# Grafico a barre impilate
fig = px.bar(
    df_N,
    x='Classificazione',
    y='Tempo',
    color='AZ',
    custom_data=['Tooltip'],
    title='Tempo per Classificazione (impilato per AZ)',
    labels={'Tempo': 'Tempo', 'Classificazione': 'Classificazione', 'AZ': 'AZ'}
)

# Personalizza il tooltip
fig.update_traces(
    hovertemplate='%{customdata[0]}<br>Tempo: %{y}<extra></extra>'
)

# Visualizza in Streamlit
st.plotly_chart(fig, use_container_width=True)


