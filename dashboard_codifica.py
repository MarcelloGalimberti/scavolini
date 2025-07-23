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
    st.title('Dashboard GMP Ufficio Codifica')

###### Caricamento del file Excel
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

st.header('Tempo assegnato agli operatori per le attività GMP', divider='red')

# Converti % GMP in percentuale se necessario
if df_ore_GMP['% GMP'].max() <= 1:
    df_ore_GMP['% GMP'] = df_ore_GMP['% GMP'] * 100

# Ordina le opzioni (opzionale)
df_ore_GMP = df_ore_GMP.sort_values('Op').reset_index(drop=True)

# Prepara i dati per il radar (chiudi il poligono)
categories = df_ore_GMP['Op'].tolist()
values = df_ore_GMP['% GMP'].tolist()
categories += [categories[0]]
values += [values[0]]

# Crea il radar chart con linee rosse
fig = go.Figure(
    data=[
        go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            line=dict(color='red', width=3),
            name='% GMP'
        )
    ]
)

fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            tickformat='.0f',
            ticksuffix='%',
            range=[0, max(100, max(values))]
        ),
        angularaxis=dict(
            tickfont=dict(size=20)  # Cambia qui la dimensione, ad esempio 20
        )
    ),
    showlegend=False,
    #title='% GMP per Op',
    width=800,
    height=800
)

# Visualizza in Streamlit
#st.plotly_chart(fig, use_container_width=True)

col_1, col_2 = st.columns([1, 1])

with col_1:
    st.write('#### Ore giornaliere per operatore')
    st.dataframe(df_ore_GMP)
    capacity_team = df_ore_GMP['ore/giorno'].sum()
    

with col_2:
    st.write('#### Radar chart % GMP per operatore',)
    st.plotly_chart(fig, use_container_width=True)


#st.write('#### Capacità totale del team in ore/giorno: ', capacity_team)
st.write(f'#### Capacità totale del team in ore/giorno: :green[{capacity_team:.1f}]')

st.header('Database GMP assegnate e workload residuo (in giorni)', divider='red')


# Funzione per scaricare xlsx
def to_excel_bytes(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Foglio1') #index=False,
    return output.getvalue()

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

# Database GMP assegnate
df_A = df_raw[df_raw['Stato'] == 'A'].reset_index(drop=True)
st.dataframe(df_A)

# Pulsante per scaricare xlsx
# Crea il bottone per scaricare il pivot ordinato
file_GMP_assegnate = to_excel_bytes(df_A)
st.download_button(
    label="📥 Scarica file GMP assegnate",
    data=file_GMP_assegnate,
    file_name='file_GMP_assegnate.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
)

# Grafico a barre impilate per giorni di lavoro per operatore
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
    custom_data=['Tooltip'],
    height=700  # per mostrare info extra nel tooltip
)

# Modifica il tooltip in modo che mostri solo la stringa desiderata
fig.update_traces(
    hovertemplate='%{customdata[0]}<br>Giorni di lavoro: %{y}<extra></extra>'
)

fig.update_layout(
    xaxis=dict(
        title_font=dict(size=22),      # Font del titolo asse x
        tickfont=dict(size=18)         # Font dei tick asse x (etichette Operatore)
    ),
    yaxis=dict(
        title_font=dict(size=22),      # Font del titolo asse y
        tickfont=dict(size=18)         # Font dei tick asse y
    ),
    title=dict(
        font=dict(size=28)             # Font del titolo
    ),
    legend=dict(
        font=dict(size=20)   # Qui cambi la dimensione del font della legenda
    ),
    legend_title=dict(
        font=dict(size=22)   # Qui imposti la dimensione del titolo della legenda
    )
)

# Calcola la somma per ogni Operatore
totali = df_plot.groupby('Operatore')['Giorni di lavoro'].sum().reset_index()

# Prendi la posizione di ciascun operatore
x_val = totali['Operatore']
y_val = totali['Giorni di lavoro']

# Aggiungi la traccia con le etichette sopra le barre
fig.add_scatter(
    x=x_val,
    y=y_val,
    mode='text',
    text=[f"{y:.1f}" for y in y_val],  # 1 decimale
    textposition='top center',
    textfont=dict(size=20, color='black', family='Arial'),
    showlegend=False,
    hoverinfo='skip'  # Non mostra nulla al passaggio del mouse
)

# Visualizza in Streamlit
st.plotly_chart(fig, use_container_width=True)


# Calcoli per il led time di codifica in processo

operatori_team = df_ore_GMP['Op'].tolist()
giorni_residui = df_A[operatori_team].sum().sum()
st.write(f'#### Giorni residui totali in processo di codifica: :green[{giorni_residui:.1f}]')

num_operatori = (df_ore_GMP['ore/giorno'] > 0).sum()
st.write(f'#### Operatori con ore/giorno > 0: :green[{num_operatori:.0f}]')

st.write(f'#### Tempo di attraversamento medio processo codifica (in giorni):  :green[{(giorni_residui / num_operatori):.1f}]') 


st.header('Analisi del Backlog', divider='red')

# Database GMP in backlog
df_N = df_raw[df_raw['Stato'] == 'N'].reset_index(drop=True)
df_N = df_N[['AZ', 'Argomento', 'G.M.P.', 'Tempo', 'Stato', 'Priorità','Arrivo Mdp', 'Classificazione']]
st.write('### Database GMP in backlog')
st.dataframe(df_N)

# Subtotali numero di righe per AZ
num_SC = (df_N['AZ'] == 'SC').sum()
num_EM = (df_N['AZ'] == 'EM').sum()

st.write(f'##### Numero di GMP in backlog per Scavolini: :green[{num_SC}]')
st.write(f'##### Numero di GMP in backlog per Ernestomeda: :green[{num_EM}]')
st.write(f'##### Numero totale di GMP in backlog: :green[{num_SC + num_EM}]')

# Pulsante per scaricare xlsx
# Crea il bottone per scaricare il pivot ordinato
file_GMP_in_backlog = to_excel_bytes(df_N)
st.download_button(
    label="📥 Scarica file GMP in backlog",
    data=file_GMP_in_backlog,
    file_name='file_GMP_in_backlog.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
)




# Analisi backlog

# 1. Conta il numero di righe per ogni combinazione di Classificazione e AZ
df_count = df_N.groupby(['Classificazione', 'AZ']).size().reset_index(name='Conteggio')

# 2. Crea il grafico a barre
fig = px.bar(
    df_count,
    x='Classificazione',
    y='Conteggio',
    color='AZ',
    title='Numero di GMP per stratificato per Classificazione e colorato per Azienda',
    labels={'Classificazione': 'Classificazione', 'Conteggio': 'Numero GMP', 'AZ': 'Azienda'},
    height=700,
    color_discrete_map={
        'SC': 'red',
        'EM': 'gray'
    }
)

# 3. Calcola il totale per ogni Classificazione (sull'asse x)
totali = df_count.groupby('Classificazione')['Conteggio'].sum().reset_index()

# 4. Aggiungi etichette in cima alle barre impilate
fig.add_scatter(
    x=totali['Classificazione'],
    y=totali['Conteggio'],
    mode='text',
    text=[f"{valore:.0f}" for valore in totali['Conteggio']],
    textposition='top center',
    textfont=dict(size=22, color='black', family='Arial'),
    showlegend=False,
    hoverinfo='skip'
)

# 5. Aumenta dimensioni font di etichette, legenda, titolo, ecc.
fig.update_layout(
    xaxis=dict(
        title_font=dict(size=22),
        tickfont=dict(size=18)
    ),
    yaxis=dict(
        title_font=dict(size=22),
        tickfont=dict(size=18)
    ),
    title=dict(
        font=dict(size=30)
    ),
    legend=dict(
        font=dict(size=20)
    ),
    legend_title=dict(
        font=dict(size=24)
    )
)

# Calcola il massimo dei totali
max_tot = totali['Conteggio'].max()

fig.update_yaxes(range=[0, max_tot * 1.15])  # aumenta la scala del 15%

# 6. Visualizza in Streamlit
st.plotly_chart(fig, use_container_width=True)



# Crea la colonna per il tooltip personalizzato
df_N['Tooltip'] = df_N['G.M.P.'].astype(str) + " | " + df_N['Argomento'].astype(str)

# Grafico a barre impilate con colori personalizzati e altezza aumentata
fig = px.bar(
    df_N,
    x='Classificazione',
    y='Tempo',
    color='AZ',
    custom_data=['Tooltip'],
    title='Tempo assegnato [ore] stratificato per Classificazione eimpilato per Azienda',
    labels={'Tempo': 'Tempo', 'Classificazione': 'Classificazione', 'AZ': 'Azienda'},
    height=700,
    color_discrete_map={
        'SC': 'red',
        'EM': 'gray'
    }
)

# Tooltip personalizzato
fig.update_traces(
    hovertemplate='%{customdata[0]}<br>Tempo: %{y}<extra></extra>'
)

# Calcola il totale del tempo per ogni Classificazione
totali = df_N.groupby('Classificazione')['Tempo'].sum().reset_index()

# Aggiungi etichette in cima alle barre impilate (totale Tempo per ciascuna Classificazione)
fig.add_scatter(
    x=totali['Classificazione'],
    y=totali['Tempo'],
    mode='text',
    text=[f"{valore:.0f}" for valore in totali['Tempo']],
    textposition='top center',
    textfont=dict(size=22, color='black', family='Arial'),
    showlegend=False,
    hoverinfo='skip'
)

# Allarga la scala dell'asse y per non tagliare le etichette
max_tot = totali['Tempo'].max()
fig.update_yaxes(range=[0, max_tot * 1.15])

# Migliora i font di assi, titolo e legenda
fig.update_layout(
    xaxis=dict(
        title_font=dict(size=22),
        tickfont=dict(size=18)
    ),
    yaxis=dict(
        title_font=dict(size=22),
        tickfont=dict(size=18)
    ),
    title=dict(
        font=dict(size=30)
    ),
    legend=dict(
        font=dict(size=20)
    ),
    legend_title=dict(
        font=dict(size=24)
    )
)

# Visualizza in Streamlit
st.plotly_chart(fig, use_container_width=True)


tempo_totale_backlog = df_N['Tempo'].sum()
st.write(f'#### Tempo totale in backlog (in ore di lavoro):  :green[{tempo_totale_backlog}]')

tempo_di_coda = df_N['Tempo'].sum() / capacity_team
st.write(f'#### Tempo di coda del backlog (in giorni):  :green[{tempo_di_coda:.1f}]')

st.header('Summary tempi di coda e di processo [giorni lavorativi]', divider='red')
st.write(f'#### Tempi di coda (backlog): :green[{tempo_di_coda:.1f}]', )
st.write(f'#### Tempi di processo (in corso): :green[{(giorni_residui / num_operatori):.1f}]')
st.write(f'#### Tempi totali di processo (backlog + in corso): :green[{(giorni_residui / num_operatori) + tempo_di_coda:.1f}]')
