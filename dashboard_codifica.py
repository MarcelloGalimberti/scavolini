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

st.write('Note:')
st.write('1. Dati relativi a GMP entrate dal 2023 in poi')
st.write('2. GMP Elettrodomestici escluse')
st.write('3. La % di completamento è presa dal Planner')


###### Caricamento del file Excel
st.header('Caricamento database codifica', divider='red')
uploaded_raw = st.file_uploader(
    "Carica file Impegni ufficio")
if not uploaded_raw:
    st.stop()

###### Caricamento del file Planner
st.header('Caricamento Planner', divider='red')
uploaded_planner = st.file_uploader(
    "Carica file Planner")
if not uploaded_planner:
    st.stop()


###### Caricamento file stato GMP
st.header('Caricamento estrazione portale GMP', divider='red')
uploaded_stato_gmp = st.file_uploader(
    "Carica file Stato GMP")
if not uploaded_stato_gmp:
    st.stop()

# st.write('### Anteprima del file Stato GMP caricato')
df_GMP = pd.read_excel(uploaded_stato_gmp, parse_dates=True)

#st.dataframe(df_GMP)

def converti_colonne_data(df, colonne):
    for col in colonne:
        df[col] = pd.to_datetime(df[col], errors='coerce').dt.normalize()
    return df

colonne_date = ['Data_Chiusura_Richiesta', 'Data_Inizio_Lavorazione', 'Data_Chiusura',
                'Scadenza_GMP']

df_GMP = converti_colonne_data(df_GMP, colonne_date)


# Filtro 
# 1. Elimina le righe dove Numero_GMP inizia con "PMP"
mask_numero = ~df_GMP["Numero_GMP"].astype(str).str.startswith("PMP")

# 2. Elimina le righe dove Stato_GMP è "Annullata"
mask_stato = df_GMP["Stato_GMP"] != "Annullata"

# 3. Elimina le righe dove Tipo_GMP è "Elettrodomestico"
mask_tipo = df_GMP["Tipo_GMP"] != "Elettrodomestico"

# Applica tutte le condizioni insieme
df_GMP = df_GMP[mask_numero & mask_stato & mask_tipo].reset_index(drop=True)

# st.write('### Anteprima del file Stato GMP caricato dopo il filtro')
# st.dataframe(df_GMP)

df_filtrato = df_GMP[df_GMP["Data_Chiusura"].notnull()]

pivot = df_filtrato.pivot_table(
    index="Numero_GMP",
    values="Data_Chiusura",
    aggfunc="max"
).reset_index()

pivot = pivot.sort_values(by="Data_Chiusura", ascending=True).reset_index(drop=True)


# Assumiamo che il dataframe 'pivot' abbia le colonne: Numero_GMP, Data_Chiusura (in formato datetime o date)
# Se non sono datetime, converti:
pivot["Data_Chiusura"] = pd.to_datetime(pivot["Data_Chiusura"], errors="coerce")

# 1. Crea la colonna Mese-Anno
pivot["Mese"] = pivot["Data_Chiusura"].dt.to_period("M").dt.to_timestamp()

# 2. Calcola il numero di Numero_GMP per mese
counts = pivot.groupby("Mese")["Numero_GMP"].count().reset_index(name="Numero GMP")

# 3. Calcola le medie mobili
counts["Media mobile 3 mesi"] = counts["Numero GMP"].rolling(window=3, min_periods=1).mean()
counts["Media mobile 9 mesi"] = counts["Numero GMP"].rolling(window=9, min_periods=1).mean()

# 4. Crea il diagramma Plotly
fig_MA = go.Figure()

fig_MA.add_trace(go.Scatter(
    x=counts["Mese"], y=counts["Numero GMP"], 
    mode='lines+markers', 
    name='Numero GMP'
))

fig_MA.add_trace(go.Scatter(
    x=counts["Mese"], y=counts["Media mobile 3 mesi"], 
    mode='lines',
    name='Media mobile 3 mesi'
))

fig_MA.add_trace(go.Scatter(
    x=counts["Mese"], y=counts["Media mobile 9 mesi"], 
    mode='lines',
    name='Media mobile 9 mesi'
))

fig_MA.update_layout(
    title="Numero di Numero_GMP chiusi per mese e media mobile",
    xaxis_title="Mese",
    yaxis_title="Numero GMP",
    legend_title="Legenda"
)

# Layout con font grandi
fig_MA.update_layout(
    title="Numero_GMP chiusi per mese e media mobile",
    title_font=dict(size=26),
    xaxis_title="Mese",
    yaxis_title="Numero GMP",
    xaxis=dict(title_font=dict(size=20), tickfont=dict(size=16)),
    yaxis=dict(title_font=dict(size=20), tickfont=dict(size=16)),
    legend=dict(font=dict(size=18), title_font=dict(size=20)),
    margin=dict(t=60, b=40, l=10, r=10),
    height=700,
)




df_planner = pd.read_excel(uploaded_planner, skiprows=8, header=0, parse_dates=True)

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

# Processing di df_raw per abbinare l'avanzamento da Planner
def get_percent_complete(gmp, df_planner):
    gmp_str = str(int(gmp))
    filtro = df_planner["Nome"].astype(str).str.contains(gmp_str, na=False)
    if filtro.any():
        return df_planner.loc[filtro, "% di completamento"].iloc[0]
    else:
        return 0

df_raw["Test"] = df_raw["G.M.P."].apply(lambda x: get_percent_complete(x, df_planner))


# Aggiorna "Avanzamento" con il massimo tra "Avanzamento" e "Test"
df_raw["Avanzamento"] = df_raw[["Avanzamento", "Test"]].max(axis=1)

# Elimina la colonna "Test"
df_raw = df_raw.drop(columns=["Test"])

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


col_1, col_2 = st.columns([1, 1])

with col_1:
    st.write('#### Ore giornaliere per operatore')
    st.dataframe(df_ore_GMP)
    capacity_team = df_ore_GMP['ore/giorno'].sum()
    

with col_2:
    st.write('#### Radar chart % GMP per operatore',)
    st.plotly_chart(fig, use_container_width=True)


st.write(f'#### Capacità totale del team in ore/giorno: :green[{capacity_team:.1f}]')

st.header('Database GMP assegnate e workload residuo (in giorni)', divider='red')


# Funzione per scaricare xlsx
def to_excel_bytes(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Foglio1') #index=False,
    return output.getvalue()

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

tempo_residuo = (df_A["Tempo"] / df_A['numero_op_assegnati'] * (1-df_A["Avanzamento"])).sum()
tempo_complessivo = df_A["Tempo"].sum()
completamento_percentuale = tempo_residuo / tempo_complessivo * 100
numero_GMP_assegnate = df_A.shape[0]
GMP_residue_equivalenti = numero_GMP_assegnate*(1-completamento_percentuale/100)

st.write(f'#### Numero di GMP assegnate: :green[{numero_GMP_assegnate}]')
st.write(f'#### Tempo complessivo assegnato [ore]: :green[{tempo_complessivo:.1f}]')
st.write(f'#### Completamento percentuale: :green[{completamento_percentuale:.1f}%]')
st.write(f'#### Tempo residuo totale [ore]: :green[{tempo_residuo:.1f}]')
#st.write(f'#### GMP residue equivalenti (in base al completamento): :green[{GMP_residue_equivalenti:.1f}]')

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

num_operatori = (df_ore_GMP['ore/giorno'] > 1).sum() # modificare qui per ridurre il numero di operatori
#st.write(f'#### Operatori con ore/giorno dedicate a GMP > 1: :green[{num_operatori:.0f}]')

st.write(f'#### Tempo di attraversamento medio processo codifica [giorni]:  :green[{(tempo_residuo / capacity_team):.1f}]') 


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
    title='Tempo assegnato [ore] stratificato per Classificazione e impilato per Azienda',
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
st.write(f'#### Tempo totale in backlog [ore]:  :green[{tempo_totale_backlog}]')

tempo_di_coda = df_N['Tempo'].sum() / capacity_team
st.write(f'#### Tempo di coda del backlog [giorni] = ore di backlog / capacità del Team:  :green[{tempo_di_coda:.1f}]')

st.header('Tempi di coda e di processo in base a workload e assegnazioni', divider='red')
st.write(f'#### Tempi di coda (backlog) [giorni]: :green[{tempo_di_coda:.1f}]', )
st.write(f'#### Tempi di processo (in corso) [giorni]: :green[{(tempo_residuo / capacity_team):.1f}]')
st.write(f'#### Tempi totali di processo (backlog + in corso) [giorni]: :green[{(tempo_residuo / capacity_team) + tempo_di_coda:.1f}]')
st.write(f'#### Tempi totali di processo (backlog + in corso) [mesi]: :green[{((tempo_residuo / capacity_team) + tempo_di_coda)/21:.1f}]')

st.header('Andamento GMP chiuse per mese', divider='red')
# 5. Visualizza in Streamlit
st.plotly_chart(fig_MA, use_container_width=True)

ma_3 = counts["Media mobile 3 mesi"].dropna().iloc[-1]
st.write(f'#### Media mobile 3 mesi [GMP chiuse / mese]: :green[{ma_3:.1f}]')
st.write(f'#### GMP residue equivalenti: :green[{GMP_residue_equivalenti:.1f}]')
st.write(f'#### GMP in backlog: :green[{num_SC + num_EM}]')
GMP_totali = GMP_residue_equivalenti + num_SC + num_EM
st.write(f'#### Totale GMP in processo: :green[{GMP_totali:.1f}]')
tempo_di_attraversamento_Little = (GMP_residue_equivalenti+num_SC + num_EM)/ma_3
st.write(f'#### Tempo di attraversamento medio = Totale GMP / Media mobile 3 mesi : :green[{tempo_di_attraversamento_Little:.1f}] [mesi]')

