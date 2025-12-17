import streamlit as st
import pandas as pd
import plotly.express as px
from pyproj import Transformer
from scipy.spatial.distance import braycurtis
from scipy.spatial.distance import braycurtis, jaccard

# -----------------------------------------------------------------------------
# 1. KONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Fytobentos Kort & Statistik", layout="wide")

st.title("🌿 Fytobentos Data Explorer")

# -----------------------------------------------------------------------------
# 2. DATA LOADER FUNKTION
# -----------------------------------------------------------------------------
@st.cache_data
def load_and_prep_data():
    # BEMÆRK: Vi leder nu efter .parquet filer
    file_so = 'Fytobenthos - Sø.parquet'
    file_vandlob = 'Fytobenthos - Vandløb.parquet'
    
    def read_file(fname, kilde):
        try:
            # Parquet gemmer datatyper, så vi behøver ikke angive separator eller encoding længere
            d = pd.read_parquet(fname)
            d['Kilde'] = kilde
            return d
        except FileNotFoundError:
            return None

    df_so = read_file(file_so, 'Sø')
    df_vandlob = read_file(file_vandlob, 'Vandløb')
    
    if df_so is None and df_vandlob is None:
        return None

    df = pd.concat([df_so, df_vandlob], ignore_index=True)
    
    # --- DATAVASK (Samme som før) ---
    # Selvom Parquet husker typer, er det sikrest at køre vasken igen, 
    # da vi indlæste alt som strings (dtype=str) under konverteringen for at være sikre.
    
    df['Dato'] = pd.to_datetime(df['Dato'], format='%d-%m-%Y', errors='coerce')
    df = df.dropna(subset=['Dato'])
    df['År'] = df['Dato'].dt.year.astype(int)

    # Sikrer at kolonner er strings før replace (hvis parquet har gemt dem som andet)
    df['x-koordinat'] = pd.to_numeric(df['x-koordinat'].astype(str).str.replace(',', '.'), errors='coerce')
    df['y-koordinat'] = pd.to_numeric(df['y-koordinat'].astype(str).str.replace(',', '.'), errors='coerce')
    df = df.dropna(subset=['x-koordinat', 'y-koordinat'])
    
    df['Tælletal'] = pd.to_numeric(df['Tælletal'].astype(str).str.replace(',', '.'), errors='coerce').fillna(0)

    df['Stedtekst'] = df['Stedtekst'].fillna('')
    df['Art latin'] = df['Art latin'].fillna('Ukendt')
    df['Indsamlet prøve fra'] = df['Indsamlet prøve fra'].fillna('Ukendt')
    
    df['Lokation_Visning'] = df['StedID'].astype(str) + ": " + df['Stedtekst']

    # --- BEREGNING AF RELATIV FOREKOMST ---
    df['Total_I_Prøve'] = df.groupby(['StedID', 'Dato'])['Tælletal'].transform('sum')
    
    df['Relativ_Forekomst'] = 0.0
    mask = df['Total_I_Prøve'] > 0
    df.loc[mask, 'Relativ_Forekomst'] = (df.loc[mask, 'Tælletal'] / df.loc[mask, 'Total_I_Prøve']) * 100

    # --- KOORDINAT TRANSFORMATION ---
    if not df.empty:
        transformer = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(df['x-koordinat'].values, df['y-koordinat'].values)
        df['lon'] = lon
        df['lat'] = lat
    
    return df
data = load_and_prep_data()

if data is None or data.empty:
    st.error("Kunne ikke finde datafiler.")
    st.stop()

# -----------------------------------------------------------------------------
# 3. FANER (TABS)
# -----------------------------------------------------------------------------

tab1, tab2, tab3 = st.tabs(["🗺️ Kort & Oversigt", "📈 Tidsudvikling (Station)", "🪨 Substrat Analyse"])

# =============================================================================
# FANE 1: KORT & OVERSIGT
# =============================================================================
with tab1:
    # --- FILTRERING ---
    with st.expander("🔎 Filtreringsmuligheder", expanded=True):
        # Række 1: De overordnede filtre
        c1, c2 = st.columns([1, 2])
        with c1:
            kilder = st.multiselect("Vælg Medie", options=data['Kilde'].unique(), default=data['Kilde'].unique())
        with c2:
            min_year = int(data['År'].min())
            max_year = int(data['År'].max())
            if min_year == max_year:
                year_range = (min_year, max_year)
                st.info(f"Data indeholder kun år {min_year}")
            else:
                year_range = st.slider("Vælg Periode (År)", min_year, max_year, (min_year, max_year))

        # Række 2: De detaljerede filtre
        c3, c4, c5 = st.columns(3)
        with c3:
            prv_options = sorted(data['Indsamlet prøve fra'].unique())
            selected_prv = st.multiselect("Prøvetype", options=prv_options)
        with c4:
            art_options = sorted(data['Art latin'].unique())
            selected_art = st.multiselect("Art (Latin)", options=art_options)
        with c5:
            loc_options = sorted(data['Lokation_Visning'].unique())
            selected_loc = st.multiselect("Lokation", options=loc_options)

    # --- DATABEHANDLING FOR FANE 1 ---
    df_map = data.copy()
    
    # Anvend filtre
    df_map = df_map[df_map['Kilde'].isin(kilder)]
    df_map = df_map[(df_map['År'] >= year_range[0]) & (df_map['År'] <= year_range[1])]

    if selected_prv:
        df_map = df_map[df_map['Indsamlet prøve fra'].isin(selected_prv)]
    if selected_art:
        df_map = df_map[df_map['Art latin'].isin(selected_art)]
    if selected_loc:
        df_map = df_map[df_map['Lokation_Visning'].isin(selected_loc)]

    st.divider()

    # --- VISUALISERING ---
    m1, m2, m3 = st.columns(3)
    m1.metric("Antal observationer", len(df_map))
    m2.metric("Unikke Arter", df_map['Art latin'].nunique())
    m3.metric("Unikke Lokationer", df_map['StedID'].nunique())

    if not df_map.empty:
        # Farve logik: Hvis få arter er valgt -> farv efter art. Ellers efter Medie.
        color_col = "Art latin" if (len(selected_art) > 0 and len(selected_art) < 15) else "Kilde"

        # Kort
        fig_map = px.scatter_mapbox(
            df_map,
            lat="lat", lon="lon",
            color=color_col,
            size="Relativ_Forekomst",
            size_max=25,
            hover_name="Lokation_Visning",
            hover_data={"Art latin": True, "Relativ_Forekomst": ":.2f", "År": True},
            zoom=5, mapbox_style="open-street-map", height=500
        )
        st.plotly_chart(fig_map, use_container_width=True)

        # Bar Chart
        st.subheader("Artsfordeling (Median for valgte data)")
        
# Box Plot
        st.subheader("Artsfordeling (Boxplot)")
        st.markdown("Viser fordelingen af relativ forekomst. Boksen viser de midterste 50% af data (interkvartil), stregen er medianen. De mest almindelige arter ligger øverst.")
        
        # 1. BEREGN SORTERING (Median)
        # Vi grupperer på art, finder medianen, og sorterer (ascending=True).
        # ascending=True betyder: Laveste tal først i listen -> Tegnes nederst i grafen.
        # Højeste tal sidst i listen -> Tegnes øverst i grafen.
        if not df_map.empty:
            median_series = df_map.groupby('Art latin')['Relativ_Forekomst'].median()
            median_order = median_series.sort_values(ascending=False).index.tolist()
            
            # Beregn dynamisk højde
            antal_arter = df_map['Art latin'].nunique()
            dynamisk_hojde = max(400, antal_arter * 25)

            with st.container(height=600):
                fig_box = px.box(
                    df_map,
                    x="Relativ_Forekomst",
                    y="Art latin",
                    orientation='h',
                    
                    # Her anvender vi sorteringen
                    category_orders={"Art latin": median_order},
                    
                    height=dynamisk_hojde,
                    labels={'Relativ_Forekomst': 'Relativ Forekomst (%)', 'Art latin': 'Art'},
                    hover_data=["StedID", "År", "Dato"],
                    points="outliers"
                )
                
                fig_box.update_layout(
                    xaxis_title="Relativ Forekomst (%)", 
                    yaxis_title="",
                    margin=dict(l=10, r=10, t=10, b=10)
                )
                
                st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("Ingen data til graf med de valgte filtre.")
    else:
        st.warning("Ingen data matcher dine valg.")

# =============================================================================
# FANE 2: TIDSUDVIKLING (STATION)
# =============================================================================
# =============================================================================
# FANE 2: TIDSUDVIKLING (STATION)
# =============================================================================
with tab2:
    st.subheader("Sammenligning af prøver over tid")
    st.markdown("Denne fane analyserer udviklingen på en valgt station.")
    
    # 1. FIND STATIONER
    station_counts = data.groupby('Lokation_Visning')['Dato'].nunique()
    valid_stations = station_counts[station_counts > 1].index.tolist()
    valid_stations.sort()
    
    if not valid_stations:
        st.error("Der findes ingen stationer i det samlede datasæt med mere end 1 prøve.")
    else:
        # Vælger
        selected_station_single = st.selectbox(
            f"Søg og vælg en station ({len(valid_stations)} mulige)", 
            options=valid_stations,
            placeholder="Skriv for at søge..."
        )
        
        # Hent data
        station_data = data[data['Lokation_Visning'] == selected_station_single].copy()
        
        # --- GRAF 1: STACKED BAR CHART ---
        top_n = 12
        species_sum = station_data.groupby('Art latin')['Relativ_Forekomst'].sum()
        top_species = species_sum.nlargest(top_n).index.tolist()
        
        station_data['Art_Plot'] = station_data['Art latin'].apply(lambda x: x if x in top_species else 'Andre')
        station_data['Dato_Str'] = station_data['Dato'].dt.strftime('%d-%m-%Y')
        
        plot_df = station_data.groupby(['Dato', 'Dato_Str', 'Art_Plot'])['Relativ_Forekomst'].sum().reset_index()
        plot_df = plot_df.sort_values(['Dato', 'Relativ_Forekomst'], ascending=[True, False])

        fig_time = px.bar(
            plot_df,
            x="Dato_Str", 
            y="Relativ_Forekomst",
            color="Art_Plot",
            title=f"Artssammensætning (Mængder)",
            labels={"Relativ_Forekomst": "Relativ Forekomst (%)", "Dato_Str": "Dato", "Art_Plot": "Art"},
            color_discrete_sequence=px.colors.qualitative.Prism,
            height=400
        )
        fig_time.update_layout(xaxis_type='category')
        fig_time.update_xaxes(tickangle=-45)
        st.plotly_chart(fig_time, use_container_width=True)
        
        st.divider()
        
        # --- FORBEREDELSE TIL MATRICER ---
        # 1. Pivotér data (Mængder til Bray-Curtis)
        pivot_qty = station_data.pivot_table(
            index='Dato_Str', columns='Art latin', values='Relativ_Forekomst', fill_value=0
        )
        # Sorter kronologisk
        sorted_dates = station_data[['Dato', 'Dato_Str']].drop_duplicates().sort_values('Dato')['Dato_Str'].tolist()
        pivot_qty = pivot_qty.reindex(sorted_dates)
        
        # 2. Lav Binær data (Tilstede/Ikke-tilstede til Jaccard)
        # Hvis mængden er > 0, sætter vi den til 1 (True), ellers 0 (False)
        pivot_bool = (pivot_qty > 0).astype(int)

        n_samples = len(pivot_qty)
        
        # Opret to kolonner til matricerne
        col_m1, col_m2 = st.columns(2)
        
        # --- MATRIX A: BRAY-CURTIS (Biologisk Lighed) ---
        with col_m1:
            st.subheader("1. Biologisk Lighed")
            st.markdown("Bray-Curtis: Tager højde for **mængden** af hver art.")
            
            bc_matrix = pd.DataFrame(index=pivot_qty.index, columns=pivot_qty.index, dtype=float)
            for i in range(n_samples):
                for j in range(n_samples):
                    dissim = braycurtis(pivot_qty.iloc[i], pivot_qty.iloc[j])
                    bc_matrix.iloc[i, j] = (1 - dissim) * 100

            fig_bc = px.imshow(
                bc_matrix,
                text_auto='.0f',
                color_continuous_scale='Blues',
                origin='lower',
                labels=dict(x="", y="", color="Lighed (%)")
            )
            fig_bc.update_layout(height=400, xaxis_title="", yaxis_title="")
            st.plotly_chart(fig_bc, use_container_width=True)

        # --- MATRIX B: JACCARD (Arts-overlap) ---
        with col_m2:
            st.subheader("2. Arts-overlap")
            st.markdown("Jaccard: Hvor stor % af **arterne** går igen? (Uanset mængde).")
            
            jac_matrix = pd.DataFrame(index=pivot_bool.index, columns=pivot_bool.index, dtype=float)
            for i in range(n_samples):
                for j in range(n_samples):
                    # Jaccard beregner ulighed på boolske vektorer (0/1)
                    # dissim = 1 betyder ingen overlap. dissim = 0 betyder fuldt overlap.
                    dissim = jaccard(pivot_bool.iloc[i], pivot_bool.iloc[j])
                    jac_matrix.iloc[i, j] = (1 - dissim) * 100

            fig_jac = px.imshow(
                jac_matrix,
                text_auto='.0f',
                color_continuous_scale='Greens', # Bruger grøn skala for at kende forskel
                origin='lower',
                labels=dict(x="", y="", color="Overlap (%)")
            )
            fig_jac.update_layout(height=400, xaxis_title="", yaxis_title="")
            st.plotly_chart(fig_jac, use_container_width=True)

        with st.expander("Se rådata for stationen"):
             st.dataframe(station_data)

# =============================================================================
# FANE 3: SUBSTRAT ANALYSE
# =============================================================================
with tab3:
    st.subheader("Sammenligning af Substrater (Prøvetype)")
    st.markdown("Her undersøges forskelle mellem substraterne (f.eks. Sten, Tagrør, Andet).")

    # --- LOKALE FILTRE FOR DENNE FANE ---
    # Vi giver mulighed for at filtrere på Medie og År, da substrater kan variere mellem sø/vandløb
    c_sub1, c_sub2 = st.columns(2)
    with c_sub1:
        # Standardvalg: Begge medier
        sub_kilder = st.multiselect("Vælg Medie (Substrat-analyse)", options=data['Kilde'].unique(), default=data['Kilde'].unique())
    with c_sub2:
        # Standardvalg: Alle år
        sub_min_year = int(data['År'].min())
        sub_max_year = int(data['År'].max())
        sub_year_range = st.slider("Vælg Periode (Substrat-analyse)", sub_min_year, sub_max_year, (sub_min_year, sub_max_year))
    
    # Opret datasæt til analysen
    df_sub = data.copy()
    df_sub = df_sub[df_sub['Kilde'].isin(sub_kilder)]
    df_sub = df_sub[(df_sub['År'] >= sub_year_range[0]) & (df_sub['År'] <= sub_year_range[1])]
    
    # Fjern prøver hvor substrat er ukendt
    df_sub = df_sub[df_sub['Indsamlet prøve fra'] != 'Ukendt']

    if df_sub.empty:
        st.warning("Ingen data fundet med de valgte filtre.")
    else:
        st.divider()
        
        # --- ANALYSE 1: DIVERSITET (Artsrigdom) ---
        st.markdown("### 1. Diversitet: Hvor mange arter findes pr. prøve?")
        st.markdown("Grafen viser spredningen i antallet af arter fundet på hvert substrat.")
        
        # Beregn antal arter pr. prøve (En prøve er unik pr. StedID + Dato + Substrat)
        # Vi tæller unikke 'Art latin'
        richness = df_sub.groupby(['StedID', 'Dato', 'Indsamlet prøve fra'])['Art latin'].nunique().reset_index()
        richness.columns = ['StedID', 'Dato', 'Substrat', 'Antal_Arter']
        
        # Sorter rækkefølgen (valgfrit)
        richness = richness.sort_values('Substrat')

        fig_div = px.box(
            richness,
            x='Substrat',
            y='Antal_Arter',
            color='Substrat',
            points='outliers', # Viser outliers som prikker. Brug 'all' for at se alle punkter.
            title="Artsrigdom pr. Substrat",
            labels={'Antal_Arter': 'Antal Arter pr. prøve', 'Substrat': 'Substrat type'},
            height=500
        )
        st.plotly_chart(fig_div, use_container_width=True)
        
        # --- ANALYSE 2: ARTS-PRÆFERENCE ---
        st.divider()
        st.markdown("### 2. Arts-præference: Er der forskel på artssammensætningen?")
        st.markdown("Grafen viser den gennemsnitlige relative forekomst for de mest almindelige arter, opdelt på substrat.")
        
        # 1. Find de N mest almindelige arter i det filtrerede datasæt (for at undgå rod)
        top_n_sub = 15
        top_species_list = df_sub.groupby('Art latin')['Relativ_Forekomst'].sum().nlargest(top_n_sub).index.tolist()
        
        # 2. Filtrer data til kun at indeholde disse arter
        df_top = df_sub[df_sub['Art latin'].isin(top_species_list)].copy()
        
        # 3. Beregn gennemsnitlig (eller median) relativ forekomst pr. substrat for disse arter
        # Vi grupperer på Substrat og Art
        preference = df_top.groupby(['Indsamlet prøve fra', 'Art latin'])['Relativ_Forekomst'].mean().reset_index()
        
        # 4. Lav Grouped Bar Chart
        fig_pref = px.bar(
            preference,
            x='Art latin',
            y='Relativ_Forekomst',
            color='Indsamlet prøve fra',
            barmode='group', # VIGTIGT: Dette sætter søjlerne ved siden af hinanden
            title=f"Top {top_n_sub} arters fordeling på substrater (Gennemsnit)",
            labels={'Relativ_Forekomst': 'Gns. Relativ Forekomst (%)', 'Indsamlet prøve fra': 'Substrat'},
            height=600
        )
        fig_pref.update_layout(xaxis_tickangle=-45)
        
        st.plotly_chart(fig_pref, use_container_width=True)