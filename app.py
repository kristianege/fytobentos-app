import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from pyproj import Transformer
from scipy.spatial.distance import braycurtis
from scipy.spatial.distance import braycurtis, jaccard
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS # Vi bruger MDS motoren til at lave PCoA
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. KONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Fytobentos Kort & Statistik", layout="wide")

st.title("Fytobentos Data")

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

tab1, tab2, tab3 = st.tabs(["Kort & Oversigt", "Tidsudvikling/Forskelle (Station)", "Forskelle"])

# =============================================================================
# FANE 1: KORT & OVERSIGT
# =============================================================================
with tab1:
    # --- FILTRERING ---
    with st.expander("Filtreringsmuligheder", expanded=True):
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
            zoom=5, 
            
            # ÆNDRING 1: Vi skifter til 'carto-positron' (virker bedre interactively)
            mapbox_style="carto-positron", 
            
            height=500
        )
        
        # ÆNDRING 2: Vi tilføjer config={'scrollZoom': True}
        # Dette tvinger kortet til at acceptere zoom med musehjulet/fingre
        st.plotly_chart(
            fig_map, 
            use_container_width=True, 
            config={'scrollZoom': True, 'displayModeBar': True}
        )

        st.divider()

        # --- SEKTION: HYPPIGHED OG UDBREDELSE ---
        st.subheader("Arts-hyppighed og Udbredelse")
        st.markdown("Venstre: Hvor mange *prøver* er arten fundet i? Højre: Hvor mange unikke *stationer* er arten fundet på?")

        if not df_map.empty:
            # Opret to kolonner til graferne
            col_freq1, col_freq2 = st.columns(2)

            # --- GRAF 1: ANTAL REGISTRERINGER (Venstre) ---
            with col_freq1:
                # Tæl rækker pr art
                obs_counts = df_map['Art latin'].value_counts().reset_index()
                obs_counts.columns = ['Art latin', 'Antal']
                obs_counts = obs_counts.sort_values('Antal', ascending=True)
                
                hojde_obs = max(400, len(obs_counts) * 25)

                with st.container(height=500):
                    fig_obs = px.bar(
                        obs_counts,
                        x='Antal',
                        y='Art latin',
                        orientation='h',
                        text_auto=True,
                        title="Antal fund totalt (alle prøver)",
                        labels={'Antal': 'Antal prøver', 'Art latin': 'Art'},
                        height=hojde_obs
                    )
                    fig_obs.update_layout(yaxis_title="", margin=dict(l=10, r=10, t=30, b=10))
                    st.plotly_chart(fig_obs, use_container_width=True)

            # --- GRAF 2: ANTAL STATIONER (Højre) ---
            with col_freq2:
                # Tæl unikke StedID pr art
                station_counts = df_map.groupby('Art latin')['StedID'].nunique().reset_index()
                station_counts.columns = ['Art latin', 'Antal']
                station_counts = station_counts.sort_values('Antal', ascending=True)
                
                hojde_stat = max(400, len(station_counts) * 25)

                with st.container(height=500):
                    fig_stat = px.bar(
                        station_counts,
                        x='Antal',
                        y='Art latin',
                        orientation='h',
                        text_auto=True,
                        title="Antal stationer (Udbredelse)",
                        labels={'Antal': 'Antal stationer', 'Art latin': 'Art'},
                        height=hojde_stat
                    )
                    # Vi beholder y-aksen (artsnavne) her også, da sorteringen kan være forskellig
                    # (En art kan være fundet 100 gange på 1 station vs 10 gange på 10 stationer)
                    fig_stat.update_layout(yaxis_title="", margin=dict(l=10, r=10, t=30, b=10))
                    st.plotly_chart(fig_stat, use_container_width=True)

        else:
            st.info("Ingen data at vise.")
            
        st.divider()
        
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
# FANE 3: MILJØ, SUBSTRAT & GEOGRAFI ANALYSE
# =============================================================================
with tab3:
    st.subheader("Sammenlignende Analyse")
    st.markdown("Undersøg forskelle i biodiversitet og artssammensætning.")

    # --- 1. INDSTILLINGER (Gruppering & Metode) ---
    c_opt1, c_opt2 = st.columns(2)
    
    with c_opt1:
        analysis_mode = st.radio(
            "1. Hvad vil du sammenligne?", 
            ["Substrater (Sten, Tagrør...)", "Vandtype (Sø vs. Vandløb)", "Geografi (Landsdel)"],
            horizontal=True
        )
        
    with c_opt2:
        metric_mode = st.radio(
            "2. Hvilken metode?",
            ["Bray-Curtis (Mængder)", "Jaccard (Artsliste)"],
            horizontal=True,
            help="Bray-Curtis vægter mængden af hver art. Jaccard kigger kun på, om arterne er til stede (God til geografi)."
        )
    
    st.divider()

    # --- FILTRE & DATAUDTRÆK ---
    c_sub1, c_sub2 = st.columns(2)
    with c_sub1:
        if "Vandtype" in analysis_mode:
            st.markdown(f"**Vælg Medie:** (Låst for sammenligning)")
            sub_kilder = data['Kilde'].unique()
        else:
            sub_kilder = st.multiselect("Vælg Medie", options=data['Kilde'].unique(), default=data['Kilde'].unique(), key="sub_kilde")
            
    with c_sub2:
        sub_min = int(data['År'].min())
        sub_max = int(data['År'].max())
        if sub_min == sub_max:
             sub_year_range = (sub_min, sub_max)
        else:
             sub_year_range = st.slider("Vælg Periode", sub_min, sub_max, (sub_min, sub_max), key="sub_aar")
    
    # Filtrer grunddata
    df_sub = data.copy()
    df_sub = df_sub[df_sub['Kilde'].isin(sub_kilder)]
    df_sub = df_sub[(df_sub['År'] >= sub_year_range[0]) & (df_sub['År'] <= sub_year_range[1])]

    # --- KONFIGURER LOGIK ---
    if "Substrater" in analysis_mode:
        group_col = 'Indsamlet prøve fra'
        group_label = 'Substrat'
        df_sub = df_sub[df_sub['Indsamlet prøve fra'] != 'Ukendt']

    elif "Vandtype" in analysis_mode:
        group_col = 'Kilde'
        group_label = 'Vandtype'

    elif "Geografi" in analysis_mode:
        group_col = 'Region'
        group_label = 'Landsdel'
        
        def assign_region(row):
            lon = row['lon']
            if lon < 9.9: return "Jylland"
            elif 9.9 <= lon < 10.9: return "Fyn"
            else: return "Sjælland/Øer"

        if not df_sub.empty:
            df_sub['Region'] = df_sub.apply(assign_region, axis=1)

    if df_sub.empty:
        st.warning("Ingen data fundet med de valgte filtre.")
    else:
        
        # --- ANALYSE 1: DIVERSITET (BOXPLOT) ---
        st.subheader(f"1. Diversitet (Artsrigdom pr. {group_label})")
        
        richness = df_sub.groupby(['StedID', 'Dato', group_col])['Art latin'].nunique().reset_index()
        richness.columns = ['StedID', 'Dato', 'Gruppe', 'Antal_Arter']
        richness = richness.sort_values('Gruppe')

        fig_div = px.box(
            richness, x='Gruppe', y='Antal_Arter', color='Gruppe', points='outliers', 
            title=f"Artsrigdom: {group_label}",
            labels={'Antal_Arter': 'Antal Arter pr. prøve', 'Gruppe': group_label},
            height=500
        )
        st.plotly_chart(fig_div, use_container_width=True)

        st.divider()

        # --- ANALYSE 2: ORDINATION (PCoA) ---
        # Bestem titel og forklaring baseret på valg
        if "Jaccard" in metric_mode:
            pcoa_title = f"PCoA (Jaccard - Artsliste)"
            pcoa_desc = "Analysen er baseret på **Jaccard**. Den ser kun på, om arter er til stede (1) eller ej (0)."
            calc_metric = 'jaccard'
        else:
            pcoa_title = f"PCoA (Bray-Curtis - Mængder)"
            pcoa_desc = "Analysen er baseret på **Bray-Curtis**. Den vægter arter med høj forekomst tungere."
            calc_metric = 'braycurtis'

        st.subheader(f"2. Artssammensætning ({pcoa_title})")
        st.markdown(pcoa_desc)
        
        # Dataklargøring
        df_sub['SampleID'] = df_sub['StedID'].astype(str) + "_" + df_sub['Dato'].astype(str) + "_" + df_sub[group_col].astype(str)
        pivot_data = df_sub.pivot_table(index='SampleID', columns='Art latin', values='Relativ_Forekomst', fill_value=0)
        
        if len(pivot_data) < 3:
            st.warning("For få prøver til at lave PCoA (kræver mindst 3).")
        else:
            try:
                # --- FORBERED DATA TIL JACCARD VS BRAY-CURTIS ---
                if "Jaccard" in metric_mode:
                    # Til Jaccard konverterer vi til binær (0 eller 1)
                    # Alt over 0 bliver til 1
                    data_for_dist = (pivot_data > 0).astype(int)
                else:
                    # Til Bray-Curtis bruger vi de relative mængder
                    data_for_dist = pivot_data

                # Beregn Afstand
                dist_matrix = pdist(data_for_dist, metric=calc_metric)
                dist_square = squareform(dist_matrix)
                
                # Kør PCoA (MDS)
                mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, n_init=4, max_iter=300)
                coords = mds.fit_transform(dist_square)
                
                pcoa_df = pd.DataFrame(data=coords, columns=['PCoA1', 'PCoA2'], index=pivot_data.index)
                
                meta_data = df_sub.groupby('SampleID')[group_col].first()
                pcoa_df['Gruppe'] = meta_data.reindex(pcoa_df.index)
                
                centroids = pcoa_df.groupby('Gruppe')[['PCoA1', 'PCoA2']].mean().reset_index()

                # Basis Plot
                fig_pcoa = px.scatter(
                    pcoa_df, x='PCoA1', y='PCoA2', color='Gruppe',
                    title=pcoa_title, hover_name=pcoa_df.index, opacity=0.5, height=650,
                    labels={'Gruppe': group_label}
                )

                # Tegn Centroids
                import plotly.graph_objects as go 
                for i, row in centroids.iterrows():
                    fig_pcoa.add_trace(go.Scatter(
                        x=[row['PCoA1']], y=[row['PCoA2']], mode='markers+text',
                        marker=dict(size=20, symbol='cross', line=dict(width=2, color='black')),
                        name=f"GNS: {row['Gruppe']}", text=[f"<b>{row['Gruppe']}</b>"], textposition="top center"
                    ))

                # Beregn Pile (Korrelation)
                # Vi beregner altid korrelation mod de oprindelige mængder (pivot_data) 
                # for at se hvilke arter der er hyppige, selvom vi kører Jaccard.
                np.seterr(divide='ignore', invalid='ignore')
                correlations = []
                for art in pivot_data.columns:
                    abundance = pivot_data[art]
                    if abundance.std() > 0:
                        corr_x = np.corrcoef(abundance, pcoa_df['PCoA1'])[0, 1]
                        corr_y = np.corrcoef(abundance, pcoa_df['PCoA2'])[0, 1]
                        if not np.isnan(corr_x) and not np.isnan(corr_y):
                            length = (corr_x**2 + corr_y**2)**0.5
                            correlations.append({'Art': art, 'x': corr_x, 'y': corr_y, 'Length': length})
                np.seterr(all='warn')

                if correlations:
                    corr_df = pd.DataFrame(correlations)
                    top_vectors = corr_df.nlargest(8, 'Length')
                    
                    max_x = max(pcoa_df['PCoA1'].abs().max(), 0.1)
                    max_y = max(pcoa_df['PCoA2'].abs().max(), 0.1)
                    scale_factor_x = max_x * 1.2
                    scale_factor_y = max_y * 1.2

                    for index, row in top_vectors.iterrows():
                        x_vec = row['x'] * scale_factor_x
                        y_vec = row['y'] * scale_factor_y
                        fig_pcoa.add_shape(type='line', x0=0, y0=0, x1=x_vec, y1=y_vec, line=dict(color="red", width=2))
                        fig_pcoa.add_annotation(x=x_vec, y=y_vec, text=row['Art'], showarrow=False, xanchor="center", yanchor="bottom", font=dict(color="darkred", size=14, family="Arial Black"))

                st.plotly_chart(fig_pcoa, use_container_width=True)
            except Exception as e:
                st.error(f"Fejl i PCoA: {e}")

        st.divider()

        # --- ANALYSE 3: HEATMAP ---
        st.subheader(f"3. Arts-præference ({group_label})")
        
        top_n_pref = 30
        top_species_total = df_sub.groupby('Art latin')['Relativ_Forekomst'].sum().nlargest(top_n_pref).index.tolist()
        df_pref = df_sub[df_sub['Art latin'].isin(top_species_total)].copy()

        avg_abundance = df_pref.groupby(['Art latin', group_col])['Relativ_Forekomst'].mean().unstack(fill_value=0)
        row_normalized = avg_abundance.div(avg_abundance.sum(axis=1), axis=0) * 100
        
        if not row_normalized.empty:
            first_col = row_normalized.columns[0]
            row_normalized = row_normalized.sort_values(first_col, ascending=False)
            fig_heat = px.imshow(
                row_normalized, labels=dict(x=group_label, y="Art", color="Affinitet (%)"),
                x=row_normalized.columns, y=row_normalized.index,
                color_continuous_scale="RdBu_r", aspect="auto", height=800
            )
            st.plotly_chart(fig_heat, use_container_width=True)
