import os
import io

import json
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
import statsmodels.api as sm
import streamlit as st
import folium
from streamlit_folium import st_folium


# Função para carregar os arquivos de shapefile uma vez
def load_shapefiles():
    if 'MUNICIPIOS' not in st.session_state:
        # Caminho para os arquivos JSON
        municipios_json = os.path.join(os.path.dirname(__file__), "..", "data", "Mapas", "municipios.json")
        estados_json = os.path.join(os.path.dirname(__file__), "..", "data", "Mapas", "estados.json")
        
        # Carregar os JSONs como DataFrames, especificando a codificação utf-8-sig
        with open(municipios_json, 'r', encoding='utf-8-sig') as f:
            municipios = pd.DataFrame(json.load(f))
        
        with open(estados_json, 'r', encoding='utf-8-sig') as f:
            estados = pd.DataFrame(json.load(f))
        
        # Transformar em GeoDataFrame, utilizando as colunas de latitude e longitude
        municipios = gpd.GeoDataFrame(
            municipios, 
            geometry=gpd.points_from_xy(municipios.longitude, municipios.latitude),
            crs="EPSG:4326"  # Definindo o CRS como WGS 84 (usado em sistemas de GPS)
        ).rename(columns={"codigo_ibge": "Código IBGE"})

        estados = gpd.GeoDataFrame(
            estados, 
            geometry=gpd.points_from_xy(estados.longitude, estados.latitude),
            crs="EPSG:4326"
        ).rename(columns={"uf": "UF"})

        st.session_state['MUNICIPIOS'] = municipios
        st.session_state['ESTADOS'] = estados

def read_excel_files(path):
    dataframes = {}
    if not os.path.exists(path):
        st.error(f"O diretório {path} não existe.")
        return dataframes
    
    files = os.listdir(path)
    for file in files:
        if file.endswith('.xlsx'):
            file_path = os.path.join(path, file)
            df_name = file.replace('.xlsx', '').replace(' ', '_').lower()
            try:
                dataframes[df_name] = pd.read_excel(file_path)
            except Exception as e:
                st.error(f"Erro ao ler {file}: {str(e)}")
    return dataframes


def plot_interactive_map(gdf, metric_column, chave, additional_columns=[]):
    # Verifique se a coluna métrica existe no dataframe
    if metric_column not in gdf.columns:
        st.error(f"A coluna {metric_column} não está presente no dataframe.")
        return

    # Use o formato correto de key_on com base no tipo de mapa
    if chave == "UF":
        key_on = "feature.properties.UF"
    else:
        key_on = f"feature.properties.{chave}"

    # Crie o mapa base
    m = folium.Map(location=[-15.788497, -47.879873], zoom_start=4)

    # Adicione o choropleth ao mapa
    choropleth = folium.Choropleth(
        geo_data=gdf,
        name="choropleth",
        data=gdf,
        columns=[chave, metric_column],
        key_on=key_on,
        fill_color="YlOrRd",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=metric_column,
    ).add_to(m)

    # Adicione tooltips para mostrar os valores ao passar o mouse
    fields = [chave, metric_column] + additional_columns
    aliases = [chave.capitalize(), metric_column] + [col.capitalize() for col in additional_columns]

    choropleth.geojson.add_child(
        folium.features.GeoJsonTooltip(
            fields=fields,
            aliases=aliases,
            style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;")
        )
    )

    # Adicione um controle de camadas
    folium.LayerControl().add_to(m)

    # Exiba algumas informações de depuração
    st.write(f"Número de regiões no mapa: {len(gdf)}")
    st.write(f"Colunas disponíveis: {gdf.columns.tolist()}")
    st.write(f"Amostra de dados:")
    st.write(gdf[fields].head())

    # Exiba o mapa
    st_folium(m, width=700)




def plot_bar_chart_vertical(df, x_column, y_column, max_categories, sort_order):
    if sort_order == "Crescente":
        category_values = df.groupby(x_column)[y_column].sum().nsmallest(max_categories)
    else:
        category_values = df.groupby(x_column)[y_column].sum().nlargest(max_categories)

    fig = px.bar(category_values, x=category_values.index, y=category_values.values,
                 labels={x_column: x_column, y_column: y_column},
                 title=f'Gráfico de Barras Verticais de {x_column}')

    fig.update_traces(texttemplate='%{y:.0f}', textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', xaxis_title=x_column, yaxis_title=y_column)

    st.plotly_chart(fig)


def plot_bar_chart_horizontal(df, x_column, y_column, max_categories, sort_order):
    if sort_order == "Crescente":
        category_values = df.groupby(x_column)[y_column].sum().nsmallest(max_categories)
    else:
        category_values = df.groupby(x_column)[y_column].sum().nlargest(max_categories)

    fig = px.bar(category_values, x=category_values.values, y=category_values.index, orientation='h',
                 labels={x_column: x_column, y_column: y_column},
                 title=f'Gráfico de Barras Horizontais de {x_column}')

    fig.update_traces(texttemplate='%{x:.0f}', textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', xaxis_title=y_column, yaxis_title=x_column)

    st.plotly_chart(fig)



def plot_scatter(df, x_column, y_column, add_category_labels=False, add_trendline=False, add_value_labels=False, show_ols_stats=False):
    fig = px.scatter(df, x=x_column, y=y_column, labels={x_column: x_column, y_column: y_column},
                     title=f'Gráfico de Dispersão: {x_column} vs {y_column}')
    
    if add_trendline:
        fig = px.scatter(df, x=x_column, y=y_column, trendline="ols", labels={x_column: x_column, y_column: y_column},
                         title=f'Gráfico de Dispersão: {x_column} vs {y_column} com Tendência')
    
    if add_value_labels or add_category_labels:
        for i in range(df.shape[0]):
            fig.add_annotation(x=df[x_column].iloc[i], y=df[y_column].iloc[i],
                               text=(f'({df[x_column].iloc[i]:.2f}, {df[y_column].iloc[i]:.2f})' if add_value_labels else '') +
                                    (df.index[i] if add_category_labels else ''),
                               showarrow=True)

    st.plotly_chart(fig)

    if show_ols_stats:
        # Preparar os dados para a regressão
        X = df[x_column]
        Y = df[y_column]
        
        # Adicionar constante (intercepto) ao modelo
        X = sm.add_constant(X)
        
        # Ajustar o modelo OLS
        model = sm.OLS(Y, X).fit()
        
        # Mostrar as estatísticas do modelo
        st.write("### Estatísticas do Modelo MQO")
        st.write(model.summary())


def display_paginated_table(df, page_size=10):
    total_rows = df.shape[0]
    total_pages = (total_rows - 1) // page_size + 1

    page_number = st.number_input('Página', min_value=1, max_value=total_pages, value=1)
    start_row = (page_number - 1) * page_size
    end_row = start_row + page_size

    st.write(f"Exibindo linhas {start_row + 1} a {min(end_row, total_rows)} de {total_rows}")
    height = min(400, total_rows * 35)
    st.dataframe(df.iloc[start_row:end_row], height=height)
    
def icon(emoji: str):
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )

st.set_page_config(layout="wide")
def main():
    icon("📅")
    
    st.title('Análise de Dados')

    # Carregar os shapefiles apenas uma vez
    load_shapefiles()
    MUNICIPIOS = st.session_state['MUNICIPIOS']
    ESTADOS = st.session_state['ESTADOS']

    # Opção para o usuário escolher a pasta de dados
    st.sidebar.header("Configuração de Pasta")
    pasta_opcao = st.sidebar.radio(
        "Escolha a pasta para importar os dados:",
        ("Conectividade", "ADAPI")
    )

    if pasta_opcao == "Conectividade":
        path = os.path.join(os.path.dirname(__file__), "..", "data", "conectividade")
    else:
        path = os.path.join(os.path.dirname(__file__), "..", "data", "adapi")

    if 'dataframes' not in st.session_state:
        st.session_state['dataframes'] = read_excel_files(path)

    if not st.session_state['dataframes']:
        st.error("Nenhum dataframe foi carregado. Verifique o caminho dos arquivos.")
        return

    dataframes = st.session_state['dataframes']

    st.sidebar.header("Configuração")
    selected_df_name = st.sidebar.selectbox("Escolha o dataframe", list(dataframes.keys()))
    df = dataframes[selected_df_name]

    st.sidebar.header("Filtros")
    for column in df.select_dtypes(include=['object', 'category']).columns:
        unique_values = df[column].unique()
        selected_values = st.sidebar.multiselect(f"Filtrar por {column}", unique_values)
        if selected_values:
            df = df[df[column].isin(selected_values)]

    if "Código IBGE" in df.columns:
        df["Código IBGE"] = df["Código IBGE"].astype(str)
    if "Código IBGE" in MUNICIPIOS.columns:
        MUNICIPIOS["Código IBGE"] = MUNICIPIOS["Código IBGE"].astype(str)

    st.write(f"## Dataframe: {selected_df_name}")
    display_paginated_table(df)

    st.write("### Estatísticas Descritivas")
    st.write(df.describe())

    st.write("### Visualizações")
    st.sidebar.header("Opções de Gráfico")
    chart_type = st.sidebar.radio("Escolha o tipo de gráfico:", ("Barras", "Dispersão", "Mapa"))

    if chart_type == "Mapa":
        type_map = st.sidebar.radio("Tipo de mapa:", ("Estadual", "Municipal"))
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
        map_column = st.selectbox("Escolha uma métrica para o mapa", numeric_columns)
        category_columns = df.select_dtypes(include=['object', 'category']).columns
        selected_additional_columns = st.multiselect("Escolha categorias adicionais para exibir no mapa", category_columns)
        chave = ''
        
        if type_map == "Estadual":
            chave = "UF"
            df_grouped = df.groupby(chave)[map_column].sum().reset_index()
            df = df_grouped.merge(ESTADOS, on=chave)
            df = gpd.GeoDataFrame(df, geometry='geometry')
            st.session_state['df_estadual_merged'] = df

        elif type_map == "Municipal":
            chave = "Código IBGE"
            df_grouped = df.groupby(chave)[map_column].sum().reset_index()
            df = df_grouped.merge(MUNICIPIOS, on=chave)
            df = gpd.GeoDataFrame(df, geometry='geometry')
            st.session_state['df_municipal_merged'] = df

        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
        if numeric_columns.empty:
            st.error("Não há colunas numéricas disponíveis para o mapa.")
        else:
            plot_interactive_map(df, map_column, chave, additional_columns=selected_additional_columns)


    elif chart_type == "Barras":
        bar_orientation = st.sidebar.radio("Orientação do gráfico de barras:", ("Vertical", "Horizontal"))
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
        if len(categorical_columns) > 0 and len(numeric_columns) > 0:
            x_column = st.selectbox("Escolha a coluna categórica (eixo X)", categorical_columns)
            y_column = st.selectbox("Escolha a coluna numérica (eixo Y)", numeric_columns)
            max_categories = st.number_input("Quantas categorias exibir?", min_value=1, max_value=50, value=20)
            sort_order = st.radio("Ordenação das categorias:", ("Decrescente", "Crescente"))

            if bar_orientation == "Vertical":
                plot_bar_chart_vertical(df, x_column, y_column, max_categories, sort_order)
            else:
                plot_bar_chart_horizontal(df, x_column, y_column, max_categories, sort_order)
    
    elif chart_type == "Dispersão":
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_columns) >= 2:
            x_column = st.selectbox("Escolha a coluna para o eixo X", numeric_columns)
            y_column = st.selectbox("Escolha a coluna para o eixo Y", numeric_columns)
            
            # Opções para o gráfico de dispersão
            add_category_labels = st.sidebar.checkbox("Adicionar rótulos das categorias", value=False)
            add_trendline = st.sidebar.checkbox("Adicionar linha de tendência", value=False)
            add_value_labels = st.sidebar.checkbox("Adicionar rótulos de valor", value=False)
            show_ols_stats = st.sidebar.checkbox("Mostrar Estatísticas do MQO", value=True)

            plot_scatter(df, x_column, y_column, add_category_labels, add_trendline, add_value_labels, show_ols_stats)

        
    st.markdown("---")
    st.markdown(
        '<h6>Made in &nbsp<img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="Streamlit logo" height="16">&nbsp by <a href="https://instagram.com/alexand7e_">@Alexand7e</a></h6>',
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
