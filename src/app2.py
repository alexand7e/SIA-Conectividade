import os
import json
import geopandas as gpd
import pandas as pd
import plotly.express as px
import streamlit as st
import folium
from streamlit_folium import st_folium
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
import graphviz
import statsmodels.api as sm

def train_and_display_decision_tree(df, target_column, feature_columns, model_type='Regressão'):
    # Dividir os dados em conjunto de treinamento e teste
    X = df[feature_columns]
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Treinar o modelo de árvore de decisão
    if model_type == 'Regressão':
        model = DecisionTreeRegressor(random_state=42)
    else:
        model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Fazer previsões no conjunto de teste
    y_pred = model.predict(X_test)

    # Exibir resultados
    if model_type == 'Regressão':
        mae = mean_absolute_error(y_test, y_pred)
        st.write(f"Erro Médio Absoluto (MAE) no conjunto de teste: {mae:.2f}")
    else:
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Acurácia no conjunto de teste: {accuracy:.2f}")

    # Importância das variáveis
    importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    st.write("### Importância das Variáveis")
    st.write(importance_df)

    # Função para criar e renderizar a árvore
    def render_tree(model, max_depth=None):
        dot_data = export_graphviz(
            model,
            feature_names=feature_columns,
            filled=True,
            rounded=True,
            special_characters=True,
            max_depth=max_depth
        )
        graph = graphviz.Source(dot_data)
        # Converter para formato que o Streamlit pode renderizar
        st.graphviz_chart(dot_data)

    # Visualizar a árvore de decisão interativamente
    st.write("### Visualização Interativa da Árvore de Decisão")
    render_tree(model)

    # Adicionar controles para ajustar a profundidade da árvore
    st.write("### Ajustar Profundidade da Árvore")
    max_depth = st.slider("Profundidade Máxima", min_value=1, max_value=10, value=3)
    
    # Treinar novo modelo com a profundidade ajustada
    if model_type == 'Regressão':
        adjusted_model = DecisionTreeRegressor(random_state=42, max_depth=max_depth)
    else:
        adjusted_model = DecisionTreeClassifier(random_state=42, max_depth=max_depth)
    adjusted_model.fit(X_train, y_train)
    
    # Visualizar a árvore ajustada
    render_tree(adjusted_model, max_depth)

def load_shapefiles():
    if 'MUNICIPIOS' not in st.session_state:
        municipios_json = os.path.join(os.path.dirname(__file__), "..", "data", "mapas", "municipios.json")
        estados_json = os.path.join(os.path.dirname(__file__), "..", "data", "mapas", "estados.json")
        
        with open(municipios_json, 'r', encoding='utf-8-sig') as f:
            municipios = pd.DataFrame(json.load(f))
        
        with open(estados_json, 'r', encoding='utf-8-sig') as f:
            estados = pd.DataFrame(json.load(f))
        
        municipios = gpd.GeoDataFrame(
            municipios, 
            geometry=gpd.points_from_xy(municipios.longitude, municipios.latitude),
            crs="EPSG:4326"
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

def read_uploaded_files(uploaded_files):
    dataframes = {}
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith('.xlsx'):
            df_name = uploaded_file.name.replace('.xlsx', '').replace(' ', '_').lower()
            dataframes[df_name] = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.csv'):
            df_name = uploaded_file.name.replace('.csv', '').replace(' ', '_').lower()
            dataframes[df_name] = pd.read_csv(uploaded_file)
    return dataframes

def plot_interactive_map(gdf, metric_column, chave, additional_columns=[], label=None):
    if metric_column not in gdf.columns:
        st.error(f"A coluna {metric_column} não está presente no dataframe.")
        return

    if label is None:
        label = metric_column

    if chave == "Código IBGE":
        geo_data = gpd.read_file(os.path.join(os.path.dirname(__file__), "..", "data", "mapas", "geojs-100-mun.json"))
        geo_data = geo_data.rename(columns={"id": "Código IBGE"})
        gdf = gdf.merge(geo_data[['Código IBGE', 'geometry']], on='Código IBGE', how='left')
    elif chave == "UF":
        geo_data = gpd.read_file(os.path.join(os.path.dirname(__file__), "..", "data", "mapas", "geojs-100-uf.json"))
        geo_data = geo_data.rename(columns={"id": "UF"})
        gdf = gdf.merge(geo_data[['UF', 'geometry']], on='UF', how='left')
        
    gdf = gpd.GeoDataFrame(gdf, geometry='geometry')

    m = folium.Map(location=[-15.788497, -47.879873], zoom_start=4)

    if chave == "Código IBGE":
        key_on = "feature.properties.Código IBGE"
    else:
        key_on = f"feature.properties.{chave}"

    choropleth = folium.Choropleth(
        geo_data=gdf,
        name="choropleth",
        data=gdf,
        columns=[chave, metric_column],
        key_on=key_on,
        fill_color="YlOrRd",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=label,
    ).add_to(m)

    fields = [chave, metric_column] + additional_columns
    aliases = [chave.capitalize(), metric_column] + [col.capitalize() for col in additional_columns]

    choropleth.geojson.add_child(
        folium.features.GeoJsonTooltip(
            fields=fields,
            aliases=aliases,
            style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;")
        )
    )

    folium.LayerControl().add_to(m)
    st.write(f"Número de regiões no mapa: {len(gdf)}")
    st.write(f"Colunas disponíveis: {gdf.columns.tolist()}")
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
        X = df[x_column]
        Y = df[y_column]
        
        X = sm.add_constant(X)
        
        model = sm.OLS(Y, X).fit()
        
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

    load_shapefiles()
    MUNICIPIOS = st.session_state['MUNICIPIOS']
    ESTADOS = st.session_state['ESTADOS']

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

    uploaded_files = st.sidebar.file_uploader("Carregar arquivos Excel ou CSV", accept_multiple_files=True)

    if uploaded_files:
        uploaded_dataframes = read_uploaded_files(uploaded_files)
        st.session_state['dataframes'].update(uploaded_dataframes)

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

    st.sidebar.header("Árvore de Decisão")
    if st.sidebar.checkbox("Aplicar Árvore de Decisão"):
        model_type = st.sidebar.radio("Escolha o tipo de modelo:", ("Regressão", "Classificação"))
        
        target_column = st.sidebar.selectbox("Escolha a variável dependente (target)", df.columns)
        
        feature_columns = st.sidebar.multiselect("Escolha as variáveis independentes (features)", df.columns)
        
        if feature_columns and target_column:
            st.write("### Resultados da Árvore de Decisão")
            train_and_display_decision_tree(df, target_column, feature_columns, model_type)
            
    if chart_type == "Mapa":
        type_map = st.sidebar.radio("Tipo de mapa:", ("Estadual", "Municipal"))
        
        aggregation_type = st.sidebar.selectbox(
            "Escolha o tipo de agregação:",
            ("Soma", "Média")
        )
        
        label = st.text_input("Digite o rótulo a ser exibido no mapa:", "Métrica")

        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
        map_column = st.selectbox("Escolha uma métrica para o mapa", numeric_columns)
        category_columns = df.select_dtypes(include=['object', 'category']).columns
        selected_additional_columns = st.multiselect("Escolha categorias adicionais para exibir no mapa", category_columns)
        
        chave = ''
        
        if type_map == "Estadual":
            chave = "UF"
        elif type_map == "Municipal":
            chave = "Código IBGE"
        
        if aggregation_type == "Soma":
            df_grouped = df.groupby([chave, *selected_additional_columns])[map_column].sum().reset_index()
        elif aggregation_type == "Média":
            df_grouped = df.groupby([chave, *selected_additional_columns])[map_column].mean().reset_index()
        
        if type_map == "Estadual":
            st.session_state['df_estadual_merged'] = df
        elif type_map == "Municipal":
            st.session_state['df_municipal_merged'] = df

        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
        if numeric_columns.empty:
            st.error("Não há colunas numéricas disponíveis para o mapa.")
        else:
            plot_interactive_map(df, map_column, chave, additional_columns=selected_additional_columns, label=label)

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
