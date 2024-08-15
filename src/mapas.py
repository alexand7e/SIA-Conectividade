import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from os import path

data_path = path.join(path.dirname(__file__), "..", "data", "mapas")

def criar_mapa_territorios(data: pd.DataFrame, variaveis: list[str]):
    """
    Cria um mapa dos territórios usando um shapefile e dados de um arquivo Excel.

    :param shapefile_territorios: Caminho para o shapefile dos territórios
    :param dados_excel: Caminho para o arquivo Excel com os dados
    :param variaveis: Lista de variáveis a serem plotadas
    """
    # Carregar o Shapefile dos Territórios
    gdf_territorios = gpd.read_file(path.join(data_path, "Territórios.shp"))
    gdf_territorios = gdf_territorios[['Nome_Micro', 'geometry']]
    dados = pd.read_excel(data)

    gdf_territorios_dados = gdf_territorios.merge(dados, on='Nome_Micro')

    num_variaveis = len(variaveis)
    ncols = 2
    nrows = (num_variaveis + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 8 * nrows))

    for i, variavel in enumerate(variaveis):
        ax = axes[i // ncols, i % ncols] if nrows > 1 else axes[i % ncols]  # Seleciona o eixo correto
        gdf_territorios_dados.plot(ax=ax, column=variavel, cmap='Greens', legend=True, edgecolor='black')

        for idx, row in gdf_territorios_dados.iterrows():
            x, y = row.geometry.centroid.x, row.geometry.centroid.y
            label = f"{int(row[variavel] * 100):2,}"
            bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="b", lw=0.5)
            ax.annotate(label, xy=(x, y), xytext=(0, 0), textcoords="offset points",
                        ha='center', va='center', fontsize=6, bbox=bbox_props)

        ax.set_title('Distribuição de ' + variavel)
        ax.set_axis_off()  # Remove os eixos para um visual mais limpo

    plt.tight_layout()
    plt.show()


def criar_mapa_municipios_pi(dados_excel, variaveis, output_dir="."):
    """
    Cria mapas dos municípios usando um shapefile e dados de um arquivo Excel,
    e salva cada mapa como uma imagem PNG.

    :param dados_excel: Caminho para o arquivo Excel com os dados
    :param variaveis: Lista de variáveis a serem plotadas
    :param output_dir: Diretório onde os arquivos PNG serão salvos
    """
    # Carregar o Shapefile dos Municípios
    gdf_municipios = gpd.read_file(path.join(data_path, "PI_Municipios_2022.shp"))

    # Carregar os dados do Excel
    dados = pd.read_excel(dados_excel)
    gdf_municipios['CD_MUN'] = gdf_municipios['CD_MUN'].astype(str)
    dados['CD_MUN'] = dados['CD_MUN'].astype(str)

    gdf_municipios_dados = gdf_municipios.merge(dados, on='CD_MUN')

    for variavel in variaveis:
        fig, ax = plt.subplots(figsize=(8, 8))  # Cria um novo gráfico para cada variável

        gdf_municipios_dados.plot(ax=ax, column=variavel, cmap='Greens', legend=True, edgecolor='black')

        # Adiciona título e formatação ao mapa
        ax.set_title('Distribuição de ' + variavel)
        ax.set_axis_off()  # Remove os eixos para um visual mais limpo

        output_path = f"{output_dir}/distribuicao_municipio_{variavel}.png"
        plt.savefig(output_path, bbox_inches='tight')

        plt.close(fig)  # Fecha a figura para liberar memória

    print("Gráficos exportados com sucesso!")


def criar_mapa_municipios_br(dados_excel, variaveis, output_dir="."):
    """
    Cria mapas dos municípios usando um shapefile e dados de um arquivo Excel,
    e salva cada mapa como uma imagem PN

    :param dados_excel: Caminho para o arquivo Excel com os dados
    :param variaveis: Lista de variáveis a serem plotadas
    :param output_dir: Diretório onde os arquivos PNG serão salvos
    """
    # Carregar o Shapefile dos Municípios
    gdf_municipios = gpd.read_file(path.join(data_path, "BR_Municipios_2022.shp"))

    # Carregar os dados do Excel
    dados = pd.read_excel(dados_excel)
    gdf_municipios['CD_MUN'] = gdf_municipios['CD_MUN'].astype(str)
    dados['CD_MUN'] = dados['CD_MUN'].astype(str)

    gdf_municipios_dados = gdf_municipios.merge(dados, on='CD_MUN')

    for variavel in variaveis:
        fig, ax = plt.subplots(figsize=(8, 8))  # Cria um novo gráfico para cada variável

        gdf_municipios_dados.plot(ax=ax, column=variavel, cmap='Greens', legend=True, edgecolor='black')

        # Adiciona título e formatação ao mapa
        ax.set_title('Distribuição de ' + variavel)
        ax.set_axis_off()  # Remove os eixos para um visual mais limpo

        output_path = f"{output_dir}/distribuicao_municipio_{variavel}.png"
        plt.savefig(output_path, bbox_inches='tight')

        plt.close(fig)  # Fecha a figura para liberar memória

    print("Gráficos exportados com sucesso!")

