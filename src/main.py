import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import location


def load_gas_dt():
    dt_loaded = pd.read_csv(location.GAS, sep='\t')

    is_gas = dt_loaded['PRODUTO'] == "GASOLINA COMUM"
    dt_gas = dt_loaded[is_gas]

    # Retirando colunas desnecessárias.
    dt_gas = dt_gas.drop(['PREÇO MÉDIO DISTRIBUIÇÃO',
                          'DESVIO PADRÃO DISTRIBUIÇÃO',
                          'PREÇO MÍNIMO DISTRIBUIÇÃO',
                          'PREÇO MÁXIMO DISTRIBUIÇÃO',
                          'COEF DE VARIAÇÃO DISTRIBUIÇÃO',
                          'DATA FINAL',
                          'PRODUTO',
                          'ESTADO',
                          'DESVIO PADRÃO REVENDA',
                          'PREÇO MÍNIMO REVENDA',
                          'PREÇO MÁXIMO REVENDA',
                          'COEF DE VARIAÇÃO REVENDA'], axis=1)

    # Renomeando colunas para o padrão correto.
    dt_gas.rename(columns={
        'NÚMERO DE POSTOS PESQUISADOS': 'gas_stations_searched',
        'PREÇO MÉDIO REVENDA': 'avg_price',
        'DATA INICIAL': 'initial_date',
        'REGIÃO': 'region',
        'MES': 'month',
        'ANO': 'year'
    }, inplace=True)

    dt_gas['initial_date'] = pd.to_datetime(dt_gas['initial_date'], format='%Y-%m-%d')
    dt_gas['month'] = pd.DatetimeIndex(dt_gas['initial_date']).month
    dt_gas['year'] = pd.DatetimeIndex(dt_gas['initial_date']).year
    return dt_gas

def load_gas_df_IA():
    dt_loaded = pd.read_csv(location.GAS, sep='\t')

    is_gas = dt_loaded['PRODUTO'] == "GASOLINA COMUM"
    dt_gas = dt_loaded[is_gas]

    # Retirando colunas desnecessárias.
    dt_gas = dt_gas.drop(['PREÇO MÉDIO DISTRIBUIÇÃO',
                          'DESVIO PADRÃO DISTRIBUIÇÃO',
                          'UNIDADE DE MEDIDA',
                          'MARGEM MÉDIA REVENDA',
                          'COEF DE VARIAÇÃO DISTRIBUIÇÃO',
                          'DATA FINAL',
                          'PRODUTO',
                          'ESTADO',
                          'DESVIO PADRÃO REVENDA',
                          'COEF DE VARIAÇÃO REVENDA'], axis=1)

    # Renomeando colunas para o padrão correto.
    dt_gas.rename(columns={
        'NÚMERO DE POSTOS PESQUISADOS': 'gas_stations_searched',
        'PREÇO MÉDIO REVENDA': 'avg_price',
        'DATA INICIAL': 'initial_date',
        'REGIÃO': 'region',
        'MES': 'month',
        'ANO': 'year',
        'PREÇO MÍNIMO REVENDA': 'MINIMUM_RESALE_PRICE',
        'PREÇO MÁXIMO REVENDA': 'MAXIMUM_RESALE_PRICE',
        'PREÇO MÁXIMO DISTRIBUIÇÃO': 'MAXIMUM_PRICE_DISTRIBUTION',
        'PREÇO MÍNIMO DISTRIBUIÇÃO': 'MINIMUM_PRICE_DISTRIBUTION',

    }, inplace=True)

    dt_gas['initial_date'] = pd.to_datetime(dt_gas['initial_date'], format='%Y-%m-%d')
    dt_gas['month'] = pd.DatetimeIndex(dt_gas['initial_date']).month
    dt_gas['year'] = pd.DatetimeIndex(dt_gas['initial_date']).year
    return dt_gas

def load_petroleum_dt():
    dt_petro = pd.read_csv(location.PETROLEUM)
    dt_petro['Data'] = pd.to_datetime(dt_petro['Data'], format='%d.%m.%Y')
    dt_petro['month'] = pd.DatetimeIndex(dt_petro['Data']).month
    dt_petro['year'] = pd.DatetimeIndex(dt_petro['Data']).year

    dt_petro['Último'] = dt_petro['Último'].astype('str')
    dt_petro['Último'] = dt_petro['Último'].str.replace(',', '.')
    dt_petro['Último'] = pd.to_numeric(dt_petro['Último'], errors='coerce')

    dt_petro['Máxima'] = dt_petro['Máxima'].astype('str')
    dt_petro['Máxima'] = dt_petro['Máxima'].str.replace(',', '.')
    dt_petro['Máxima'] = pd.to_numeric(dt_petro['Máxima'], errors='coerce')

    dt_petro['Mínima'] = dt_petro['Mínima'].astype('str')
    dt_petro['Mínima'] = dt_petro['Mínima'].str.replace(',', '.')
    dt_petro['Mínima'] = pd.to_numeric(dt_petro['Mínima'], errors='coerce')

    return dt_petro


def format_number(data_value, index):
    formatter = ''
    if data_value < 1000000:
        formatter = '{:0.1f}k'.format(data_value / 1000000)
    if data_value >= 1000000:
        formatter = '{:0.2f}k'.format(data_value / 1000000)
    return formatter


def plot_total_gas_stations_searched(dt_gas):
    # TODO: Melhorar visualização do gráfico
    dt_gas_stations = dt_gas[['region', 'gas_stations_searched']].groupby(['region']).sum().sort_values(
        by='gas_stations_searched',
        ascending=False)

    dt_gas_stations.plot(legend=False, kind='bar', figsize=(16, 13)).yaxis.set_major_formatter(format_number)

    plt.tick_params(axis='both', labelsize=20)
    plt.title('Número total de postos de gasolina pesquisados', fontsize=30)
    plt.ylabel('Quantidade/mil', fontsize=20)
    plt.xlabel('Região', fontsize=20)
    plt.show()


def plot_first_insight(dt_gas, dt_petro):
    df_gas_d1 = dt_gas.drop(['region',
                             'gas_stations_searched',
                             'UNIDADE DE MEDIDA',
                             'MARGEM MÉDIA REVENDA'], axis=1)
    df_gas_d1['semestre'] = np.where(df_gas_d1['month'] < 7, 1, 2)

    df_gas_d1 = df_gas_d1.reset_index(drop=True)
    df_mask = df_gas_d1['year'] > 2010
    df_gas_d1 = df_gas_d1[df_mask]

    dt_petro['PrecoMedio'] = ((dt_petro['Máxima'] + dt_petro['Mínima']) ) / 40
    dfPetro_d1 = dt_petro.drop(['Último', 'Abertura', 'Máxima', 'Mínima', 'Vol.', 'Var%'], axis=1)
    dfPetro_d1['semestre'] = np.where(dfPetro_d1['month'] < 7, 1, 2)

    dfPetro_d1 = dfPetro_d1.drop('month', axis=1)

    dfPetro_d1 = dfPetro_d1.reset_index(drop=False)

    result = pd.merge(df_gas_d1, dfPetro_d1, how='outer', on=['year', 'semestre'])
    result.to_csv('result.csv')
    ax = result.filter(['year', 'semestre', 'PrecoMedio', 'avg_price']) \
        .groupby(['year', 'semestre']).mean() \
        .sort_values(['year', 'semestre'], ascending=True) \
        .plot(figsize=(11, 7), fontsize=15, grid=True, colormap='Dark2')

    ax.set_title("Relação entre o preço da gasolina e o do petróleo", fontsize=30)
    ax.set_ylabel("")

    ax.get_legend().remove()

    ax.xaxis.set_minor_locator(MultipleLocator())
    ax.xaxis.set_major_locator(MultipleLocator())

    ax.tick_params(which='both', width=2, color='b')
    ax.tick_params(which='major', length=4, color='g')
    ax.tick_params(which='minor', length=6, color='r')

    labels = ax.get_xticklabels()
    plt.setp(labels, rotation=45, horizontalalignment='right')
    plt.ylabel('Preço da Gasolina - R$/l', fontsize=20)
    plt.xlabel("Ano/Semestre", fontsize=20)

    plt.show()


def load_inflation_dt():
    dt_inflation = pd.read_csv(location.INFLATION_RATE)
    dt_inflation['date'] = pd.to_datetime(dt_inflation['date']).dt.to_period('M')
    inflation_rate_dt = dt_inflation.set_index('date')['2011':'2019']
    inflation_rate_dt.sample(n=5)
    return inflation_rate_dt


def filter_dt_gas_to_inflation(dt_gas):
    # TODO: Trocar linguagem das colunas para inglês

    used_columns = [
        'DATA',
        'PRODUTO',
        'ESTADO',
        'REGIÃO',
        'PREÇO MÉDIO DISTRIBUIÇÃO',
        'PREÇO MÉDIO REVENDA',
    ]

    monthly_df = dt_gas.groupby(['PRODUTO', 'ESTADO', 'REGIÃO']).resample('M').mean().reset_index()
    monthly_df['DATA'] = monthly_df['DATA FINAL'].dt.to_period('M')

    gas_prices_df = monthly_df[used_columns]
    gas_prices_df = gas_prices_df.dropna(how='any')

    gas_prices_df['VARIAÇÃO DIST X REV'] = gas_prices_df['PREÇO MÉDIO REVENDA'] - gas_prices_df[
        'PREÇO MÉDIO DISTRIBUIÇÃO']
    gas_prices_df['VARIAÇÃO PERCENTUAL DIST X REV'] = gas_prices_df['VARIAÇÃO DIST X REV'] / gas_prices_df[
        'PREÇO MÉDIO DISTRIBUIÇÃO'] * 100

    gas_prices_df.set_index(['DATA', 'PRODUTO', 'ESTADO', 'REGIÃO'], inplace=True)
    gas_prices_df.sample(n=5)
    return gas_prices_df


def prepare_gas_dt_to_inflation_compare(dt_gas):
    dt_loaded = pd.read_csv(location.GAS, sep='\t')

    is_gas = dt_loaded['PRODUTO'] == "GASOLINA COMUM"
    dt_gas = dt_loaded[is_gas]

    dt_gas['DATA FINAL'] = pd.to_datetime(dt_gas['DATA FINAL'])

    dt_gas['PREÇO MÉDIO DISTRIBUIÇÃO'] = pd.to_numeric(dt_gas['PREÇO MÉDIO DISTRIBUIÇÃO'], errors='coerce')
    dt_gas['PREÇO MÉDIO REVENDA'] = pd.to_numeric(dt_gas['PREÇO MÉDIO REVENDA'], errors='coerce')

    dt_gas.set_index(['DATA FINAL'], inplace=True)

    dt_gas = filter_dt_gas_to_inflation(dt_gas)
    return dt_gas


def prepare_inflation_rate_dt(dt_gas, dt_inflation_rate):
    inflation_rate_gas_price_dt = dt_gas.copy()
    inflation_rate_gas_price_dt.sample(5)
    inflation_rate_gas_price_dt.reset_index(inplace=True)
    inflation_rate_gas_price_dt.set_index('DATA', inplace=True)
    inflation_rate_gas_price_dt['INFLAÇÃO ANUAL'] = dt_inflation_rate['annual_accumulation']
    inflation_rate_gas_price_dt['INFLAÇÃO ABSOLUTA'] = dt_inflation_rate['absolute_index']

    inflation_rate_gas_price_dt = inflation_rate_gas_price_dt.to_timestamp()

    inflation_rate_gas_price_dt = inflation_rate_gas_price_dt[:'2019']
    inflation_rate_gas_price_dt.sample(5)
    return inflation_rate_gas_price_dt


def graph_inflation_rate_gas_price_compare(inflation_rate_gas_price_dt):
    # TODO: Limpar datas que não estão sendo utilizadas do dt

    annual_price_change_df = inflation_rate_gas_price_dt.groupby(['DATA']).mean()
    annual_price_change_df = annual_price_change_df['2010':]
    annual_price_change_df['VARIAÇÃO'] = annual_price_change_df['PREÇO MÉDIO REVENDA'].pct_change()
    annual_price_change_df['VARIAÇÃO 12 MESES'] = annual_price_change_df['VARIAÇÃO'].rolling(min_periods=12,
                                                                                             window=12).sum() * 100
    annual_price_change_df.tail()

    fig, ax_gas = plt.subplots()
    annual_price_change_df.plot(y='VARIAÇÃO 12 MESES', c='#4c72b0', ax=ax_gas)
    fig.suptitle('Variação da Inflação e Preço Acumulado - GASOLINA COMUM')
    ax_gas.set_xlabel('Data de pesquisa')
    ax_gas.set_ylabel('% - Variação Acumulativa - 12 meses', color='#4c72b0')

    ax_gas.get_legend().remove()
    ax_gas.grid(True)

    # plot_inflation
    ax_inflation = ax_gas.twinx()

    annual_price_change_df.plot(y='INFLAÇÃO ANUAL', ax=ax_inflation, c='#55a868')
    ax_inflation.set_ylabel('% - Inflação Acumulada - 12 meses', color='#55a868')
    ax_inflation.get_legend().remove()
    ax_inflation.grid(False)
    plt.show()


def plot_inflation_rate_over_gas_price(dt_gas, dt_inflation_rate):
    dt_gas_prepared = prepare_gas_dt_to_inflation_compare(dt_gas)
    inflation_rate_gas_price_dt = prepare_inflation_rate_dt(dt_gas_prepared, dt_inflation_rate)
    graph_inflation_rate_gas_price_compare(inflation_rate_gas_price_dt)
    return inflation_rate_gas_price_dt


def plot_avg_gas_price_region(inflation_rate_gas_price_dt):
    avg_per_region_for_product_df = inflation_rate_gas_price_dt['2011':].groupby(['PRODUTO', 'REGIÃO', 'DATA']).mean()

    fig, ax = plt.subplots()
    fig.suptitle('Preço por Região - GASOLINA COMUM')

    for key, grp in avg_per_region_for_product_df.groupby('REGIÃO'):
        grp_as_timeseries = grp.reset_index().set_index('DATA')
        grp_as_timeseries.plot(y='PREÇO MÉDIO REVENDA', label=key, ax=ax)

        ax.set_xlabel('Data')
        ax.set_ylabel('Preço - R$/l')

    plt.grid(True)
    plt.show()


def config_visual_plot():
    plt.rcParams['figure.figsize'] = (11, 7)
    plt.style.use('seaborn')


if __name__ == '__main__':
    config_visual_plot()
    dt_gas = load_gas_dt()
    dt_petro = load_petroleum_dt()
    dt_inflation_rate = load_inflation_dt()

    # # Geração de gráficos
    # plot_total_gas_stations_searched(dt_gas)
    # # TODO: Remover esse retorno e corrigir o dataset de gas para ser usado para as duas situações
    # inflation_rate_gas_price_dt = plot_inflation_rate_over_gas_price(dt_gas, dt_inflation_rate)
    # plot_avg_gas_price_region(inflation_rate_gas_price_dt)
    plot_first_insight(dt_gas, dt_petro)
