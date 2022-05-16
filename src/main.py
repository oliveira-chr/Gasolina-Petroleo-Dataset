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
        formatter = '{:0.1f}k'.format(data_value/1000000)
    if data_value >= 1000000:
        formatter = '{:0.2f}k'.format(data_value/1000000)
    return formatter


def graph_total_gas_stations_searched(dt_gas):
    dt_gas_stations = dt_gas[['region', 'gas_stations_searched']].groupby(['region']).sum().sort_values(
        by='gas_stations_searched',
        ascending=False)

    dt_gas_stations.plot(kind='bar', figsize=(11, 7)).yaxis.set_major_formatter(format_number)
    plt.show()


def first_insight(dt_gas, dt_petro):
    df_gas_d1 = dt_gas.drop(['region', 'gas_stations_searched', 'UNIDADE DE MEDIDA', 'MARGEM MÉDIA REVENDA'], axis=1)
    df_gas_d1['semestre'] = np.where(df_gas_d1['month'] < 7, 1, 2)
    # df_gas_d1 = df_gas_d1.groupby(by=['year', 'semestre']).mean()
    # df_gas_d1 = df_gas_d1.drop('month', axis=1)
    # df_gas_d1 = df_gas_d1.sort_values(by=['year', 'semestre'], ascending=True)
    df_gas_d1 = df_gas_d1.reset_index(drop=True)
    df_mask = df_gas_d1['year'] > 2010
    df_gas_d1 = df_gas_d1[df_mask]
    df_gas_d1.plot()
    plt.show()

    dt_petro['PrecoMedio'] = ((dt_petro['Máxima'] + dt_petro['Mínima']) / 2) / 158.98722
    dfPetro_d1 = dt_petro.drop(['Último', 'Abertura', 'Máxima', 'Mínima', 'Vol.', 'Var%'], axis=1)
    dfPetro_d1['semestre'] = np.where(dfPetro_d1['month'] < 7, 1, 2)
    # dfPetro_d1.insert(2, "avg_price", df_gas_d1['avg_price'], allow_duplicates=False)
    dfPetro_d1 = dfPetro_d1.drop('month', axis=1)

    # dfPetro_d1 = dfPetro_d1.groupby(by=['year', 'semestre']).mean()

    # dfPetro_d1 = dfPetro_d1.sort_values(['year', 'semestre'], ascending=False)
    dfPetro_d1 = dfPetro_d1.reset_index(drop=False)
    # dfPetro_d1.filter(['ANO','semestre','PrecoMedio'])\
    #     .groupby(['ANO', 'semestre']).mean()\
    #     .sort_values(['ANO', 'semestre'], ascending=True)\
    #     .plot()

    # frames = [df_gas_d1, dfPetro_d1]
    # result = pd.concat([df_gas_d1, dfPetro_d1], axis=1, join='inner')
    result = pd.merge(df_gas_d1, dfPetro_d1, how='outer', on=['year', 'semestre'])
    # result = result.sort_values(['year', 'semestre'], ascending=False)
    # fig, ax = plt.subplots(figsize=(20,10))
    # plt.suptitle("testename")
    ax = result.filter(['year', 'semestre', 'PrecoMedio', 'avg_price']) \
        .groupby(['year', 'semestre']).mean() \
        .sort_values(['year', 'semestre'], ascending=True) \
        .plot(figsize=(20, 10), fontsize=20, grid=True, colormap='Dark2')

    ax.set_title("testename", fontsize=20)
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)

    # ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(MultipleLocator())
    ax.xaxis.set_major_locator(MultipleLocator())
    ax.tick_params(which='both', width=2, color='b')
    ax.tick_params(which='major', length=4, color='g')
    ax.tick_params(which='minor', length=6, color='r')

    labels = ax.get_xticklabels()
    plt.setp(labels, rotation=45, horizontalalignment='right')
    ax.set(xlim=[0, 21], xlabel='Total Revenue', ylabel='Company')
    # ax.set_xticks([0, 2, 5, 10, 15, 20, 25])

    # labels[1] = 0
    # ax.set_xticklabels(labels)
    plt.show()


if __name__ == '__main__':
    dt_gas = load_gas_dt()
    dt_petro = load_petroleum_dt()

    graph_total_gas_stations_searched(dt_gas)

    first_insight(dt_gas, dt_petro)

