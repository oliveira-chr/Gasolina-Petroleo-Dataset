import requests

query = {'dataInicial': '01-01-2021',
         'dataFinalCotacao': '01-03-2021'}

response = requests.get("https ://olinda.bcb.gov.br/olinda/servico/PTAX/versao/v1/odata/CotacaoDolarPeriodo?$format=json", params=query)

print(response)
print(response.content)  # Return the raw bytes of the data payload
print(response.text)  # Return a string representation of the data payload
print(response.json())  # This method is convenient when the API returns JSON
