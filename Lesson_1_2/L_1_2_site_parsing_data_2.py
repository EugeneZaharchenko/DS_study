
'''
Приклад парсингу табличних даних - пандемічні захворювання
https://www.worldometers.info/coronavirus/
'''


import pandas as pd
import re
import requests
import io


#---------------- Парсер САЙТУ для отримання числових даних в dataframe pandas ----------------
def Parsing_Site_coronavirus(url_DS):

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, '
                      'like Gecko) Chrome/116.0.0.0 Safari/537.36'
    }
    html_source = requests.get(url_DS, headers=headers).text

    html_source = re.sub(r'<.*?>', lambda g: g.group(0).upper(), html_source)
    dataframe = pd.read_html(io.StringIO(html_source))

    print(dataframe)

    return dataframe


if __name__ == '__main__':

    print('Парсинг табличних даних https://www.worldometers.info/coronavirus/')
    url_DS = "https://www.worldometers.info/coronavirus/"
    Parsing_Site_coronavirus(url_DS)
