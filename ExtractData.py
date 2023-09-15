import requests
from bs4 import BeautifulSoup
import pandas as pd

excel_file = 'Input.xlsx'
df = pd.read_excel(excel_file)
print(df.head())

def extract_data(url):

    """
    Extract only the heading and the article data and return result
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')
        # class Name of the heading div ...
        div_with_class = soup.find('div', class_='td-post-content')
        heading = soup.find('h1').text

        data = ""
        if div_with_class:
            paragraphs = div_with_class.find_all('p')
            for paragraph in paragraphs:
                data += paragraph.text
        else:
            print(f"Div with class 'td-post-content' not found on the page for URL: {url}")

        return {
            'heading': heading,
            'data': data
        }
    

    except requests.exceptions.RequestException as e:
        print(f"an errror occurred while making the httpp request for url : {url}, error: {e}")
    except Exception as e:
        print(f"an error occurred for URL: {url}, error: {e}")

for index, row in df.iterrows():
    result = extract_data(row.URL)
    if result!=None:
        df.at[index ,'heading'] = result['heading']
        df.at[index ,'data'] = result['data']
    else:
        df.at[index ,'heading'] = None
        df.at[index ,'data'] = None

df.to_csv('Extract.xlsx' , index=False)
df.to_excel('Extract.xlsx', index=False)
