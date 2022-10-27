from time import sleep
from datetime import date
import sys
import os

import pandas as pd
from pybliometrics.scopus import ScopusSearch

HELP_MESSAGE = 'Usage:\npython {} --kw=<keywords> --output=<output_dir> optional flags: --countries --year'
INVALID_ARG_NUMBER = 'Expected 2 argument, but {} received.'
INSTRUCTIONS_MESSAGE = 'keywords: comma-separated\ncountries_file: list of countries (csv file)\nyear: "from" year'


SEARCH_QUERY = "{} AND PUBYEAR AFT {} AND ({})"
SEARCH_QUERY_NO_COUNTRIES = "{} AND PUBYEAR AFT {}"
SEARCH_FIELD = "TITLE-ABS-KEY({})"
COUNTRY_FIELD = "AFFILCOUNTRY({})"
YEAR_AFT = date.today().year - 11

#Output files
PUBLICATIONS_FILE = "publicaciones_{}_{}.csv"

def create_query(keywords_list, year, list_countries=None):
    keywords_list = keywords_list.split(",")
    subquery_keywords = ""
    for i, keyword in enumerate(keywords_list):
        subquery_keywords += SEARCH_FIELD.format(keyword)
        if (i != len(keywords_list)-1):
            subquery_keywords += " AND "
            
    if list_countries is None:
        return SEARCH_QUERY_NO_COUNTRIES.format(subquery_keywords, year)
        
    subquery_countries = ""
    for i, pais in enumerate(list_countries):
        subquery_countries += COUNTRY_FIELD.format(pais)
        if (i != len(list_countries)-1):
            subquery_countries += " OR "
    return SEARCH_QUERY.format(subquery_keywords, year, subquery_countries)

def get_scopus(keywords_list, countries_file, year):
    if (countries_file is None):
        query = create_query(keywords_list, year)
    else:
        countries_df = pd.read_csv(countries_file)
        list_countries = countries_df["country"].unique()
        query = create_query(keywords_list, year, list_countries)
    print("Query generated for Scopus Search API: {}".format(query))
    request = True
    while(request):
        try:
            publications = ScopusSearch(query, refresh=6)
            request = False
        except ConnectionError:
            print("Connection error, retrying in 10 seconds")
            sleep(10)
    try:
        pubs_df = pd.DataFrame(pd.DataFrame(publications.results))
    except TypeError:
        pubs_df = []
        return None
    
    if(len(pubs_df)==0):
        return None    
    return pubs_df

def download_publications(keywords_list, output_dir, countries_file, year):
    pubs_df = get_scopus(keywords_list, countries_file, year)
    pubs_df.drop_duplicates(subset="eid", inplace=True)
    pubs_df.to_csv(os.path.join(output_dir, f'{PUBLICATIONS_FILE.format(keywords_list, year)}'), index=False, sep='\t')

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print(INVALID_ARG_NUMBER.format(len(sys.argv) - 1))
        print(HELP_MESSAGE.format(sys.argv[0]))
        print(INSTRUCTIONS_MESSAGE)
        exit(1)

    keywords_list = ''
    output_dir = ''
    countries_file = None
    year = YEAR_AFT
    
    for arg in sys.argv[1:]:
        if arg.startswith('--kw='): list_keywords = arg.split('=')[-1]
        elif arg.startswith('--output=') : output_dir = arg.split('=')[-1]
        elif arg.startswith('--countries=') : countries_file = arg.split('=')[-1]
        elif arg.startswith('--year=') : year = int(arg.split('=')[-1])
    
    download_publications(list_keywords, output_dir, countries_file, year)