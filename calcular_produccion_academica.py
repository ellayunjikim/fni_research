import sys
import os

import pandas as pd
from pybliometrics.scopus import AuthorRetrieval

HELP_MESSAGE = 'Usage:\npython {} --data=<input_file> --output=<output_dir>'
INVALID_ARG_NUMBER = 'Expected 2 argument, but {} received.'
INSTRUCTIONS_MESSAGE = 'data: publications file'

#Output files
COLLABORATION_FILE = "collaboration.csv"
RESEARCHERS_FILE = "researchers.csv"
RESEARCHER_PUBLICATION_FILE = "author_publication.csv"

def get_researchers(pub_data, researchers_list):
    authors = pub_data.author_ids
    try:
        coauthors = authors.split(";")
        for coauthor in coauthors:
            researchers_list.append(coauthor)
    except AttributeError:
        print("Authors not found for publication {}".format())
    
def get_coauthors(researcher_pubs):
    list_coauthors = []
    for authors in researcher_pubs.author_ids:
        coauthors = authors.split(";")
        for coauthor in coauthors:
            list_coauthors.append(coauthor.strip("''"))
        
    list_coauthors = list(set(list_coauthors))
    if '' in list_coauthors:
        list_coauthors.remove('')
    return list_coauthors
    
def get_data(researcher_pubs):
        num_publications = len(researcher_pubs)
        num_cited_by = researcher_pubs.citedby_count.sum()
        last_pub_date=researcher_pubs.coverDate.max()
        return num_publications, num_cited_by, last_pub_date
        
def search_researcher_afilliation(researcher_id):
    country = "Not found"
    affiliation = "Not found"
    request = True
    
    while (request):
        try:
            author_info = AuthorRetrieval(researcher_id)
            request = False
        except ConnectionError:
            print("Connection error, retrying in 10 seconds")
            sleep(10)
            
    if (author_info.affiliation_current is not None):
        country = author_info.affiliation_current[0].country
        affiliation = author_info.affiliation_current[0].preferred_name
    return country, affiliation
    
def get_researcher_data(researcher_pubs, author):
    num_publications, num_cited_by, last_pub_date = get_data(researcher_pubs)
    
    network_keywords=researcher_pubs.authkeywords[~researcher_pubs['authkeywords'].isna()].sum()
    num_network_keywords = 0
    if (type(network_keywords) is not int):
        num_network_keywords = len(set(network_keywords.split(" | ")))
    
    last_pub = researcher_pubs.loc[researcher_pubs["coverDate"]==last_pub_date].iloc[0]
    index_autor = last_pub["author_ids"].split(";").index(author)
    name = last_pub["author_names"].split(";")[index_autor]
    country, affiliation = search_researcher_afilliation(author)
           
    return [author,name,num_publications,num_cited_by,num_network_keywords,last_pub_date,country,affiliation]

def get_collaboration(input_file, output_dir):
    publications_df = pd.read_csv(input_file, sep='\t')
    researchers_list = []
    publications_df.apply(lambda row: get_researchers(row, researchers_list), axis=1)
    researchers_list = list(set(researchers_list))

    collaboration_data = []
    researchers_data = []
    researcher_pub_data = []

    for i in range(len(researchers_list)):
        researcher_a = str(researchers_list[i])
        sys.stdout.write("Current researcher: %s --- Progress: %s%%  \r" % (researcher_a,100*(i+1)/len(researchers_list)))
        
        researcher_a_pubs = publications_df[publications_df.author_ids.str.contains(researcher_a, na=False)]
        researchers_data.append(get_researcher_data(researcher_a_pubs, researcher_a))
        
        researcher_a_coauthors = get_coauthors(researcher_a_pubs)
        for researcher_b in researcher_a_coauthors:
            if (researcher_a != researcher_b):
                collaboration_pubs = researcher_a_pubs.loc[(researcher_a_pubs['author_ids'].str.contains(researcher_b))]
                if len(collaboration_pubs)>0:
                    num_publications, num_cited_by, last_pub_date = get_data(collaboration_pubs)
                    collaboration_data.append([researcher_a,researcher_b,num_publications, num_cited_by, last_pub_date])
         
        for pub_id in researcher_a_pubs.eid:
            researcher_pub_data.append([researcher_a, pub_id])
            
    collaboration_df = pd.DataFrame(collaboration_data, columns = ['source', 'target', 'num_publications','num_cited_by','last_pub_date'])
    researchers_df = pd.DataFrame(researchers_data, columns = ["scopusID","name","num_publicaciones","num_cited_by","num_network_keywords","last_pub_date","country","affiliation"])
    researcher_pub_df = pd.DataFrame(researcher_pub_data, columns = ['scopusID', 'publication'])

    researchers_df.drop_duplicates(subset="scopusID", inplace=True)
    researcher_pub_df.drop_duplicates(inplace=True)

    collaboration_df.to_csv(os.path.join(output_dir, f'{COLLABORATION_FILE}'), index=False)
    researchers_df.to_csv(os.path.join(output_dir, f'{RESEARCHERS_FILE}'), index=False)
    researcher_pub_df.to_csv(os.path.join(output_dir, f'{RESEARCHER_PUBLICATION_FILE}'), index=False)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(INVALID_ARG_NUMBER.format(len(sys.argv) - 1))
        print(HELP_MESSAGE.format(sys.argv[0]))
        print(INSTRUCTIONS_MESSAGE)
        exit(1)
        
    input_file = ''
    output_dir = ''
    
    for arg in sys.argv[1:]:
        if arg.startswith('--data='): input_file = arg.split('=')[-1]
        elif arg.startswith('--output=') : output_dir = arg.split('=')[-1]
        
    get_collaboration(input_file, output_dir)