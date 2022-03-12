import os
import os.path
import zipfile
import datetime
import sqlite3

#packages used to download data
import requests
from urllib.request import HTTPError

#packages used to manipulate data
import pandas as pd
from pandas.io.sql import DatabaseError

#other packages
from tqdm import tqdm
from workalendar.america import Brazil
from dateutil.relativedelta import relativedelta

def cvm_informes (year: int, mth: int) -> pd.DataFrame:
    """Downloads the daily report (informe diario) from CVM for a given month and year\n
    <b>Parameters:</b>\n
    year (int): The year of the report the function should download\n
    mth (int): The month of the report the function should download\n
    <b>Returns:</b>\n
    pd.DataFrame: Pandas dataframe with the report for the given month and year. If the year is previous to 2017, will contain data regarding the whole year
   """

    if int(year) >= 2017: #uses download process from reports after the year of 2017
        try:
            mth = f"{mth:02d}"
            year = str(year)
            #creates url using the parameters provided to the function
            url = 'http://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/inf_diario_fi_'+year+mth+'.csv'
            
            #reads the csv returned by the link
            cotas = pd.read_csv(url, sep =';')
            cotas['DT_COMPTC'] = pd.to_datetime(cotas['DT_COMPTC']) #casts date column to datetime
            
            try:
                #removes column present in only a few reports to avoid inconsistency when making the union of reports
                cotas.drop(columns = ['TP_FUNDO'], inplace = True)
            except KeyError:
                pass
            
            return cotas
        except HTTPError:
            print('theres no report for this date yet!.\n')
    
    if int(year) < 2017:
        try:
            year = str(year)

            url = 'http://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/HIST/inf_diario_fi_' + year + '.zip'
            #sends request to the url
            r = requests.get(url, stream=True, allow_redirects=True)
            
            with open('informe' + year + '.zip', 'wb') as fd: #writes the .zip file downloaded
                fd.write(r.content)

            zip_inf = zipfile.ZipFile('informe' + year + '.zip') #opens the .zip file
            
            #le os arquivos csv dentro do arquivo zip
            informes = [pd.read_csv(zip_inf.open(f), sep=";") for f in zip_inf.namelist()] 
            cotas = pd.concat(informes,ignore_index=True)
            
            cotas['DT_COMPTC'] = pd.to_datetime(cotas['DT_COMPTC']) #casts date column to datetime

            zip_inf.close() #fecha o arquivo zip
            os.remove('informe' + year + '.zip') #deletes .zip file
            
            return cotas
        
        except Exception as E:
            print(E)           


def start_db(db_dir: str = 'investments_database.db', start_year: int = 2005, target_funds: list = []):
    """Starts a SQLite database with 3 tables: daily_quotas (funds data), ibov_returns (ibovespa index data) and selic_rates (the base interest rate for the brazilian economy).\n 
    <b>Parameters:</b>\n
    db_dir (str): The path of the dabatabse file to be created. Defaults to 'investments_database.db', creating the file in the current working directory.\n
    start_year (int): Opitional (Defaults to 2005). Starting year for the data collection. . Can be use to reduce the size of the database.\n
    target_funds (list): Opitional (Defaults to []). List of target funds CNPJs. Only funds with CNPJs contained in this list will be included in the database. Can be used to radically reduce the size of the database. If none is specified, all funds will be included.\n
    <b>Returns:</b>\n
    Theres no return from the function.
   """
    ##STEP 1:
    #starts the new database
    print (f'creating SQLite database: {db_dir} \n')
    con = sqlite3.connect(db_dir)


    ##STEP 2:
    #downloads each report in the cvm website and pushes it to the sql database daily_quotas table
    print('downloading daily reports from the CVM website... \n')

    #for each year between 2017 and now
    for year in tqdm(range(start_year, datetime.date.today().year + 1), position = 0, leave=True): 
        for mth in range(1, 13): #for each month
            #loop structure for years equal or after 2017
            if year>=2017: 
                informe = cvm_informes(str(year), mth)

                try:
                    if target_funds: #if the target funds list is not empty, uses it to filter the result set
                        informe = informe[informe.CNPJ_FUNDO.isin(target_funds)]
                    #appends information to the sql database
                    informe.to_sql('daily_quotas', con , if_exists = 'append', index=False)
                except AttributeError:
                    pass
            
            elif year<2017: #loop structure to handle years before 2017 (they have a different file structure)
                #only executes the download function once every year to avoid duplicates (unique file for each year)       
                if mth == 12:
                    informe = cvm_informes(str(year), mth)

                    try:
                        if target_funds: #if the target funds list is not empty, uses it to filter the result set
                            informe = informe[informe.CNPJ_FUNDO.isin(target_funds)]
                        #appends information to the sql database
                        informe.to_sql('daily_quotas', con , if_exists = 'append', index=False)
                    except AttributeError:
                        pass

    #pushes target funds to sql for use when updating the database
    if target_funds:
        target_df = pd.DataFrame({'targets':target_funds})
        target_df.to_sql('target_funds', con , index=False)                    
    ##STEP 3:                    
    #creates index in the daily_quotas table to make future select queries faster. 
    #tradeoff: The updating proceesses of the database will be slower.
    print('creating sql index on "CNPJ_FUNDO", "DT_COMPTC" ... \n')
    index = '''
    CREATE INDEX "cnpj_date" ON "daily_quotas" (
        "CNPJ_FUNDO" ASC,
        "DT_COMPTC" ASC
    )'''

    cursor = con.cursor()
    cursor.execute(index)
    con.commit()

    cursor.close()

    
    ##STEP 4:
    #downloads cadastral information from CVM of the fundos and pushes it to the database
    print('downloading cadastral information from cvm...\n')
    info_cad = pd.read_csv('http://dados.cvm.gov.br/dados/FI/CAD/DADOS/cad_fi.csv', sep = ';', encoding='latin1',
                           dtype = {'RENTAB_FUNDO': object,'FUNDO_EXCLUSIVO': object, 'TRIB_LPRAZO': object, 'ENTID_INVEST': object,
                                    'INF_TAXA_PERFM': object, 'INF_TAXA_ADM': object, 'DIRETOR': object, 'CNPJ_CONTROLADOR': object,
                                    'CONTROLADOR': object}
                            )
    if target_funds:
        info_cad = info_cad[info_cad.CNPJ_FUNDO.isin(target_funds)]
    info_cad.to_sql('info_cadastral_funds', con, index=False)


    ##STEP 5:
    #creates a table with a log of the execution timestamps of the script
    print('creating the log table...\n')
    update_log = pd.DataFrame({'date':[datetime.datetime.now()], 'log':[1]})
    update_log.to_sql('update_log', con, if_exists = 'append', index=False)


    ##STEP 6
    #closes the connection with the database
    con.close()
    print('connection with the database closed! \n')

    print(f'Success: database created in {db_dir} !\n')


def update_db(db_dir: str = r'investments_database.db'):
    """Updates the database.\n
    <b>Parameters:</b>\n
    db_dir (str): The path of the dabatabse file to be updated. Defaults to 'investments_database.db'.\n
    <b>Returns:</b>\n
    Theres no return from the function.
   """
    ##STEP 1
    #connects to the database
    print(f'connected with the database {db_dir}\n')
    con = sqlite3.connect(db_dir)


    ##STEP 2
    #calculates relevant date limits to the update process
    Cal=Brazil() #inicializes the brazillian calendar
    today = datetime.date.today()

    #queries the last update from the log table
    last_update = pd.to_datetime(pd.read_sql('select MAX(date) from update_log', con).iloc[0,0])

    last_quota = Cal.sub_working_days(last_update, 2) #date of the last published cvm repport
    num_months = (today.year - last_quota.year) * 12 + (today.month - last_quota.month) + 1


    ##STEP 3
    #delete information that will be updated from the database tables
    print('deleting redundant data from the database... \n')
    tables = {'daily_quotas' : ['DT_COMPTC',last_quota.strftime("%Y-%m-01")],
              'ibov_returns' : ['Date',last_update.strftime("%Y-%m-%d")]}
    
    cursor = con.cursor()
    
    #sql delete statement to the database
    cursor.execute('delete from daily_quotas where DT_COMPTC >= :date', {'date': last_quota.strftime("%Y-%m-01")})
    cursor.execute('delete from ibov_returns where Date >= :date', {'date': last_update.strftime("%Y-%m-%d")})
        
    con.commit()  
    cursor.close()


    ##STEP 4
    #Pulls new data from CVM, investpy and the brazilian central bank
    #and pushes it to the database

    try:#tries to read targets funds if they were specified when starting the database
        target_funds = pd.read_sql('select targets from target_funds', con).targets.to_list()
    except DatabaseError:
        target_funds = []
    
    print('downloading new daily reports from the CVM website...\n')
    # downloads the daily cvm repport for each month between the last update and today
    for m in range(num_months+1): 
        data_alvo = last_quota + relativedelta(months=+m) 
        informe = cvm_informes(data_alvo.year, data_alvo.month)
        if target_funds:
            informe = informe[informe.CNPJ_FUNDO.isin(target_funds)]
        try:
            informe.to_sql('daily_quotas', con , if_exists = 'append', index=False)
        except AttributeError:
            pass 

    #downloads cadastral information from CVM of the fundos and pushes it to the database
    print('downloading updated cadastral information from cvm...\n')
    info_cad = pd.read_csv('http://dados.cvm.gov.br/dados/FI/CAD/DADOS/cad_fi.csv', sep = ';', encoding='latin1',
                           dtype = {'RENTAB_FUNDO': object,'FUNDO_EXCLUSIVO': object, 'TRIB_LPRAZO': object, 'ENTID_INVEST': object,
                                    'INF_TAXA_PERFM': object, 'INF_TAXA_ADM': object, 'DIRETOR': object, 'CNPJ_CONTROLADOR': object,
                                    'CONTROLADOR': object}
                            )
    if target_funds: #filters target funds if they were specified when building the database.
        info_cad = info_cad[info_cad.CNPJ_FUNDO.isin(target_funds)]
    info_cad.to_sql('info_cadastral_funds', con, if_exists='replace', index=False)

    ##STEP 3
    #updates the log in the database
    print('updating the log...\n')
    update_log = pd.DataFrame({'date':[datetime.datetime.now()], 'log':[1]})
    update_log.to_sql('update_log', con, if_exists = 'append', index=False)


    ##STEP 7
    #closes the connection with the database
    con.close()
    print('connection with the database closed!\n')

    print(f'database {db_dir} updated!\n')

