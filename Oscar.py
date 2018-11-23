import re
import nltk
import pandas as pd
import numpy as np
from unidecode import unidecode

def O_print_full(x):
    pd.set_option('display.max_rows', len(x))
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.float_format', '{:20,.2f}'.format)
    pd.set_option('display.max_colwidth', -1)
    print(x)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_colwidth')

def O_word_frequency(df_text, s, idioma = "portuguese"):
    """
    Implement a function O_word_frequency() in Python that takes as input
    a dataframe df_text, a string s, and the idioma (the idioma will be to
    define the list of stop_words used by nltk). This funtion returns a datafrime 
    with the n most frequently-occuring words in column s.
    """
    dfList = df_text[s].tolist()
    dfList2 = " ".join(re.sub('[^A-Za-z]', ' ', unidecode(str(x).lower().replace("\r", " ").replace("\n", " ").replace(".", " "))) for x in dfList).replace("  ", " ").replace("  ", " ").replace("  ", " ")

    # Make a unique word list
    words = dfList2.split()
    words_unique = set(words)
    words_lst = list(words_unique)
    
    stop_words = nltk.corpus.stopwords.words(idioma)
    for i in stop_words:
        if i in words_lst:
            words_lst.remove(i)
    
    # Make a corresponding count list
    count_lst = []
    for word in words_lst:
        count = words.count(word)
        count_lst.append(count)

    # Build the tuple list combining words and counts
    d = {'word': words_lst, 'frequency': count_lst}
    df = pd.DataFrame(data=d).sort_values(['frequency'], ascending=False)

    return df


def O_check_null(base):
    s_types = base.dtypes
    df_types = pd.DataFrame({'columns':s_types.index, 'types':s_types.values})
    print ('Column  |  Type  |  Missing')
    for i in base.columns:
        print (i + ' (' + str((df_types.loc[df_types['columns'] == i, 'types'].iloc[0])) + ') : ' + str(sum(base[i].isnull())))
        
def O_check_base(base):
    s_types = base.dtypes
    
    s_miss = np.zeros(base.columns.size)
    j=0
    for i in base.columns:
        s_miss[j] = sum(base[i].isnull())
        j+=1

    df = pd.DataFrame({'columns':base.columns, 'types':s_types.values, 'missing':s_miss})
        
    return df[['columns','types','missing']] 


def O_input_column(df1, df2, id1, id2, idNew, tp='begin'):
    if tp == 'end':
        result = pd.merge(df1,
                          df2[[id2,idNew]], 
                          left_on=id1,
                          right_on=id2,
                          how='left')        
    else:
        result = pd.merge(df2[[id2,idNew]], 
                          #df_idi[['CodCand_idioma','Idioma_idioma']],
                          df1,
                          left_on=id2,
                          right_on=id1,
                          how='right')

    if id1 == id2:
        return result
    else: 
        return result.drop(id2, 1) #retirando coluna duplicada

    
    
def O_join_df(df1, df2, id1, id2):
        result = pd.merge(df2, 
                          df1,
                          left_on=id2,
                          right_on=id1,
                          how='right')
        #result.rename(columns={"N_CPF": "Nro_doc"}, inplace=True)
        return result
    
    
def O_grouping_column(df_exp, fila):
    return df_exp[[fila]].groupby([fila])[fila] \
                             .count() \
                             .reset_index(name='idi_count') \
                             .sort_values(['idi_count'], ascending=False)
            
def O_filter_last(df_cpfs, col1, ultima, argumento='max'):
    return df_cpfs[df_cpfs.groupby(col1).CodCand_doc.transform(argumento) == df_cpfs[ultima]]


def O_count_words(x):
    y = x.split()
    if x is np.nan or x =="":
        return 0
    return len(y)
	
	
def O_add_missing_dummy_columns(new_data, train_columns):
	missing_cols = set( train_columns ) - set( new_data.columns )
	for c in missing_cols:
		new_data[c] = 0

def O_fix_columns(new_data, train_columns):  
	add_missing_dummy_columns(new_data, train_columns)

	# make sure we have all the columns we need
	assert(set(train_columns) - set(new_data.columns) == set())

	extra_cols = set(new_data.columns) - set(train_columns)
	if extra_cols:
		print ("colunas extras na base teste", extra_cols)

	new_data = new_data[ train_columns ]
	return new_data

def O_nearZeroVariance(v, freqCut, uniqueCut = 0.10):
    if len (v.value_counts()) == 1:
        return True
    mostFrequent = v.value_counts()[v.value_counts().keys()[0]]
    secondMostFrequent = v.value_counts()[v.value_counts().keys()[1]]
    if mostFrequent / secondMostFrequent >= freqCut and len (v.value_counts()) / len (v) <= uniqueCut:
            return True
    return False

def O_get_nearZero(X_train, freqCut = 76/3, uniqueCut = 0.005):
	array_nzv = []
	for i in X_train.columns:
		nzv = nearZeroVariance (X_train[i],freqCut = 76/3, uniqueCut = 0.005)
		array_nzv.append([i, nzv])
	return array_nzv

