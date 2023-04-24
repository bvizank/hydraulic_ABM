import pandas as pd
import os


def read_data(loc, read_list):
    ''' Function to read in data from either excel or pickle '''
    output = dict()
    data_file = loc + 'datasheet.xlsx'
    pkls = [file for file in os.listdir(loc) if file.endswith(".pkl")]

    for name in read_list:
        index_col = 0
        print("Reading " + name + " data")
        if name == 'seir':
            sheet_name = 'seir_data'
            index_col = 1
        elif name == 'agent':
            sheet_name = 'agent locations'
        else:
            sheet_name = name

        file_name = name + '.pkl'
        if file_name not in pkls:
            print("No pickle file found, importing from excel")
            locals()[name] = pd.read_excel(data_file,
                                           sheet_name=sheet_name,
                                           index_col=index_col)
            if name == 'seir':
                locals()[name].index = locals()[name].index.astype("int64")

            locals()[name].to_pickle(loc + file_name)
        else:
            print("Pickle file found, unpickling")
            locals()[name] = pd.read_pickle(loc + name + '.pkl')

        output[name] = locals()[name]

    return output
