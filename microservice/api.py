import flask
import pickle
from flask import request
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

class Model():
    def __init__(self,model,minMaxScaler, encoder):
        self.model=model
        self.minMaxScaler=minMaxScaler
        self.encoder=encoder
        
    def preprocess(self,df):
        # fix format of float variables
        for var in ["variable2", "variable3","variable8"]:
            df[var]=df[var].apply(func=lambda x: float(str(x).replace(',','.')))

        # handle NAN values
        df.drop(["variable18"],axis =1,inplace=True)
        df.dropna(inplace=True)
        df.reset_index(inplace=True,drop=True)

        redundant_cols = ["variable17","variable19"]
        df.drop(columns=redundant_cols,inplace=True)
        
        # normalization and one hot encoding
        min_max_columns = [columnName for (columnName, column) in df.iteritems() if column.dtype != np.object]
        one_hot_columns = [columnName for (columnName, column) in df.iteritems() if column.dtype == np.object]
        
        # data normalization
        df[min_max_columns] = self.minMaxScaler.transform(df[min_max_columns])        

        # encode categorical variables
        one_hot_encdoing = pd.DataFrame(self.encoder.transform(df[one_hot_columns]).toarray())
        df = pd.concat([df,one_hot_encdoing],axis=1)
        df = df.drop(one_hot_columns,axis=1)
        return df
    
    def predict(self,df):
        x = self.preprocess(df)
        preds = self.model.predict(x).astype(str)
        mask = preds == '1'
        preds[mask] = "yes."
        preds[~mask] = "no."
        return preds

model = pickle.load(open("model","rb"))
app = flask.Flask(__name__)

@app.route('/predict', methods=['post'])
def predict():
    df = pd.read_json(request.data)
    df.columns = ["variable1","variable2","variable3","variable4","variable5","variable6","variable7","variable8","variable9","variable10","variable11","variable12","variable13","variable14","variable15","variable17","variable18","variable19"]
    return flask.jsonify(model.predict(df).tolist())

app.run()
