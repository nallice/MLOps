import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import pickle
# from app.controllers import *

class Model():
    def __init__(self):
        self.df = pd.read_csv('app/controllers/df_test_final.csv')
        self.df_y = pd.read_csv('app/controllers/df_test_final_y.csv')

    def predict(self, dados):
        X = self.df.drop(columns=['Neighbourhood'], axis=1)
        y = np.ravel(pd.DataFrame(columns=['Class'], data=self.df_y))
        y = y.astype('int')
        self.model = KNeighborsClassifier(n_neighbors=11).fit(X, y)
        prediction = self.model.predict(dados)
        return prediction[0]

test = Model()
filename = 'modelo.pkl'
pickle.dump(test, open(filename, 'wb'))

