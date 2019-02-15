from flask import Flask, request, jsonify
from flask_cors import CORS

import pandas as pd
from sklearn.cluster import KMeans

app = Flask( __name__ )
CORS( app )

@app.route( '/' )
def hello():
    return "i2DR"

@app.route( '/kmeans/<n_clusters>', methods = [ 'POST' ] )
def kmeans( n_clusters ):
    
    # Parse body request to dataframe
    data_dict = []
    data_json = request.get_json()
    for d in data_json:
        data_dict.append( dict( d ) )
    data_df = pd.DataFrame( data_dict, dtype = int )

    # Cluster data
    clusters = KMeans( n_clusters = int( n_clusters ), random_state = 0 ).fit_predict( data_df )

    return " ".join( str( c ) for c in clusters )


if __name__ == '__main__':
    app.run()