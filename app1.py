import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pymongo
import altair as alt
import plotly.graph_objects as go
from tensorflow import keras 
from keras.models import load_model
import re
import datetime
from datetime import timedelta


# Funcion que devuelve el ticker entre paréntesis del activo de la lista 
def getTicker(cad):
    m = re.match("[A-Za-z]+ \(([A-Za-z]+)\)", cad)
    if m:
        return m.group(1).lower()
#En el caso de que no se cumpla la "realidad necesaria" de las velas hay que reajustar. High es el mayor de los 4 y Low el menor.
def Normaliza(valores):
    #Al realizar la estimación, hay valores que pueden quedar fuera de rango, por lo que ajustamos high a maximo y low a minimo
    valores[1] = max(valores)
    valores[2] = min(valores)


def main():
    menu = ["Nasdaq (NQ)"]
    timeframe = ["5m"]
    choice_activo = st.sidebar.selectbox("Activos",menu)
    choice_tf = st.sidebar.selectbox("Marco Temporal",timeframe)
    st.title("Pronósticos Redes Neuronales")
    
    
    # Conexión con la base de datos para obtener los datos a descargar
    client = pymongo.MongoClient("mongodb+srv://tfmunedesm:Ab123456@cluster0.c1gnb.mongodb.net/test?retryWrites=true&w=majority")
    db = client.test
    resul=db.nq5m.find().sort("Date",-1).limit(20)
    
    dfdb = pd.DataFrame(resul)
    dfdb.drop("_id",axis=1,inplace=True)
    dftmp = dfdb[['Date','Open','High','Low','Close']]
    dftmp.set_index('Date',inplace=True)
    dftmp.sort_index(inplace=True)
    st.dataframe(dftmp)

    
    # Inicializacion de diccionarios que contendra los modelos
    fields = ['High','Low','Close']
    valoresP = {} #Almacena los n ultimos valores para pasar al modelo y hacer el pronostico
    model_mlp = {}
    model_vanilla = {}
    model_stacked = {}
    model_bidirectional = {}
    model_cnn = {}
    #Carga de modelos y valores para las predicciones
    for f in fields:
        valoresP[f] = np.array( [dftmp.iloc[-3][f], dftmp.iloc[-2][f], dftmp.iloc[-1][f]])
        model_mlp[f] = load_model("modelos/"+getTicker(choice_activo)+"_"+choice_tf+"/MLP_"+f+".h5")
        model_vanilla[f] = load_model("modelos/"+getTicker(choice_activo)+"_"+choice_tf+"/Vanilla_"+f+".h5")
        model_stacked[f] = load_model("modelos/"+getTicker(choice_activo)+"_"+choice_tf+"/Stacked_"+f+".h5")
        model_bidirectional[f] = load_model("modelos/"+getTicker(choice_activo)+"_"+choice_tf+"/MiBidirectional_"+f+".h5")
        model_cnn[f] = load_model("modelos/"+getTicker(choice_activo)+"_"+choice_tf+"/MiCNN_"+f+".h5")


    #Realizacion de predicciones.
    mlpp = []
    vanillap = []
    stackedp = []
    bidirectionalp = []
    cnnp = []

    mlpp.append(dftmp.iloc[-1]['Close'])
    vanillap.append(dftmp.iloc[-1]['Close'])
    stackedp.append(dftmp.iloc[-1]['Close'])
    bidirectionalp.append(dftmp.iloc[-1]['Close'])
    cnnp.append(dftmp.iloc[-1]['Close'])

    for f in fields:
        mival = model_mlp[f].predict(valoresP[f].reshape((1, 3)),verbose=0)
        mlpp.append(float(mival[0][0]))

        mival = model_vanilla[f].predict(valoresP[f].reshape((1, 3, 1)),verbose=0)
        vanillap.append(float(mival[0][0]))
        mival = model_stacked[f].predict(valoresP[f].reshape((1, 3, 1)),verbose=0)
        stackedp.append(float(mival[0][0]))
        mival = model_bidirectional[f].predict(valoresP[f].reshape((1, 3, 1)),verbose=0)
        bidirectionalp.append(float(mival[0][0]))
        mival = model_cnn[f].predict(valoresP[f].reshape((1, 3, 1)),verbose=0)
        cnnp.append(float(mival[0][0]))
       
    #Nos aseguramos del cumplimiento de High valor máximo y Low valor mínimo de los 4 de la vela
    Normaliza(mlpp)
    Normaliza(vanillap)
    Normaliza(stackedp)
    Normaliza(bidirectionalp)
    Normaliza(cnnp)

    

    # Creación de los dataframes con las predicciones MLP
    st.write("## MLP")
    df_mlp = dftmp.copy()
    df_mlp.reset_index(inplace=True)
    mlpp.insert(0,df_mlp.iloc[-1]['Date']+timedelta(minutes=5))
    dfaux = pd.DataFrame(mlpp)
    dfaux=dfaux.transpose()
    dfaux.columns=['Date','Open','High','Low','Close']
    
    df_mlp=df_mlp.append(dfaux)
    df_mlp.set_index("Date")
    

    # Representación del DataFrame MLP
    base = alt.Chart(df_mlp).encode(
    alt.X('Date:T', axis=alt.Axis(labelAngle=-45)),
    color=alt.condition("datum.Open <= datum.Close",
                        alt.value("#06982d"), alt.value("#ae1325")))

    chart = alt.layer(base.mark_rule().encode(alt.Y('Low:Q', title='Price',
                                    scale=alt.Scale(zero=False)), alt.Y2('High:Q')), base.mark_bar().encode(alt.Y('Open:Q'), alt.Y2('Close:Q')),).interactive()
    st.altair_chart(chart, use_container_width=True)

    # Creación de los dataframes con las predicciones Vanilla
    st.write("## Vanilla LSTM")
    df_vanilla = dftmp.copy()
    df_vanilla.reset_index(inplace=True)
    vanillap.insert(0,df_vanilla.iloc[-1]['Date']+timedelta(minutes=5))
    dfaux = pd.DataFrame(vanillap)
    dfaux=dfaux.transpose()
    dfaux.columns=['Date','Open','High','Low','Close']
    
    df_vanilla=df_vanilla.append(dfaux)
    df_vanilla.set_index("Date") 

    # Representación del DataFrame Vanilla   
    base = alt.Chart(df_vanilla).encode(
    alt.X('Date:T', axis=alt.Axis(labelAngle=-45)),
    color=alt.condition("datum.Open <= datum.Close",
                        alt.value("#06982d"), alt.value("#ae1325")))

    chart = alt.layer(base.mark_rule().encode(alt.Y('Low:Q', title='Price',
                                    scale=alt.Scale(zero=False)), alt.Y2('High:Q')), base.mark_bar().encode(alt.Y('Open:Q'), alt.Y2('Close:Q')),).interactive()
    st.altair_chart(chart, use_container_width=True) 

    modeloMLPH = keras.models.load_model("./Documents/MLPH.h5")
   
    # Creación de los dataframes con las predicciones Stacked
    st.write("## Stacked LSTM")
    df_stacked = dftmp.copy()
    df_stacked.reset_index(inplace=True)
    stackedp.insert(0,df_stacked.iloc[-1]['Date']+timedelta(minutes=5))
    dfaux = pd.DataFrame(stackedp)
    dfaux=dfaux.transpose()
    dfaux.columns=['Date','Open','High','Low','Close']
    
    df_stacked=df_stacked.append(dfaux)
    df_stacked.set_index("Date")    
    # Representación del DataFrame Stacked 
    base = alt.Chart(df_stacked).encode(
    alt.X('Date:T', axis=alt.Axis(labelAngle=-45)),
    color=alt.condition("datum.Open <= datum.Close",
                        alt.value("#06982d"), alt.value("#ae1325")))

    chart = alt.layer(base.mark_rule().encode(alt.Y('Low:Q', title='Price',
                                    scale=alt.Scale(zero=False)), alt.Y2('High:Q')), base.mark_bar().encode(alt.Y('Open:Q'), alt.Y2('Close:Q')),).interactive()
    st.altair_chart(chart, use_container_width=True) 
 
    # Creación de los dataframes con las predicciones Bidirectional
    st.write("## Bidirectional LSTM")
    df_bidirectional = dftmp.copy()
    df_bidirectional.reset_index(inplace=True)
    bidirectionalp.insert(0,df_bidirectional.iloc[-1]['Date']+timedelta(minutes=5))
    dfaux = pd.DataFrame(bidirectionalp)
    dfaux=dfaux.transpose()
    dfaux.columns=['Date','Open','High','Low','Close']
    # Representación del DataFrame Bidirectional 
    df_bidirectional=df_bidirectional.append(dfaux)
    df_bidirectional.set_index("Date")    
    base = alt.Chart(df_bidirectional).encode(
    alt.X('Date:T', axis=alt.Axis(labelAngle=-45)),
    color=alt.condition("datum.Open <= datum.Close",
                        alt.value("#06982d"), alt.value("#ae1325")))

    chart = alt.layer(base.mark_rule().encode(alt.Y('Low:Q', title='Price',
                                    scale=alt.Scale(zero=False)), alt.Y2('High:Q')), base.mark_bar().encode(alt.Y('Open:Q'), alt.Y2('Close:Q')),).interactive()
    st.altair_chart(chart, use_container_width=True)    

    # Creación de los dataframes con las predicciones CNN
    st.write("## CNN")
    df_cnn = dftmp.copy()
    df_cnn.reset_index(inplace=True)
    cnnp.insert(0,df_cnn.iloc[-1]['Date']+timedelta(minutes=5))
    dfaux = pd.DataFrame(cnnp)
    dfaux=dfaux.transpose()
    dfaux.columns=['Date','Open','High','Low','Close']
    
    df_cnn=df_cnn.append(dfaux)
    df_cnn.set_index("Date") 
    # Representación del DataFrame CNN LSTM   
    base = alt.Chart(df_cnn).encode(
    alt.X('Date:T', axis=alt.Axis(labelAngle=-45)),
    color=alt.condition("datum.Open <= datum.Close",
                        alt.value("#06982d"), alt.value("#ae1325")))

    chart = alt.layer(base.mark_rule().encode(alt.Y('Low:Q', title='Price',
                                    scale=alt.Scale(zero=False)), alt.Y2('High:Q')), base.mark_bar().encode(alt.Y('Open:Q'), alt.Y2('Close:Q')),).interactive()
    st.altair_chart(chart, use_container_width=True) 






if __name__ == "__main__":
    main()