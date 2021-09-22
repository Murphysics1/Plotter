# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 11:51:45 2021

@author: mikem
"""
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
sns.set()


def Load_File():
    filename = filedialog.askopenfilename()
    root.quit()
    Make_Plot(filename)

def Make_Plot(dataset):
    data = pd.read_csv(dataset)
    #data = list(csv.reader(load))
    data.describe()
    
    x_label = data.columns[0]
    y_label = data.columns[1]
    x = data[x_label]
    y = data[y_label]
    
    model, rmse,r2,y_pred = Model(x,y)
    
    xu = max(x)/20
    yu = max(y)/20
    
    plt.scatter(x,y)
    plt.plot(x,y_pred,c='orange')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.text(1*xu,19*yu,f'Slope: {model.coef_[0]:.2f}')
    plt.text(1*xu,17*yu,f'Intercept: {model.intercept_:.2f}')
    plt.text(1*xu,15*yu,f'R^2: {r2:.2f}')
    plt.show()
    
    
def Model(x,y):
    
    x = x.values.reshape(-1,1)
    
    model = LinearRegression()
    model.fit(x,y)
    y_pred = model.predict(x)

    rmse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    return model,rmse,r2, y_pred
    
root = tk.Tk()
root.title("Plotter")
root.geometry('100x50')
root.resizable(width=False, height=False)

button = tk.Button(root, text='Select Data', command=Load_File)
button.pack()

root.mainloop()