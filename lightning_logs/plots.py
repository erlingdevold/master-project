import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import yaml
def read_params(file):
    # reads hparams.yaml file and returns params as dict

    with open(file) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    
    return params
def parse_params(params):
    if params['criterion'] == 'bce':
        if params['n_output'] > 1:
           return 'multi'
        else:
            return 'binary'
        
    return 'regression'





def plot(file : str , ):
    try:
        df = pd.read_csv( file + '/metrics.csv')
        params = read_params(file + '/hparams.yaml')
    except FileNotFoundError:
        print('FileNotFoundError')
        return

    list = [df['val_loss_epoch'].dropna(), df['val_loss_epoch'].dropna(), ]
    x = 'regression'
    print(params)
    print(df.columns)
    if (x := parse_params(params) ) == 'multi' :
        list = [df['val_acc_epoch'].dropna(), df['val_recall_epoch'].dropna(), df['val_precision_epoch'].dropna(), df['val_loss_epoch'].dropna()]
    if (x := parse_params(params) ) == 'binary' :
        list = [df['val_acc_epoch'].dropna(), df['val_recall_epoch'].dropna(), df['val_precision_epoch'].dropna(), df['val_loss_epoch'].dropna()]

    
    return x, params, list

def plot_dict(dict,num_rows=2):
    for model_type in dict:
        print(dict[model_type])
        try:
            num_plots = len(list(dict[model_type].values())[0][0])
        except:
            continue
        print(num_plots)
        fig,ax = plt.subplots(num_plots,2,figsize=(10,10))
        
        handles = []
        for i,threshold in enumerate(dict[model_type]):
            for t in range(len(dict[model_type][threshold])):
                for j,metric in enumerate(dict[model_type][threshold][t]):
                        try:
                            h = ax[j][t].plot(metric.values,label= threshold.replace('_',' ' + 'km labels') )
                        except:
                            continue
                        ax[j][t].set_ylabel(metric.name.split('_')[1].capitalize())
                        ax[j][t].grid()
        labels = [ item + ' km threshold label' for item in list(dict[model_type].keys())]
        fig.legend(labels,loc='upper center', ncol=4)
        # fig.suptitle(model_type)

        plt.savefig('plots/models/' +model_type + '.png',bbox_inches='tight',dpi=300)
        

        plt.close()


import os
if __name__ == "__main__":
    dirs = os.listdir('lightning_logs')
    dirs.sort()
    # threshold_dict = { "1_1" :{'multi' : [],'regression' : []}, "_5" : {}, "_10" : {}, "_20": {}}
    threshold_dict = {'multi' : {},'regression' : {},'binary' : {}}
    
    for dir in dirs:
        if dir.startswith('version'):
            try:
                type, params, l  =plot('lightning_logs/' + dir )
            except :
                continue
            # if params['threshold'] not in threshold_dict[type]:
                
            thresh = params['threshold'].split('_')[1]
            if thresh not in threshold_dict[type]:
                threshold_dict[type][thresh] = [[0],[0]]
            threshold_dict[type][thresh][params['temporal']] = l

    # print(threshold_dict['multi'].keys())
    plot_dict(threshold_dict)
"""
eg foreslår følgende struktur her: seksjonn requirements åpner med lsiten av req's slik som den gjør akkurat nå. deretter deler du section requirements inn i subsections der du beskriver dypere hvert punkt i lista over requirements

neste seksjon blir noe sånt som proposed design, og forteller hva du har tenkt å gjøre for å oppfylle kravene. bruk referanser tilbake til de subsections som beskriver kravene. noe som "To fulfil the XX requirement described in section \ref{YY} we wil ..."
May 26, 2023 6:28 PM

einar.j.holsbo: Så vil antakelig dette med model architectures falle inn i ett av disse, antakelig det med training/inference?
""


"""