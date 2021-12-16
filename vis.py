import os
import tensorflow as tf
import glob
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# %matplotlib inline


def log_rewards(file_name_list):

    file_name = file_name_list[0]
    data = log_data(file_name)
    # data_tag_dict[tag].append(data[tag])
        
    return data


def log_data(file_name):
    logdir = 'data/*' + file_name + '*/events*'
    print('search file name: ', logdir)
    eventfile = glob.glob(logdir)[0]
    print('found file name: ', eventfile)

    data_dict = get_section_results(eventfile)
    return data_dict


def get_section_results(file):
    """
        requires tensorflow==1.12.0
    """
    data_dict = {}

    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            # print(v.tag)
            if v.tag not in data_dict:
                data_dict[v.tag] = []

            data_dict[v.tag].append(v.simple_value)
            
    # data_dict = {tag: np.array(data_dict[tag]) for tag in data_dict}
    return data_dict


def set_plot_env(iterations, rewards_dict, exp_name, curve_names=None):

    plt.figure(figsize=(10,5))
    style = "whitegrid"
    sns.set_theme(style=style) # background color
    ax = plt.gca()
    
    color_list = ['b', 'r', 'y', 'g', 'm', 'k', 'b-.', 'r-.', 'y-.']

    for idx, name in enumerate(rewards_dict):       
        curve_name = curve_names[idx] if curve_names is not None else name    
        plot_reward(ax, iterations, rewards_dict[name][0], curve_name, color=color_list[idx])
        plot_std(ax, iterations, rewards_dict[name][0], rewards_dict[name][1], color=color_list[idx])

    # ax.legend(loc='center right')
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.legend(fontsize=15)
    ax.set_xlabel('Iterations',fontsize=15)
    ax.set_ylabel(exp_name, fontsize=15)
    ax.set_title(exp_name, fontsize=15)

    exp_dir = 'plots/'
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    plt.savefig(fname=exp_dir + exp_name + '.png', format='png', dpi=300,  bbox_inches='tight')

    # plt.show()

    
def plot_reward(ax, iterations, rewards, name, color):
    rewards = np.array(rewards)
    min_len = min(len(iterations), len(rewards))
    ax.plot(iterations[:min_len], rewards[:min_len], color=color, label=name)

def plot_std(ax, iterations, mean, std, color):
    up = np.array(mean) + np.array(std)
    dn = np.array(mean) - np.array(std)

    min_len = min(len(iterations), len(mean))
    ax.fill_between(iterations, dn, up, alpha=0.2)


def main():

    file_name_a = ['stage2']
    
    data_dict_a = log_rewards(file_name_a)

    # print('data_dict_a ', data_dict_a.keys())

    iterations = data_dict_a['itr']

    rewards_dict = { 'Eval_AverageReturn': (data_dict_a['Eval_AverageReturn'], data_dict_a['Eval_StdReturn']),
                    'Train_AverageReturn': (data_dict_a['Train_AverageReturn'], data_dict_a['Train_StdReturn'])}
    set_plot_env(iterations, rewards_dict, exp_name= file_name_a[0] + ' reward')

    food_dict = {'Eval_AverageFood': (data_dict_a['Eval_AverageFood'], data_dict_a['Eval_StdFood']),
                 'Train_AverageFood' : (data_dict_a['Train_AverageFood'], data_dict_a['Train_StdFood'])    }

    set_plot_env(iterations, food_dict, exp_name= file_name_a[0] + ' food')


    ep_len_dist = {'Eval_AverageEpLen' : (data_dict_a['Eval_AverageEpLen'], data_dict_a['Eval_StdEpLen']),
                 'Train_AverageEpLen': (data_dict_a['Train_AverageEpLen'], data_dict_a['Train_StdEpLen'])}

    set_plot_env(iterations, ep_len_dist, exp_name= file_name_a[0] + ' epoch_length')

if __name__ == "__main__":
    main()