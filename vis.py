import os
import tensorflow as tf
import glob
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# %matplotlib inline


def get_section_results(file, tags):
    """
        requires tensorflow==1.12.0
    """
    data_dict = {tag: [] for tag in tags}
    # for e in tf.compat.v1.train.summary_iterator(file):
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
#             print(v.tag)
            if v.tag in data_dict:
                data_dict[v.tag].append(v.simple_value)
    data_dict = {tag: np.array(data_dict[tag]) for tag in data_dict}
    return data_dict


def log_rewards(file_name_list, tags):
    data_tag_dict = {}
    for tag in tags:
        data_tag_dict[tag] = []
    
    for file_name in file_name_list:
        data = log_data(file_name, tags)
        for tag in tags:
            data_tag_dict[tag].append(data[tag])
        
    return data_tag_dict

def log_data(file_name, tags):
    logdir = 'data/*' + file_name + '*/events*'
    print('search file name: ', logdir)
    eventfile = glob.glob(logdir)[0]
    print('found file name: ', eventfile)

    data_dict = get_section_results(eventfile, tags)
    return data_dict


def set_plot_env(iterations, rewards_dict, exp_name, curve_names=None):

    plt.figure(figsize=(10,5))
    style = "whitegrid"
    sns.set_theme(style=style) # background color
    ax = plt.gca()
    
    color_list = ['b', 'r', 'y', 'g', 'm', 'k', 'b-.', 'r-.', 'y-.']

    for idx, name in enumerate(rewards_dict):       
        curve_name = curve_names[idx] if curve_names is not None else name    
        plot_reward(ax, iterations, rewards_dict[name], curve_name, color=color_list[idx])

    # ax.legend(loc='center right')
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.legend()
    ax.set_xlabel('Time steps')
    ax.set_ylabel('eval average return')
    ax.set_title(exp_name +' experiment')

    exp_dir = 'plots/'
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    plt.savefig(fname=exp_dir + exp_name + '.png', format='png', dpi=1000,  bbox_inches='tight')

    # plt.show()

    
def plot_reward(ax, iterations, rewards, name, color):
    rewards = np.array(rewards)
    min_len = min(len(iterations), len(rewards))
#     iterations = np.array(iterations).T
#     rewards = np.array(rewards).T

    ax.plot(iterations[:min_len], rewards[:min_len], color=color, label=name)


def main():
    tag_space = ['Train_EnvstepsSoFar', 'Exploitation_Data_q-values', 'Train_AverageReturn', 'Eval_AverageReturn']

    file_name_a = ['todo']
    data_dict_a = log_rewards(file_name_a, tag_space)

    iterations = data_dict_a['Train_EnvstepsSoFar'][0]
    rewards_dict_a = data_dict_a['Eval_AverageReturn']


    rewards_dict = {}
    for idx in range(len(file_name_a)):
        rewards_dict[file_name_a[idx]] = rewards_dict_a[idx]

    set_plot_env(iterations, rewards_dict, exp_name='q2 part3')

if __name__ == "__main__":
    main()