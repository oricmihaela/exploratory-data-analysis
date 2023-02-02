import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import statistics

from mat4py import loadmat
from mlxtend.plotting import scatter_hist


BASE_PATH = os.getcwd()


#   All body variables except gender
body_variables = [
    'Biacromial diameter', 'Shoulder girth over deltoid muscles', 
    'Chest girth', 'Waist girth', 'Navel girth', 'Hip girth', 
    'Thigh girth', 'Bicep girth', 'Forearm girth', 'Knee girth', 
    'Calf maximum girth', 'Ankle minimum girth', 'Wrist minimum girth', 
    'Age', 'Weight', 'Height']


#   Load mat file into a dictionary
bodydata_dict = loadmat("bodydata.mat")

#   Dictionary to list
bodydata_list = list(bodydata_dict.items())

#   First element is 'bodydat', second is data
#   Convert to array
bodydata = np.array(bodydata_list[0][1])

#   Find all places where gender is 0 for females and 1 for males
female_indexes = np.where(bodydata[:, 16] == 0)
male_indexes = np.where(bodydata[:, 16] == 1)

#   Squeeze removes all dimensions of size 1
female_bodydata = np.squeeze(bodydata[female_indexes, :])
male_bodydata = np.squeeze(bodydata[male_indexes, :])

#   Check the split
if(female_bodydata.shape[0] + male_bodydata.shape[0] == bodydata.shape[0]):
    print("Dataset correctly split.")
else:
    print("ERROR - Check the dataset split!")
    sys.exit()


def task_1():

    #   Make task directory if it doesn't exist
    if(os.path.exists('task_1') == False):
        os.mkdir(os.path.join(BASE_PATH, 'task_1'))

    female_statistics = np.zeros((len(body_variables), 3))
    male_statistics = np.zeros((len(body_variables), 3))

    for i in range(len(body_variables)):

        female_statistics[i][0] = statistics.mean(female_bodydata[:, i])
        female_statistics[i][1] = statistics.median(female_bodydata[:, i])
        female_statistics[i][2] = statistics.stdev(female_bodydata[:, i])


        male_statistics[i][0] = statistics.mean(male_bodydata[:, i])
        male_statistics[i][1] = statistics.median(male_bodydata[:, i])
        male_statistics[i][2] = statistics.stdev(male_bodydata[:, i])

        info = '# mean, median, standard deviation\n'

        female_file = open(os.path.join(BASE_PATH, 'task_1', 'female_statistics.txt'), 'w')
        female_file.write(info + str(female_statistics))
        female_file.close()

        male_file = open(os.path.join(BASE_PATH, 'task_1', 'male_statistics.txt'), 'w')
        male_file.write(info + str(male_statistics))
        male_file.close()


def task_2():

    #   Make task directory if it doesn't exist
    if(os.path.exists('task_2') == False):
        os.mkdir(os.path.join(BASE_PATH, 'task_2'))

    for i in range(len(body_variables)):

        figure = plt.figure()
        plt.hist(female_bodydata[:, i], 10, color='red', alpha=0.5, label='female')
        plt.hist(male_bodydata[:, i], 10, color='blue', alpha=0.5, label='male')
        plt.legend(loc='upper right')
        plt.title(body_variables[i])
        plt.grid(True)
        plt.savefig('task_2/' + body_variables[i] + '.png')


def task_3():

    #   Make task directory if it doesn't exist
    if(os.path.exists('task_3') == False):
        os.mkdir(os.path.join(BASE_PATH, 'task_3'))

    for i in range(len(body_variables)):

        female_count, female_bins_count = np.histogram(female_bodydata[:, i], bins=10)
        female_pdf = female_count/sum(female_count)
        female_cdf = np.cumsum(female_pdf)

        male_count, male_bins_count = np.histogram(male_bodydata[:, i], bins=10)
        male_pdf = male_count/sum(male_count)
        male_cdf = np.cumsum(male_pdf)

        female_first_quartile = np.percentile(female_bodydata[:, i], 25)  # Q1
        female_third_quartile = np.percentile(female_bodydata[:, i], 75)  # Q3

        male_first_quartile = np.percentile(male_bodydata[:, i], 25)  # Q1
        male_third_quartile = np.percentile(male_bodydata[:, i], 75)  # Q3

        figure = plt.figure()
        plt.plot(female_bins_count[1:], female_cdf, label='female', color='red')
        plt.plot(male_bins_count[1:], male_cdf, label='male', color='blue')
        plt.grid(True, axis='y')
        plt.axvline(x=male_first_quartile, label='male Q1', color='blue', alpha=0.2, linestyle='dashed')
        plt.axvline(x=male_third_quartile, label='male Q3', color='blue', alpha=0.5, linestyle='dashed')
        plt.axvline(x=female_first_quartile, label='female Q1', color='red', alpha=0.2, linestyle='dashed')
        plt.axvline(x=female_third_quartile, label='female Q3', color='red', alpha=0.5, linestyle='dashed')
        plt.legend()
        plt.title(body_variables[i])
        plt.savefig('task_3/' + body_variables[i] + '.png')


def task_4():

    #   Make task directory if it doesn't exist
    if(os.path.exists('task_4') == False):
        os.mkdir(os.path.join(BASE_PATH, 'task_4'))

    for i in range(len(body_variables)):

        figure = plt.figure()
        plt.boxplot([female_bodydata[:, i], male_bodydata[:, i]], labels=['female', 'male'])
        plt.grid(True, axis='y')
        plt.title(body_variables[i])
        plt.savefig('task_4/' + body_variables[i] + '.png')


def task_5():

    #   Make task directory if it doesn't exist
    if(os.path.exists('task_5') == False):
        os.mkdir(os.path.join(BASE_PATH, 'task_5'))

    #   Weight - 14
    #   Height - 15

    wanted_body_variables = [0, 2, 3, 4]

    for body_variable in wanted_body_variables:
        
        y_female = female_bodydata[:, body_variable]
        y_male = male_bodydata[:, body_variable]

        #   Weight
        x_female = female_bodydata[:, 14]
        fig = scatter_hist(x_female, y_female, xlabel='weight', ylabel=body_variables[body_variable])
        plt.savefig('task_5/weight/female_' + body_variables[body_variable] + '.png')

        x_male = male_bodydata[:, 14]
        fig = scatter_hist(x_male, y_male, xlabel='weight', ylabel=body_variables[body_variable])
        plt.savefig('task_5/weight/male_' + body_variables[body_variable] + '.png')

        #   Height
        x_female = female_bodydata[:, 15]
        fig = scatter_hist(x_female, y_female, xlabel='height', ylabel=body_variables[body_variable])
        plt.savefig('task_5/height/female_' + body_variables[body_variable] + '.png')

        x_male = male_bodydata[:, 15]
        fig = scatter_hist(x_male, y_male, xlabel='height', ylabel=body_variables[body_variable])
        plt.savefig('task_5/height/male_' + body_variables[body_variable] + '.png')

        
if __name__=='__main__':
    
    # task_1()
    # task_2()
    task_3()
    # task_4()
    # task_5()
