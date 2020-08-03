import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.axes as axes
import numpy as np
from collections import Counter
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()


def least_common_digit(x_set, y_set):
    '''
       Input: x_set, the x values of the dataset and y_set, the y values of the  dataset
       Expected Output: The image from the x set of the least common digit.
    '''
    
    digits_count={x:0 for x in range(10)}
    for n in range(len(y_set)):
        for i in digits_count:
            if y_set[n]==i:
                digits_count[i]+=1
                break
    
    minimum=digits_count[0]
    least_common=0
    
    for n in range(1,10):
        if digits_count[n]<minimum:
            minimum=digits_count[n]
            least_common=n
    
    for n in range(len(y_set)):
        if n==least_common:
            return x_set[n]
    
lc_train = least_common_digit(x_train, y_train)
lc_test = least_common_digit(x_test, y_test)


def most_common_digit(x_set, y_set):
    '''
       Input: x_set, the x values of the dataset and y_set, the y values of the  dataset
       Expected Output: The image from the x set of the most common digit.
    '''
    
    digits_count={x:0 for x in range(10)}
    for n in range(len(y_set)):
        for i in digits_count:
            if y_set[n]==i:
                digits_count[i]+=1
                break
    
    maximum=digits_count[0]
    most_common=0
    
    for n in range(1,10):
        if digits_count[n]>maximum:
            maximum=digits_count[n]
            most_common=n
    
    for n in range(len(y_set)):
        if n==most_common:
            return x_set[n]

mc_train = most_common_digit(x_train, y_train)
mc_test = most_common_digit(x_test, y_test)


def plot_two(im1, title1, im2, title2):
    '''
        Input: im1, a matrix representing a grayscale image and title1 a string,im2 a matrix representing
        a grayscale image and title2 a string
        Expected Output: A tuple (fig, ax) representing a generated figure from matplotlib and two subplots
        ready to display the inputed images with the given titles
    '''

    fig,ax = plt.subplots(1, 2)
    ax[0].imshow(im1, cmap='gray')
    ax[0].set_title(title1)
    ax[1].imshow(im2, cmap='gray')
    ax[1].set_title(title2)
    fig.tight_layout()

    return fig,ax

plot_two(lc_train, 'Least Common Train', lc_test, 'Least Common Test'), plot_two(mc_train, 'Most Common Train', mc_test, 'Most Common Test')
plt.show()


def how_many_of_each_digit(y_set):
    '''
       Input: y_set, the y values of the training set
       Expected Output: A dict of the count of each digit in the set
    '''
    
    digits_count={x:0 for x in range(10)}
    for n in range(len(y_set)):
        for i in digits_count:
            if y_set[n]==i:
                digits_count[i]+=1
                break
                
    return digits_count

count_train = how_many_of_each_digit(y_train)
count_test = how_many_of_each_digit(y_test)


def bar_chart(train, test):
    '''
    Inputs: train, a dictionary of count of each digit of the training set and test, a dictionary of the count
    of each digit for the test set
    Expected Output: A tuple (fig, ax) ready to show using matplotlib
    '''
    
    columns=10
    test_list=[]
    train_list=[]
    for n in range(10):
        test_list.append(test[n])
    
    for n in range(10):
        train_list.append(train[n])
    
    ind = np.arange(columns)
    width = 0.5

    p1 = plt.bar(ind, test_list, width)
    p2 = plt.bar(ind, train_list, width,
                 bottom=test_list)

    plt.xticks(np.arange(0, 11, 2))
    plt.yticks(np.arange(0, 7100, 1000))
    plt.legend((p1[0], p2[0]), ('Test set', 'Training set'))

    plt.show()
    
bar_chart(count_train,count_test)


def interesting_visualization(test):

    labels = '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
    test_occurrences=[]
    for n in range(10):
        test_occurrences.append(test[n])

    fig, axe = plt.subplots()
    axe.pie(test_occurrences, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    axe.axis('equal')
    plt.title('Distribution of Digits in Test Set')

    plt.show()

interesting_visualization(count_test)
