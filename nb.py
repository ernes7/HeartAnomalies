#Ernesto Martinez
#CS441 PDX
#Heart Anomalies HW
#Naive Bayessian

import numpy as np 
import sys

# To a better understanding of Naive bayes I used:
# https://www.geeksforgeeks.org/naive-bayes-classifiers/

# Other references
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.loadtxt.html


#Function: inputUser
#Input : user commands
#Output : name of files
def inputUser(commands):

    if(len(sys.argv) >= 2):
        fileTrain = commands[1]
        fileTest = commands[2]

    else:
        print("Provide 2 files for train and test")
        exit()
    
    if(fileTrain.endswith(".train.csv") and fileTest.endswith(".test.csv")):
        return fileTrain, fileTest
    else:
        print("Provide 2 files ending in train.csv and test.csv")
        exit()

#Function: readFile
#Input : file
#Output : data into array, lenght of data
def readFile(dFile):

    #loads data from text file, where every row must have same # of values.
    data = np.loadtxt(dFile, delimiter=",", dtype=int)

    dataLen = len(data)

    return data, dataLen

#fileP = probabilities of abnormal and normal hearts inside file
#input = data
#output = total of normal and abnormal heart + Probabilities of each.
# note : Because the log of 0 is undefined, it would probably be a 
# good idea not to go there, so we usem-estimation byadding an arbitrary 0.5 to the 
# numerator and denominator counts of each probability.
def fileP(data):
    hearts = len(data)

    # count_nonzero is used due to all 1's been in the beggining of the file
    # and the 0's at the end
    n = np.count_nonzero(data) + 0.5
    nP = n / float(hearts)

    ab = hearts - n
    abP = ab / float(hearts)

    return n, nP, ab, abP, hearts
    

def main():

    # Take the input
    fileTrain, fileTest = inputUser(sys.argv)

    #1D array of probabilities for normal or abnormal hearts
    naList = []

    # Read data and length from training file
    data, dataLen = readFile(fileTrain)

    # Read data and length from test file
    dataTest, dataTestLen = readFile(fileTest)

    # total of normal and abnormal heart + Probabilities of each given data
    n, nP, ab, abP, hearts = fileP(data[:,0])
 

    #-----------------------------------------------------------
    print(fileTrain)
    print(fileTest)

    print(data)
    print(dataLen)
    print(dataTest)
    print(dataTestLen)

    print(n,nP,ab,abP,hearts)
    #-----------------------------------------------------------

main()
