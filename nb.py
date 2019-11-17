#Ernesto Martinez
#CS441 PDX
#Heart Anomalies HW
#Naive Bayessian

import numpy as np 
import sys

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

def main():

    # Take the input
    fileTrain, fileTest = inputUser(sys.argv)

    #1D array of probabilities for normal or abnormal hearts
    naList = []

    print(fileTrain)
    print(fileTest)

main()
