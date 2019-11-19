#Ernesto Martinez
#CS441 PDX
#Heart Anomalies HW
#Naive Bayessian

import numpy as np 
import sys

# To a better understanding of Naive bayes I used:
# https://www.geeksforgeeks.org/naive-bayes-classifiers/
# Key concepts in probabilities() and classifier() are based on this website.

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
def fileP(features):
    hearts = len(features) + 0.5

    # count_nonzero is used due to all 1's been in the beggining of the file
    # and the 0's at the end
    n = np.count_nonzero(features) + 0.5
    nP = n / float(hearts) #probability of normal

    ab = hearts - n + 0.5
    abP = ab / float(hearts) # Probability of abnormal

    return n, nP, ab, abP, hearts

# Looping through features
#output : determine probability for each feature
def looping(naList, data, n, ab):

    def probabilities(intances, features, n, ab):

        #initialize
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html
        # the 0.5 is due to log of 0 undefined. Explained in fileP()
        # np.where Returns elements chosen from x or y depending on condition
        # *** Example for a random feature ***
        # Suppose there are 200 hearts, 120 normal
        #         abnormal normal
        #             0     1     P(0)     P(1)
        # feature=1  30    100    30/80    100/120
        # feature=0  50    20     50/80    20/120
        # total      80   120     80/80    120/120

        # 30/80: represents probability of feature being 1 given that heart is abnormal
        # 50/80: represents probability of feature being 0 given that heart is abnormal
        # 100/120: represents probability of feature being 1 given that heart is normal
        # 20/120: represents probability of feature being 1 given that heart is normal

        # *****************************
        # --------------
        # abnormal = 0 & feature = 1   
        ab1 = len(np.where((intances==0) & (features==1))[0]) 
        ab1 += 0.5 
        # normal = 1 & feature = 0
        n1 = len(np.where((intances==1) & (features==1))[0]) 
        n1 += 0.5
         # abnormal = 0 & feature = 1
        ab0 = ab - ab1 + .5                                
        # normal = 1 & feature = 1
        n0 = n - n1 + .5     

        # arrays of normal and abnormal probabilities
        # create the arrays and fill them with zeros to later use of those spaces
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.zeros.html
        normal = np.zeros(2, dtype= float)
        abnormal = np.zeros(2, dtype= float)
        # calculate and store probabilities.
        normal[0] = ab0/float(ab)
        normal[1] = ab1/float(ab)
        abnormal[0] = n0/float(n)
        abnormal[1] = n1/float(n)  

        #log to make them smaller
        # this is not really necessary or affects the
        # final outcome, but it definetely simplifies
        # the work. 
        # https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.log2.html
        logs = np.log2( [normal, abnormal])

        return logs         

    # goes through the 22 features ( in orig ) or all feautures depending on file
    # append output of probalitites of that feature to array of probabilities
    for i in range(1, (len(data[0]))): 
        naList.append( probabilities(data[:,0], data[:,i], n, ab) )
    

    return naList

def classifier(dataTest, naList):

    learner = []
    dataL = len(dataTest)

    for i in range(dataL):
        logN = 0 
        logAB = 0    

        for j in range(1, (len(dataTest[0]))):
            x = dataTest[i][j]
            logAB += naList[j-1][0][x]
            logN += naList[j-1][1][x]
        
        if logN > logAB:
            learner.append(1)
        else:
            learner.append(0)

    return learner
    
def main():

    # Take the input
    fileTrain, fileTest = inputUser(sys.argv)

    # Read data and length from training file
    data, dataLen = readFile(fileTrain)

    # Read data and length from test file
    dataTest, dataTestLen = readFile(fileTest)

    # total of normal and abnormal heart + Probabilities of each given data
    n, nP, ab, abP, hearts = fileP(data[:,0])

    # determines probability for each feature - EXPLAIN
    #1D array of probabilities for normal or abnormal hearts
    naList = []
    # fill array
    naList = looping(naList, data, n, ab)

    # EXPLAIN
    learner = classifier(dataTest, naList)

    #EXPLAIN Calculate and Merge Accuracy
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.equal.html
    calculate = np.equal(dataTest[:,0], learner)
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.sum.html
    accuracy = [np.sum(calculate), len(dataTest), np.sum(calculate)/ float(len(dataTest))]

    # OUTPUT TO FILE ---------------------------------------------
    filename = sys.argv[2] + ".txt" #name of file ending in txt
    f = open(filename, "w") # create file
    accuracy[2] = format(accuracy[2], '.2g') #floating point arithmetic

    f.writelines("Accuracy: " + str(accuracy[0]) + "/" + str(accuracy[1]) + "(" + str(accuracy[2]) + ")")
    f.writelines("\nTrue Positive: ")
    f.writelines("\nTrue Negative: ") 
    f.close()
    # ------------------------------------------------------------
    #-----------------------------------------------------------
    print(accuracy)
    #-----------------------------------------------------------

main()
