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

#fileP = probabilities of abnormal and normal hearts inside file
#input = data
#output = total of normal and abnormal heart
# note : Because the log of 0 is undefined, it would probably be a 
# good idea not to go there, so we usem-estimation byadding an arbitrary 0.5 to the 
# numerator and denominator counts of each probability.
def fileP(features):
    hearts = len(features) + 0.5

    # count_nonzero is used due to all 1's been in the beggining of the file
    # and the 0's at the end
    n = np.count_nonzero(features) + 0.5
    ab = hearts - n + 0.5

    return n, ab

# Looping through features
#output : determine probability for each feature
def learnerFunction(naList, data, n, ab):

    def probabilities(intances, features, n, ab):

        # Feature Probabilities Function
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html
        # the +0.5 is due to log of 0 undefined. Explained in fileP()
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

        # log to make them smaller
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

    # Logic Behind the classifier
    # --------------------------
    #  P(abnormal)               P(Feature1)
    #  80/200  +  30/80(Feature = 1) OR 50/80(Feature=0) + ... (all feautures)
    #
    #  P(normal)                 P(Feature1)
    #  120/200 +  100/120(Feature = 1) OR 20/120(Feature = 0) + ... (all features)

    learner = [] # array of probabilities learned from every heart
    dataL = len(dataTest)

    # goes through all hearts
    for i in range(dataL):
        # set back to 0 every time that switch to new heart
        n = 0 
        ab = 0    

        # goes through all the feautures in that heart
        for j in range(1, (len(dataTest[0]))):
            x = dataTest[i][j]
            # this calculates the probability of a heart to be abnormal
            ab += naList[j-1][0][x]
            # this calculates the probability of a heart to be normal
            n += naList[j-1][1][x]
        
        # if normal > abnormal, a "1" gets added to the learner
        if n > ab:
            learner.append(1)
        # "0" is added otherwise
        else:
            learner.append(0)

    return learner

def main():

    # Take the input
    fileTrain, fileTest = inputUser(sys.argv)

    # Read data and length from training file
    data = np.loadtxt(fileTrain, delimiter=",", dtype=int)

    # Read data and length from test file
    dataTest = np.loadtxt(fileTest, delimiter=",", dtype=int)

    # total of normal and abnormal heart
    n, ab = fileP(data[:,0])

    # determines probability for each feature 
    #array of probabilities for normal or abnormal hearts
    naList = []
    # fill array
    naList = learnerFunction(naList, data, n, ab)

    # Creates the learner array based on the probabilities from the learnerFunction
    learner = classifier(dataTest, naList)

    # Calculate and Merge Accuracy
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.equal.html
    # If the heart #1 at dataTest and Learner are the same, 
    # returns True at that index in array named calculate.
    # This is used to test the learner against the test data. 
    # The result of this is used to output the accuracy
    calculate = np.equal(dataTest[:,0], learner)
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.sum.html
    accuracy = [np.sum(calculate)-1, len(dataTest), np.sum(calculate)/ float(len(dataTest))]


    # True Positive and Negative Calculation
    #https://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html
    #https://docs.scipy.org/doc/numpy/reference/generated/numpy.count_nonzero.html
    #https://docs.scipy.org/doc/numpy/reference/generated/numpy.asarray.html
    # TRUE POSITIVE ---------------------------------------
    same = np.where((dataTest[:,0] == 1) & (np.asarray(learner) == 1)) 
    count = np.count_nonzero(dataTest[:,0] == 1) # total number of respective heart(normal or abnormal)
    truePositive = [ len(same[0])-1, count, len(same[0])/float(count)] # format

    # TRUE NEGATIVE ----------------------------------------
    x = accuracy[0]-truePositive[0]
    y = accuracy[1]-truePositive[1]
    trueNegative = [ x, y, x/y]
    #trueNegative = true_pn(0,learner, dataTest[:0])


    # OUTPUT TO FILE ---------------------------------------------
    filename = sys.argv[2] + ".txt" #name of file ending in txt
    f = open(filename, "w") # create file
    accuracy[2] = format(accuracy[2], '.2g') #floating point arithmetic
    trueNegative[2] = format(trueNegative[2], '.2g')
    truePositive[2] = format(truePositive[2], '.2g')

    f.writelines("Accuracy: " + str(accuracy[0]) + "/" + str(accuracy[1]) + "(" + str(accuracy[2]) + ")")
    f.writelines("\nTrue Negative: " + str(trueNegative[0]) + "/" + str(trueNegative[1]) + "(" + str(trueNegative[2]) + ")")
    f.writelines("\nTrue Positive: " + str(truePositive[0]) + "/" + str(truePositive[1]) + "(" + str(truePositive[2]) + ")") 
    f.close()
    # ------------------------------------------------------------
    #-----------------------------------------------------------
    print("Accuracy: ",accuracy)
    print("True Positive: ",truePositive)
    print("True Negative: " ,trueNegative)
    #-----------------------------------------------------------

main()
