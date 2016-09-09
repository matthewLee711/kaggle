import csv as csv
import numpy as np
# Load in the csv file
# Skip the fist line as it is a header
# Create a variable to hold the data
csv_file_object = csv.reader(open('train.csv', 'rb'))
header = csv_file_object.next()
data=[]

for row in csv_file_object: # Skip through each row in the csv file
    data.append(row)        # adding each row to the data variable
data = np.array(data)       # Then convert from a list to an array

"""
Binning the ticket price into four bins and modeling
the outcome on class, gender, and ticket price
-----------------------------------------------------
Modify the data in the fare column to equal 39,
if it is greater than or equal to ceiling
"""
fare_ceiling = 40
data[ data[0::,9].astype(np.float) >= fare_ceiling, 9 ] = fare_ceiling - 1.0
fare_bracket_size = 10
number_of_price_brackets = fare_ceiling / fare_bracket_size

#Grab number of classes
number_of_classes = len(np.unique(data[0::,2]))

#Initialize the survival table with all zeros
survival_table = np.zeros((2, number_of_classes, number_of_price_brackets))

#Loop through each passenger and find passenger with those variables
for i in xrange(number_of_classes):             #each class
    for j in xrange(number_of_price_brackets):  #each price bins
        women_only_stats = data[                                    \
            (data[0::,4] == "female")                               \
            &(data[0::,2].astype(np.float) == i + 1)                \
            &(data[0:,9].astype(np.float) >= j*fare_bracket_size)   \
            &(data[0:,9].astype(np.float) < (j+1)*fare_bracket_size)\
            , 1
        ]
        men_only_stats = data[
            (data[0::,4] != "female")                               \
            &(data[0::,2].astype(np.float) == i + 1)                \
            &(data[0:,9].astype(np.float) >= j*fare_bracket_size)   \
            &(data[0:,9].astype(np.float) < (j+1)*fare_bracket_size)\
            , 1
        ]
        #Store export
        survival_table[0,i,j] = np.mean(women_only_stats.astype(np.float))
        survival_table[1,i,j] = np.mean(men_only_stats.astype(np.float))
        survival_table[ survival_table != survival_table ] = 0.

print survival_table
"""
[[[ 0. 0. 0.83333333 0.97727273]
  [ 0. 0.91428571 0.9 1. ]
  [ 0.59375 0.58139535 0.33333333 0.125 ]]

 [[ 0. 0. 0.4 0.38372093]
  [ 0. 0.15873016 0.16 0.21428571]
  [ 0.11153846 0.23684211 0.125 0.24 ]]]
"""

#Write to file
test_file = open('../csv/test.csv', 'rb')
test_file_object = csv.reader(test_file)
header = test_file_object.next()
predictions_file = open("../csv/genderclassmodel.csv", "wb")
p = csv.writer(predictions_file)
p.writerow(["PassengerId", "Survived"])
