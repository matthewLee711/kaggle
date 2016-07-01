import csv
import numpy as np

csv_file_object = csv.reader(open('train.csv', 'rb'))
header = csv_file_object.next()#skips first line

data=[]
for row in csv_file_object:
    data.append(row)#add each row to data object
data = np.array(data)#convert list to array

#csv reader works with strings by default
#need to convert to float

#Finds number of survivors
number_passengers = np.size(data[0::,1].astype(np.float))
number_survived = np.sum(data[0::,1].astype(np.float))
proportion_survivors = number_survived / number_passengers

#Find all genders


#finds where all elements in geneder column equals female
women_only_stats = data[0::,4] == "female"
men_only_stats = data[0::,4] != "female"

#select data of women and men who survived 0-dead 1-alive
women_onboard = data[women_only_stats,1].astype(np.float)
men_onboard = data[men_only_stats,1].astype(np.float)

#Create training set of survivors
proportion_women_survived = np.sum(women_onboard) / np.size(women_onboard)
proportion_men_survived = np.sum(men_onboard) / np.size(men_onboard)

print 'Proportion of women who survived is %s' % proportion_women_survived
print 'Proportion of men who survived is %s' % proportion_men_survived

#Extract information from test.csv
test_file = open('test.csv', 'rb')
test_file_object = csv.reader(test_file)
header = test_file_object.next()

#Create write to file
prediction_file = open("genderbasedmodel.csv", "wb")
prediction_file_object = csv.writer(prediction_file)

#Write survival prediction to new file
prediction_file_object.writerow(["PassengerId", "survived"])
for row in test_file_object:
    if row[3] == 'female':
        prediction_file_object.writerow([row[0],'1']) #predict 1
    else:
        prediction_file_object.writerow([row[0],'0']) #predict 0

test_file.close()
prediction_file.close()
