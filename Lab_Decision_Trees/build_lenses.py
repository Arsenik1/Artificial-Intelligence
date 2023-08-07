import time
import os
import pdb
import numpy as np
import matplotlib.pyplot as plt
import dec_tree_func as dt

os.system('cls')

file_name = "D:\Desktop\Salih\Belgeler\VSCCode\Python\Artificial-Intelligence\Lab_Decision_Trees\lenses\lenses.txt"
#file_name = "car/car.txt"

print("\n\n\nbuilding tree for " + file_name)
# loading data from file and dividing it into training and test sets:
# examples with odd numbers (1,3,5,...) for training
# examples with even numbers for test
train_vectors, train_classes, test_vectors, test_classes = dt.load_data_odd_even(file_name)

num_of_train_examples = train_classes.__len__()
num_of_test_examples = test_classes.__len__()
print("number of examples: training = " + str(num_of_train_examples) + " test = " + str(num_of_test_examples))

tree = dt.build_tree(train_vectors, train_classes)

print("final tree = \n" + str(tree))

dist = dt.distribution(train_vectors,train_classes,tree)
print("distribution = \n" + str(dist))

d =  dt.depth(tree)
print("depth = " + str(d))

num_of_training_errors = dt.calc_error(train_vectors,train_classes,tree)
num_of_test_errors = dt.calc_error(test_vectors,test_classes,tree)

print("training error = " + str(num_of_training_errors/num_of_train_examples) + " test error = " + 
str(num_of_test_errors/num_of_test_examples))

# pruning:

pruned_tree = np.copy(tree)
[num_of_rows, num_of_nodes] = pruned_tree.shape
for i in range(num_of_rows):
    pruned_tree[i][4] = 0
pruned_tree[num_of_rows-1][4] = 3   # simple changing node 4 to the leaf of class 3 
num_of_training_errors = dt.calc_error(train_vectors,train_classes,pruned_tree)
num_of_test_errors = dt.calc_error(test_vectors,test_classes,pruned_tree)

print("after pruning:\ntraining error = " + str(num_of_training_errors/num_of_train_examples) + " test error = " + 
str(num_of_test_errors/num_of_test_examples))




