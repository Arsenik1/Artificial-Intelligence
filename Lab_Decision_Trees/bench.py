import time
import os
import pdb
import numpy as np
import matplotlib.pyplot as plt
import dec_tree_func as dt
import math
import decimal

#os.system('cls')

#file_name = "lenses\lenses.txt"
#file_name = 'car/car.txt'
#file_name = 'tic/tic.txt'
#file_name = 'letters1.txt'
file_name = 'D:\Desktop\Salih\Belgeler\VSCCode\Python\Artificial-Intelligence\Lab_Decision_Trees\letters1.txt'
#file_name = 'letters3.txt'
#file_name = 'letters5.txt'

number_of_experiments = 10

# part of the examples for building a tree in percentage (0-100%) other axamples can be used 
# for pruning as validation subset:
part_for_constr = 100 # (..... you can choose the proportions) 



print("\n\n\nbuilding tree for " + file_name)
# loading data from file and dividing it into training and test sets:
# examples with odd numbers (1,3,5,...) for training
# examples with even numbers for test

examples = dt.load_data(file_name) 
#print(" examples = \n" + str(examples))

[num_of_examples, num_of_columns] = examples.shape
num_of_training_examples = num_of_examples//2
num_of_test_examples = num_of_examples - num_of_training_examples
number_for_constr = math.ceil(num_of_training_examples*part_for_constr/100)   # number of examples for tree construction 
#print("num_of_training_examples = " + str(num_of_training_examples)+ " num_of_test_examples = "+str(num_of_test_examples))

mean_test_error = 0
mean_test_error_prun = 0



for eksp in range(number_of_experiments):

    np.random.shuffle(examples)       # random permutation of example set
    #print(" examples after shuffle = \n" + str(examples))

    train_vectors = examples[0:num_of_training_examples,0:num_of_columns-1]
    train_classes = examples[0:num_of_training_examples,num_of_columns-1]
    test_vectors = examples[num_of_training_examples:num_of_examples,0:num_of_columns-1]
    test_classes = examples[num_of_training_examples:num_of_examples,num_of_columns-1]

    #print("train_vectors = \n" + str(train_vectors))
    #print("train_classes = \n" + str(train_classes))
    #print("test_vectors = \n" + str(test_vectors))
    #print("test_classes = \n" + str(test_classes))

    tree = dt.build_tree(train_vectors, train_classes)

    #print("final tree = \n" + str(tree))

    dist = dt.distribution(train_vectors,train_classes,tree)
    #print("distribution = \n" + str(dist))
    d =  dt.depth(tree)
    #print("depth = " + str(d))

    num_of_training_errors = dt.calc_error(train_vectors,train_classes,tree)
    num_of_test_errors = dt.calc_error(test_vectors,test_classes,tree)

    mean_test_error += num_of_test_errors/num_of_test_examples/number_of_experiments 

    pruned_tree = np.copy(tree)

    # pruning:
    # a place for your algorithm here!
    # ...............................
    # ...............................
    # ...............................
    # ...............................


    
    [num_of_rows, num_of_nodes] = pruned_tree.shape

    num_of_training_errors_prun = dt.calc_error(train_vectors,train_classes,pruned_tree)
    num_of_test_errors_prun = dt.calc_error(test_vectors,test_classes,pruned_tree)

    print("error: train = " + str(num_of_training_errors/num_of_training_examples) + " test = " + 
    str(num_of_test_errors/num_of_test_examples) + "  after pruning: train = " + 
    str(num_of_training_errors_prun/num_of_training_examples) + " test = " + str(num_of_test_errors_prun/num_of_test_examples))

    mean_test_error_prun += num_of_test_errors_prun/num_of_test_examples/number_of_experiments 

print("mean test error: before pruning = " + str(mean_test_error) + " after pruning = " + str(mean_test_error_prun) +
" reduction of average test error = " + str(100*(mean_test_error-mean_test_error_prun)/mean_test_error) + "%")


