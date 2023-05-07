import time
import os
import pdb
import numpy as np
import matplotlib.pyplot as plt
import dec_tree_func as dt

#os.system('cls')

#file_name = "lenses\lenses.txt"
#file_name = 'car/car.txt'
#file_name = 'tic/tic.txt'
#file_name = 'letters1.txt'
file_name = 'D:/Desktop/Salih/Belgeler/VSCCode/Python/AI/Lab_Decision_Trees/letters2.txt'
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
number_for_constr = np.ceil(num_of_training_examples*part_for_constr/100)   # number of examples for tree construction 
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

    # pruning:
    pruned_tree = np.copy(tree)
    [num_of_rows, num_of_nodes] = pruned_tree.shape
    for i in range(num_of_rows):
        if pruned_tree[i][1] == 0 and pruned_tree[i][2] == 0:
            # if this node is a leaf, try to remove it and see if it improves test error
            pruned_tree[i][4] = 0 # set the class of the leaf to 0 (unclassified)
            num_of_test_errors_prun = dt.calc_error(test_vectors,test_classes,pruned_tree)
            if num_of_test_errors_prun < num_of_test_errors:
                # removing the leaf improves test error, so remove it permanently
                pruned_tree[i][0] = 0
                pruned_tree[i][1] = 0
                pruned_tree[i][2] = 0
                pruned_tree[i][3] = 0
                pruned_tree[i][4] = np.argmax(dt.distribution(train_vectors,train_classes,pruned_tree)[i])
        elif pruned_tree[i][1] != 0 and pruned_tree[i][2] != 0:
            # if this node is not a leaf, try removing one of its child nodes and see if it improves test error
            left_child_index = int(pruned_tree[i][1])
            right_child_index = int(pruned_tree[i][2])
            if left_child_index >= 0 and left_child_index < num_of_rows and pruned_tree[left_child_index][1] == 0 and pruned_tree[left_child_index][2] == 0:
                # try removing the left child node
                pruned_tree[left_child_index][0] = 0
                pruned_tree[left_child_index][1] = 0
                pruned_tree[left_child_index][2] = 0
                pruned_tree[left_child_index][3] = 0
                pruned_tree[left_child_index][4] = np.argmax(dt.distribution(train_vectors,train_classes,pruned_tree)[left_child_index])
                num_of_test_errors_prun = dt.calc_error(test_vectors,test_classes,pruned_tree)
                if num_of_test_errors_prun < num_of_test_errors:
                    # removing the left child node improves test error, so remove it permanently
                    pruned_tree[i][1] = 0
            elif pruned_tree[right_child_index][1] == 0 and pruned_tree[right_child_index][2] == 0:
                # try removing the right child node
                pruned_tree[right_child_index][0] = 0
                pruned_tree[right_child_index][1] = 0
                pruned_tree[right_child_index][2] = 0
                pruned_tree[right_child_index][3] = 0
                pruned_tree[right_child_index][4] = np.argmax(dt.distribution(train_vectors,train_classes,pruned_tree)[right_child_index])
                num_of_test_errors_prun = dt.calc_error(test_vectors,test_classes,pruned_tree)
                if num_of_test_errors_prun < num_of_test_errors:
                    # removing the right child node improves test error, so remove it permanently
                    pruned_tree[i][2] = 0  


    
    [num_of_rows, num_of_nodes] = pruned_tree.shape

    num_of_training_errors_prun = dt.calc_error(train_vectors,train_classes,pruned_tree)
    num_of_test_errors_prun = dt.calc_error(test_vectors,test_classes,pruned_tree)

    print("error: train = " + str(num_of_training_errors/num_of_training_examples) + " test = " + 
    str(num_of_test_errors/num_of_test_examples) + "  after pruning: train = " + 
    str(num_of_training_errors_prun/num_of_training_examples) + " test = " + str(num_of_test_errors_prun/num_of_test_examples))

    mean_test_error_prun += num_of_test_errors_prun/num_of_test_examples/number_of_experiments 

print("mean test error: before pruning = " + str(mean_test_error) + " after pruning = " + str(mean_test_error_prun) +
" reduction of average test error = " + str(100*(mean_test_error-mean_test_error_prun)/mean_test_error) + "%")


