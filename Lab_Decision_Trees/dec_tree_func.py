import numpy as np
import matplotlib.pyplot as plt
import math

def majority_class(classes):
    """Return the majority class label for a set of examples."""
    class_counts = np.bincount(classes)
    majority_class = np.argmax(class_counts)
    return majority_class


def addnode(examples, classes, attribute_labels, number_of_values, number_of_nodes):
    [num_of_exampl, num_of_attr] = examples.shape
    num_of_classes = np.max(classes)
    if_messages = False

    if if_messages:
        print("num_of_classes = " + str(num_of_classes))

    tree = np.zeros([number_of_values+1,1], dtype=int)  # new node column

    # class distribution:
    class_distr = np.zeros([num_of_classes], dtype=int)
    for ex in range(num_of_exampl):
        class__ = classes[ex]
        class_distr[class__-1] += 1
    largest_class = np.argmax(class_distr)
    largest_class_cardinality = np.max(class_distr)
    if if_messages:
        print("largest class = " + str(largest_class+1) + " cardinality = " + str(largest_class_cardinality))    
    

    if (largest_class_cardinality == num_of_exampl)|(num_of_attr < 2): # one class only or contradiction
        # class number -> a leaf 
        if if_messages:
            print("leaf with class " + str(largest_class + 1))
        tree[number_of_values][0] = largest_class + 1
    else:
        # entropy calculation for all attributes:
        entropy_min = 10000
        attr_with_min_entropy = -1
        for atr in range(num_of_attr):
            sums_of_exampl = np.zeros([number_of_values, num_of_classes])   # array with numbers of examples for attr. value and class number 
            sums_of_ex_by_values = np.zeros([number_of_values]) # vector with numbers of examples with particular value of attr
            for ex in range(num_of_exampl):
                value = examples[ex][atr]
                class__ = classes[ex]
                sums_of_exampl[value-1][class__-1] += 1
                sums_of_ex_by_values[value-1] += 1

            entropy = 0
            for val in range(number_of_values):
                sum_for_val = 0
                for cl in range(num_of_classes):
                    if (sums_of_exampl[val][cl] > 0):
                        quotient = sums_of_exampl[val][cl]/sums_of_ex_by_values[val]
                        if quotient < 1:
                            sum_for_val += -quotient*math.log2(quotient)
                        else:
                            sum_for_val = 0
                entropy += sum_for_val*sums_of_ex_by_values[val]/num_of_exampl
            if if_messages:
                print("entropy for attr " + str(atr) + " = " + str(entropy))
            if entropy_min > entropy:
                entropy_min = entropy
                attr_with_min_entropy = atr
        if if_messages:
            print("minimum entropy = " + str(entropy_min) + " for attr index" + str(attr_with_min_entropy))

    
        # numer of attribute used to node creation:
        tree[number_of_values][0] = attribute_labels[attr_with_min_entropy]+1

        # atribute list reduction:
        new_attribute_labels = np.zeros([num_of_attr-1], dtype=int)
        index = 0
        for i in range(num_of_attr):
            if i != attr_with_min_entropy:
                new_attribute_labels[index] = attribute_labels[i]
                index += 1 
        if if_messages:
            print("new attribute labels = " + str(new_attribute_labels))


        # numbers of values calc.
        sums_of_ex_by_values = np.zeros([number_of_values]) # vector with numbers of examples with particular value of attr
        for ex in range(num_of_exampl):
            value = examples[ex][attr_with_min_entropy]
            sums_of_ex_by_values[value-1] += 1
        
        new_number_of_nodes = number_of_nodes + 1

        # subtrees building via recurrent addnode call for each best attribute value:
        for val in range(number_of_values):  
            # data for child node preparation:
            new_num_of_exampl = int(sums_of_ex_by_values[val])
            if new_num_of_exampl > 0:
                new_examples = np.zeros([new_num_of_exampl,num_of_attr], dtype=int)
                new_classes = np.zeros([new_num_of_exampl], dtype=int)
                index = 0
                for ex in range(num_of_exampl):
                    if examples[ex][attr_with_min_entropy] == val+1:
                        for atr in range(num_of_attr):
                            new_examples[index][atr] = examples[ex][atr]
                        new_classes[index] = classes[ex]
                        index += 1
                # removing column with chosen attribute:
                new_examples = np.delete(new_examples,attr_with_min_entropy,1) 
                if if_messages:
                    print("for val = " + str(val))
                    print("new_examples = \n" + str(new_examples))
                    print("new_classes = \n" + str(new_classes))
            

                tree[val][0] = new_number_of_nodes + 1 # number of column with child node (+1 for compatibility with Matlab)

                subtree = addnode(new_examples, new_classes, new_attribute_labels,number_of_values,new_number_of_nodes)

                [x, num_of_colums] = subtree.shape
                new_number_of_nodes += num_of_colums
                tree = np.concatenate((tree,subtree), axis=1)
            #else: # if there are no examples for value val : null pointer

    return tree

def build_tree(examples, classes):
    [num_of_exapl, num_of_attr] = examples.shape
    #print(" num_of_exapl = " + str(num_of_exapl) + " num_of_attr = " + str(num_of_attr))
    val_max = 0
    for i in range(num_of_exapl):
        for j in range(num_of_attr):
            if val_max < examples[i][j]:
                 val_max = examples[i][j]
    attribute_labels = np.arange(0,num_of_attr, dtype=int)
    #print("attribute_labels = " + str(attribute_labels))
    tree = addnode(examples, classes, attribute_labels, val_max, 0)

    return tree

def what_class(example,tree):
    [num_of_rows, num_of_nodes] = tree.shape
    node = 0
    if_end = False
    while if_end == False:
        sum_of_elem = 0
        for i in range(num_of_rows-1):
            sum_of_elem += tree[i][int(node)]
        if sum_of_elem == 0:
            return tree[num_of_rows-1][int(node)]   # leaf
        else:
            attr = tree[num_of_rows-1][int(node)] 
            val = example[int(attr)-1]
            next_column = tree[val-1][int(node)]
            if next_column == 0:
                return -1     # no child for this value!
            else:
                node = next_column - 1

def calc_error(examples,classes, tree):
    [num_of_examples, num_of_attributes] = examples.shape
    num_of_errors = 0
    for ex in range(num_of_examples):
        class_num = what_class(examples[ex][:],tree)
        if class_num != classes[ex]:
            num_of_errors += 1
            #print("ex = "+str(ex)+" true class = "+str(classes[ex])+" class from tree = "+str(class_num))
    return num_of_errors


def distribution(examples,classes,tree):
    [num_of_examples, num_of_attributes] = examples.shape
    [num_of_rows, num_of_nodes] = tree.shape
    num_of_classes = np.max(classes)
    dist = np.zeros([num_of_classes,num_of_nodes],dtype=int)

    for ex in range(num_of_examples):
        node = 0
        if_end = False
        while if_end == False:
            dist[int(classes[ex])-1][int(node)] += 1
            sum_of_elem = 0
            for i in range(num_of_rows-1):
                sum_of_elem += tree[i][int(node)]
            if sum_of_elem == 0:
                if_end = True  # leaf
            else:
                attr = tree[num_of_rows-1][int(node)] 
                val = examples[ex][int(attr)-1]
                next_column = tree[val-1][int(node)]
                if next_column == 0:
                    if_end = True     # no child for this value!
                else:
                    node = next_column - 1
    return dist

def depth(tree, node = 0, d = 0):
    [num_of_rows, num_of_nodes] = tree.shape
    d_vect = np.zeros([1],dtype=int)
    d_vect[0] = d
    for i in range(num_of_rows-1):
        if tree[i][node] > 0:
            child_node = tree[i][node]-1
            d_subvect = depth(tree,child_node,d+1)
            d_vect = np.concatenate((d_vect,d_subvect))
    return d_vect

# def chart():
#     x = np.arange(0, 50, 0.1);
#     y = np.sin(x)+np.sin(x*3)*0.5

#     print ('zdefiniowano x = ',x,' y = ',y)
#     print ('przystepuje do rysowania wykresu')
#     plt.plot(x, y)
#     plt.show()


# load discrete data from text file: examples in separate lines, atributes in columns separated by space
# in the last column numbers of classes. Empty lines bypassing.
def load_data(file_name):
    file_ptr = open(file_name, 'r').read()
    lines = file_ptr.split('\n')
    number_of_lines = lines.__len__() - 1
    row_values = lines[0].split()
    number_of_values = row_values.__len__()

    number_of_examples = 0
    for i in range(number_of_lines):
        row_values = lines[i].split()
        number_of_values = row_values.__len__()
        if (number_of_values > 0):
            number_of_examples = number_of_examples + 1
            num_of_columns = number_of_values

    examples = np.zeros([number_of_examples, num_of_columns], dtype=int)
    print("examples shape = " + str(examples.shape))
    
    index = 0
    for i in range(number_of_lines):
        row_values = lines[i].split()
        number_of_values = row_values.__len__()
        if (number_of_values > 0):
            for j in range(number_of_values):
                examples[index][j] = int(row_values[j])
            index = index + 1

    return examples



# load discrete data from text file: examples in separate lines, atributes in columns separated by space
# in the last column numbers of classes
# examples are divided into training and test subsets: examples with odd ordinal numbers for training,
# examples with even ordinal numbers for test
def load_data_odd_even(file_name):
    file_ptr = open(file_name, 'r').read()
    lines = file_ptr.split('\n')
    number_of_lines = lines.__len__() - 1
    row_values = lines[0].split()
    number_of_values = row_values.__len__()

    number_of_examples = 0
    for i in range(number_of_lines):
        row_values = lines[i].split()
        number_of_values = row_values.__len__()
        if (number_of_values > 0):
            number_of_examples = number_of_examples + 1
            number_of_attributes = number_of_values - 1
    test_examples_num = number_of_examples//2
    train_examples_num = number_of_examples - test_examples_num


    train_examples = np.zeros([train_examples_num, number_of_attributes], dtype=int)
    test_examples = np.zeros([test_examples_num, number_of_attributes], dtype=int)
    train_classes = np.zeros([train_examples_num], dtype=int)
    test_classes = np.zeros([test_examples_num], dtype=int)

    index = 0
    for i in range(number_of_lines):
        row_values = lines[i].split()
        number_of_values = row_values.__len__()
        if (number_of_values > 0):
            if (index+1) % 2 == 1:     # odd examples
                for j in range(number_of_attributes):
                    train_examples[index//2][j] = int(row_values[j])
                train_classes[index//2] = int(row_values[number_of_attributes])
            else:
                for j in range(number_of_attributes):
                    test_examples[index//2][j] = int(row_values[j])
                test_classes[index//2] = int(row_values[number_of_attributes])
            index = index + 1

    print(" number of attributes = "+str(number_of_attributes))
    print(" train examples = \n" + str(train_examples))
    print(" train classs = \n" + str(train_classes))
    print(" test examples = \n" + str(test_examples))
    print(" test classs = \n" + str(test_classes))

    return train_examples, train_classes, test_examples, test_classes
