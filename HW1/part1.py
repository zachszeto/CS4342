import math
import csv
import numpy as np
from collections import Counter

# Note: please don't add any new package, you should solve this problem using only the packages above.
# However, importing the Python standard library is allowed: https://docs.python.org/3/library/
#-------------------------------------------------------------------------
'''
    Part 1: Decision Tree (with Discrete Attributes) -- 60 points --
    In this problem, you will implement the decision tree method for classification problems.
    You could test the correctness of your code by typing `pytest -v test1.py` in the terminal.
'''

#-----------------------------------------------
class Node:
    '''
        Decision Tree Node (with discrete attributes)
        Inputs: 
            X: the data instances in the node, a numpy matrix of shape p by n.
               Each element can be int/float/string.
               Here n is the number data instances in the node, p is the number of attributes.
            Y: the class labels, a numpy array of length n.
               Each element can be int/float/string.
            i: the index of the attribute being tested in the node, an integer scalar 
            C: the dictionary of attribute values and children nodes. 
               Each (key, value) pair represents an attribute value and its corresponding child node.
            isleaf: whether or not this node is a leaf node, a boolean scalar
            p: the label to be predicted on the node (i.e., most common label in the node).
    '''
    def __init__(self,X,Y, i=None,C=None, isleaf= False,p=None):
        self.X = X
        self.Y = Y
        self.i = i
        self.C= C
        self.isleaf = isleaf
        self.p = p

#-----------------------------------------------
class Tree(object):
    '''
        Decision Tree (with discrete attributes). 
        We are using ID3(Iterative Dichotomiser 3) algorithm. So this decision tree is also called ID3.
    '''
    #--------------------------
    @staticmethod
    def entropy(Y: np.ndarray):
        '''
            Compute the entropy of a list of values.
            Input:
                Y: a list of values, a numpy array of int/float/string values.
            Output:
                e: the entropy of the list of values, a float scalar
            Hint: you could use collections.Counter.
        '''
        #########################################
        # Entropy(S) = - Σ [pⱼ * log₂(pⱼ)]
        counter = Counter(Y)
        total = counter.total()

        e = 0
        for element, count in counter.items(): 
            e += count/total * math.log2(count/total)
        
        #########################################
        return -e 
    
    
            
    #--------------------------
    @staticmethod
    def conditional_entropy(Y: np.ndarray, X: np.ndarray):
        '''
            Compute the conditional entropy of y given x. The conditional entropy H(Y|X) means average entropy of children nodes, given attribute X. Refer to https://en.wikipedia.org/wiki/Information_gain_in_decision_trees
            Input:
                X: a list of values , a numpy array of int/float/string values. The size of the array means the number of instances/examples. X contains each instance's attribute value. 
                Y: a list of values, a numpy array of int/float/string values. Y contains each instance's corresponding target label. For example X[0]'s target label is Y[0]
            Output:
                ce: the conditional entropy of y given x, a float scalar
        '''
        #########################################
        xCounter = Counter(X)
        xTotal = xCounter.total()

        ce = 0
        for element, count in xCounter.items():
            ce += count/xTotal * Tree.entropy(Y[X == element])
 
        #########################################
        return ce 
    
    
    
    #--------------------------
    @staticmethod
    def information_gain(Y: np.ndarray, X: np.ndarray):
        '''
            Compute the information gain of y after spliting over attribute x
            InfoGain(Y,X) = H(Y) - H(Y|X) 
            Input:
                X: a list of values, a numpy array of int/float/string values.
                Y: a list of values, a numpy array of int/float/string values.
            Output:
                g: the information gain of y after spliting over x, a float scalar
        '''
        #########################################
        ## Information Gain = entropy(parent) – [average entropy(children)]
        g = Tree.entropy(Y) - Tree.conditional_entropy(Y, X)
 
        #########################################
        return g


    #--------------------------
    @staticmethod
    def best_attribute(X: np.ndarray, Y: np.ndarray):
        '''
            Find the best attribute to split the node. 
            Here we use information gain to evaluate the attributes. 
            If there is a tie in the best attributes, select the one with the smallest index.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
            Output:
                i: the index of the attribute to split, an integer scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        information_gain = 0
        i = -1

        for attribute_index in range(len(X)):
            tmp = Tree.information_gain(Y, X[attribute_index])

            if tmp > information_gain:
                information_gain = tmp
                i = attribute_index

        # Gaurd Clause
        if i == -1: raise Exception("Failed to find best attribute")

        #########################################
        return i

        
    #--------------------------
    @staticmethod
    def split(X: np.ndarray, Y: np.ndarray, i: int):
        '''
            Split the node based upon the i-th attribute.
            (1) split the matrix X based upon the values in i-th attribute
            (2) split the labels Y based upon the values in i-th attribute
            (3) build children nodes by assigning a submatrix of X and Y to each node
            (4) build the dictionary to combine each  value in the i-th attribute with a child node.
    
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
                i: the index of the attribute to split, an integer scalar
            Output:
                C: the dictionary of attribute values and children nodes. 
                   Each (key, value) pair represents an attribute value and its corresponding child node.
        '''
        #########################################
         # Get unique values in the i-th attribute
        unique_values = np.unique(X[i])
        C = {}
        
        # For each unique value in the splitting attribute
        for value in unique_values:
            mask = (X[i] == value)            
            X_child = X[:, mask]
            Y_child = Y[mask]
            
            child_node = Node(X=X_child, Y=Y_child)
            C[value] = child_node

        #########################################
        return C

    #--------------------------
    @staticmethod
    def stop1(Y: np.ndarray):
        '''
            Test condition 1 (stop splitting): whether or not all the instances have the same label. 
    
            Input:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
            Output:
                s: whether or not Conidtion 1 holds, a boolean scalar. 
                True if all labels are the same. Otherwise, false.
        '''
        #########################################
        # Check if all elements in Y are the same as the first element
        s = np.all(Y == Y[0])
        #########################################
        return s
    
    #--------------------------
    @staticmethod
    def stop2(X: np.ndarray):
        '''
            Test condition 2 (stop splitting): whether or not all the instances have the same attribute values. 
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
            Output:
                s: whether or not Conidtion 2 holds, a boolean scalar. 
        '''
        #########################################
        # Check if all columns in X are the same as the first column
        s = np.all(X == X[:, [0]]) 
        #########################################
        return s
    
            
    #--------------------------
    @staticmethod
    def most_common(Y: np.ndarray):
        '''
            Get the most-common label from the list Y. 
            Input:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node.
            Output:
                y: the most common label, a scalar, can be int/float/string.
        '''
        #########################################
        counter = Counter(Y)
        y = counter.most_common(1)[0][0]
        #########################################
        return y
    
    
    
    #--------------------------
    @staticmethod
    def build_tree(t):
        '''
            Recursively build tree nodes.
            Input:
                t: a node of the decision tree, without the subtree built.
                t.X: the feature matrix, a numpy float matrix of shape p by n.
                   Each element can be int/float/string.
                    Here n is the number data instances, p is the number of attributes.
                t.Y: the class labels of the instances in the node, a numpy array of length n.
        '''
        #########################################
        # Case 1: All labels are the same
        if Tree.stop1(t.Y):
            t.isleaf = True
            t.p = t.Y[0]
            return t
        
        # Case 2: All instances have identical attributes
        if Tree.stop2(t.X):
            t.isleaf = True
            t.p = Tree.most_common(t.Y)
            return t
        
        # Otherwise find best attribute to split on
        best_attr_index = Tree.best_attribute(t.X, t.Y)
        t.i = best_attr_index
        t.p = Tree.most_common(t.Y)
        t.C = Tree.split(t.X, t.Y, best_attr_index)
        
        for attribute_value, child_node in t.C.items():
            Tree.build_tree(child_node)

        #########################################
        return t
    
    
    #--------------------------
    @staticmethod
    def train(X, Y):
        '''
            Given a training set, train a decision tree. 
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the training set, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
            Output:
                t: the root of the tree.
        '''
        #########################################
        t = Node(X, Y)
        Tree.build_tree(t)
        #########################################
        return t
    
    
    
    #--------------------------
    @staticmethod
    def inference(t,x):
        '''
            Given a decision tree and one data instance, infer the label of the instance recursively. 
            Input:
                t: the root of the tree.
                x: the attribute vector, a numpy vectr of shape p.
                   Each attribute value can be int/float/string.
            Output:
                y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''
        #########################################
        # Base case: If current node is a leaf, return its prediction
        if t.isleaf:
            return t.p
        
        split_index = t.i
        split_value = x[split_index]
        
        if t.C is None:
            return t.p
        
        # Check if the attribute value exists in children
        if split_value in t.C:
            child_node = t.C[split_value]
            return Tree.inference(child_node, x)
        else:
            return t.p
   
        #########################################
    
    #--------------------------
    @staticmethod
    def predict(t,X):
        '''
            Given a decision tree and a dataset, predict the labels on the dataset. 
            Input:
                t: the root of the tree.
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the dataset, p is the number of attributes.
            Output:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''
        #########################################
        n = X.shape[1]
        Y = np.empty(n, dtype=object)
        
        # Predict each instance individually
        for i in range(n):
            instance = X[:, i]
            Y[i] = Tree.inference(t, instance)

        #########################################
        return Y


    #--------------------------
    @staticmethod
    def load_dataset(filename = 'data1.csv'):
        '''
            Load dataset 1 from the CSV file: 'data1.csv'. 
            The first row of the file is the header (including the names of the attributes)
            In the remaining rows, each row represents one data instance.
            The first column of the file is the label to be predicted.
            In remaining columns, each column represents an attribute.
            Input:
                filename: the filename of the dataset, a string.
            Output:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the dataset, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''
        #########################################
        with open(filename, 'r', newline='') as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader)
            data_rows = list(csv_reader)
            data = np.array(data_rows)
            Y = data[:, 0]
            X = data[:, 1:].T
            
        #########################################
        return X,Y



