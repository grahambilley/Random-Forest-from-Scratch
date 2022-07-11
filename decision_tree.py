from util import entropy, information_gain, partition_classes, best_split
from scipy import stats
import numpy as np 
import ast

class DecisionTree(object):
    def __init__(self):
        # Initializing the tree as an empty dictionary or list, as preferred
        #self.tree = []
        self.tree = {}
        pass

    def learn(self, X, y):
        # TODO: Train the decision tree (self.tree) using the the sample X and labels y
        # You will have to make use of the functions in utils.py to train the tree
        
        # One possible way of implementing the tree:
        #    Each node in self.tree could be in the form of a dictionary:
        #       https://docs.python.org/2/library/stdtypes.html#mapping-types-dict
        #    For example, a non-leaf node with two children can have a 'left' key and  a 
        #    'right' key. You can add more keys which might help in classification
        #    (eg. split attribute and split value)

        new_tree = {}
        # Select the best split point for a dataset
        split = best_split(X, y)
        # print('split: ',split)
        if split['b_score'] == 0:
            # print('leaf')
            # print(stats.mode(y)[0][0],'\n')
            new_tree['leaf'] = stats.mode(y)[0][0]

        else:
            partition = (split['b_col'], split['b_value'])
            # print('partition: ', partition)
            X_l, X_r, y_l, y_r = partition_classes(X, y, partition[0], partition[1])
            # print('left: ', X_l, y_l)
            # print('right: ', X_r, y_r,'\n')
            new_tree['partition'] = partition
            new_tree['left'] = self.learn(X_l, y_l)
            new_tree['right'] = self.learn(X_r, y_r)

        self.tree = new_tree
        return self.tree
    
    def classify(self, record):
        # TODO: classify the record using self.tree and return the predicted label
        # Test if you are in a leaf node
        
        if isinstance(record[-1], dict):
            temp_tree = record[-1]
            del record[-1]
        else:
            temp_tree = self.tree
            
        # print(record)
        if 'leaf' in temp_tree.keys():
            # print('leaf')
            return temp_tree['leaf']
        # If you are not in a leaf node
        else:
            try:
                col = temp_tree['partition'][0]
                val = temp_tree['partition'][1]
            except:
                return 1
            # print('col:', col)
            # print('val:', val)

            if isinstance(record[col], str):
                # Categorical example
                # print('Str')
                # print('record col: ',record[col], 'val', val, '\n')
                if record[col] == val:
                    # Check left branch
                    # print('going left:')
                    # print('temp:', temp_tree, '\n')
                    record.append(temp_tree['left'])
                    # print('l:', temp_tree, '\n')
                    return self.classify(record)
                else:
                    # Check right branch
                    # print('going right:')
                    # print('temp:', temp_tree, '\n')
                    record.append(temp_tree['right'])
                    # print('r:', temp_tree, '\n')
                    return self.classify(record)
            else:
                # Numeric example
                # print('Num')
                # print('record col: ',record[col], 'val', val, '\n')
                if record[col] <= val:
                    # Check left branch
                    # print('going left:')
                    # print('temp:', temp_tree, '\n')
                    record.append(temp_tree['left'])
                    # print('l:', temp_tree, '\n')
                    return self.classify(record)
                else:
                    # Check right branch
                    # print('going right:')
                    # print('temp:', temp_tree, '\n')
                    record.append(temp_tree['right'])
                    # print('r:', temp_tree, '\n')
                    return self.classify(record)    
