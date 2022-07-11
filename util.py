from scipy import stats
import numpy as np


# This method computes entropy for information gain
def entropy(class_y):
    # Input:            
    #   class_y         : list of class labels (0's and 1's)
    
    # TODO: Compute the entropy for a list of classes
    N = len(class_y)
    n = sum(class_y)
    m = N - n
    try:
        entropy = -n/N*np.log2(n/N)-(m/N)*np.log2(m/N)
    except:
        return 0
    #entropy = 0
    if np.isnan(entropy):
        return 0
    else:
        return entropy

#     # Example:
#     #    entropy([0,0,0,1,1,1,1,1,1]) = 0.92
        



def partition_classes(X, y, split_attribute, split_val):
    # Inputs:
    #   X               : data containing all attributes
    #   y               : labels
    #   split_attribute : column index of the attribute to split on
    #   split_val       : either a numerical or categorical value to divide the split_attribute
    
    # TODO: Partition the data(X) and labels(y) based on the split value - BINARY SPLIT.
    # 
    # You will have to first check if the split attribute is numerical or categorical    
    # If the split attribute is numeric, split_val should be a numerical value
    # For example, your split_val could be the mean of the values of split_attribute
    # If the split attribute is categorical, split_val should be one of the categories.   
    #
    # You can perform the partition in the following way
    # Numeric Split Attribute:
    #   Split the data X into two lists(X_left and X_right) where the first list has all
    #   the rows where the split attribute is less than or equal to the split value, and the 
    #   second list has all the rows where the split attribute is greater than the split 
    #   value. Also create two lists(y_left and y_right) with the corresponding y labels.
    #
    # Categorical Split Attribute:
    #   Split the data X into two lists(X_left and X_right) where the first list has all 
    #   the rows where the split attribute is equal to the split value, and the second list
    #   has all the rows where the split attribute is not equal to the split value.
    #   Also create two lists(y_left and y_right) with the corresponding y labels.

    '''
    Example:
    
    X = [[3, 'aa', 10],                 y = [1,
         [1, 'bb', 22],                      1,
         [2, 'cc', 28],                      0,
         [5, 'bb', 32],                      0,
         [4, 'cc', 32]]                      1]
    
    Here, columns 0 and 2 represent numeric attributes, while column 1 is a categorical attribute.
    
    Consider the case where we call the function with split_attribute = 0 and split_val = 3 (mean of column 0)
    Then we divide X into two lists - X_left, where column 0 is <= 3  and X_right, where column 0 is > 3.
    
    X_left = [[3, 'aa', 10],                 y_left = [1,
              [1, 'bb', 22],                           1,
              [2, 'cc', 28]]                           0]
              
    X_right = [[5, 'bb', 32],                y_right = [0,
               [4, 'cc', 32]]                           1]

    Consider another case where we call the function with split_attribute = 1 and split_val = 'bb'
    Then we divide X into two lists, one where column 1 is 'bb', and the other where it is not 'bb'.
        
    X_left = [[1, 'bb', 22],                 y_left = [1,
              [5, 'bb', 32]]                           0]
              
    X_right = [[3, 'aa', 10],                y_right = [1,
               [2, 'cc', 28],                           0,
               [4, 'cc', 32]]                           1]
               
    '''     
    X_left = []
    X_right = []
    
    y_left = []
    y_right = []
    nrows = len(X)

    # Check if splitting on numerical or categorical variable
    if(type(split_val) == str):
        for i in range(nrows):
            #print(X[i][split_attribute])
            if(X[i][split_attribute] == split_val):
                #print("yes")
                X_left.append(X[i])
                y_left.append(y[i])
            else:
                #print("no")
                X_right.append(X[i])
                y_right.append(y[i])

    else:
        for i in range(nrows):
            #print(X[i][split_attribute])
            if(X[i][split_attribute] <= split_val):
                #print("yes")
                X_left.append(X[i])
                y_left.append(y[i])
            else:
                #print("no")
                X_right.append(X[i])
                y_right.append(y[i])

    return (X_left, X_right, y_left, y_right)

    
def information_gain(previous_y, current_y):
    # Inputs:
    #   previous_y: the distribution of original labels (0's and 1's)
    #   current_y:  the distribution of labels after splitting based on a particular
    #               split attribute and split value
    
    # TODO: Compute and return the information gain from partitioning the previous_y labels
    # into the current_y labels.
    # You will need to use the entropy function above to compute information gain
    # Reference: http://www.cs.cmu.edu/afs/cs.cmu.edu/academic/class/15381-s06/www/DTs.pdf
    
    """
    Example:
    
    previous_y = [0,0,0,1,1,1]
    current_y = [[0,0], [1,1,1,0]]
    
    info_gain = 0.45915
    """

    #info_gain = 0
    H = entropy(previous_y)
    H_l = entropy(current_y[0])
    H_r = entropy(current_y[1])
    P_l = len(current_y[0])/len(previous_y)
    P_r = len(current_y[1])/len(previous_y)

    info_gain = H - (H_l*P_l + H_r*P_r)

    return info_gain
    
def best_split(X, y):
    b_col, b_value, b_score = 999, 999, -1
    m = int(round(np.sqrt(len(X[0]))))
    rand_cols = np.random.choice(range(len(X[0])), size=m, replace=False)
    for col in rand_cols:
        for row in X:
            X_l, X_r, y_l, y_r = partition_classes(X, y, col, row[col])
            info_gain = information_gain(y, [y_l, y_r])
            if info_gain > b_score:
                b_col, b_value, b_score  = col, row[col], info_gain
    return {'b_col':b_col, 'b_value':b_value, 'b_score':b_score}
    