import tensorflow as tf

"""
Same as with string, we now discover basic operations in Tensorflow for interger/float
"""

############################
## DEALING WITH CONSTANT ###
############################

a = tf.constant(2)
b = tf.constant(3)

# Define some operations
add = tf.add(a, b)
mul = tf.mul(a, b)


# Launch the default graph.
with tf.Session() as sess:
    # Print the TF constants
    print("The constant a: " + str(a) ) # a is not equal to 2 but the constant a is assign to 2 during the session
    print("The constant a: " + str(b))

    # Get the values of each constants
    (a_value, b_value) = sess.run([a, b]) # a_value and b_value are not anymore Tensorflow operation
    print("a=%i, b=%i" % (a_value, b_value))

    # Basic math operation on constant
    print("Addition with constants: %i" % sess.run(add))
    print("Multiplication with constants: %i" % sess.run(mul))


#######################################
### WITH PLACEHOLDERS AND VARIABLES ###
#######################################

a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

# Define some operations
add = tf.add(a, b)
mul = tf.mul(a, b)

# Launch the default graph.
with tf.Session() as sess:
    # Run every operation with variable input
    print("Addition with variables: %i" % sess.run(add, feed_dict={a: 2, b: 3}))
    print("Multiplication with variables: %i" % sess.run(mul, feed_dict={a: 2, b: 3}))


##################
### EXERCISE 1 ###
##################
"""
Dealing with matrices
"""
import numpy as np

# My two placeholders for the matrices
A = tf.placeholder(tf.int32, shape=[2,3])
A.get_shape()
B = tf.placeholder(tf.int32, shape=[2,3])
B.get_shape()

# Compute = A'.B
mult = ?????

# Launch the default graph.
with tf.Session() as sess:
    # Create two numpy matrices
    A_value = np.array([1, 2, 3, 4, 5, 6]).reshape((2,3))
    print("A = ")
    print(A_value)
    print("")
    B_value = np.array([1, 0, 2, 0, 0, -1]).reshape((2, 3))
    print("B = ")
    print(B_value)
    print("")


    # Get value for the mult operation
    mult_value = sess.run(mult, feed_dict={A: A_value, B: B_value})
    print("mult A' by B = ")
    print(mult_value)
    print("")


##################
### EXERCISE 2 ###
##################
"""
Dealing with matrices again
"""

# My two placeholders for the matrices
A = tf.placeholder(tf.int32, shape=[2,3])
A.get_shape()
B = tf.placeholder(tf.int32, shape=[2,3])
B.get_shape()

# Pairwise multiplication and sum over the matrix
mult_element_by_element = ?????
sum = ?????

# Launch the default graph.
with tf.Session() as sess:
    # Create two numpy matrices
    A_value = np.array([1, 2, 3, 4, 5, 6]).reshape((2,3))
    print("A = ")
    print(A_value)
    print("")
    B_value = np.array([1, 0, 2, 0, 0, -1]).reshape((2, 3))
    print("B = ")
    print(B_value)
    print("")

    # Get value for the element by element multiplication operation
    mult_element_by_element_value = sess.run(mult_element_by_element, feed_dict={A: A_value, B: B_value})
    print("A by B element by element = ")
    print(mult_element_by_element_value)
    print("")

    # Get value for sum operation
    sum_value = sess.run(sum, feed_dict={A: A_value, B: B_value})
    print("A by B element by element then sum = ")
    print(sum_value)
    print("")






