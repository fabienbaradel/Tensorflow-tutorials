import tensorflow as tf

"""
First steps with Tensorflow:
- introduction to constant op
- introduction to variable
- first approach of feeding data
"""


######################################
### CREATE A CONSTANT AND PRINT IT ###
######################################

# Create a constant
hello = tf.constant('Welcome to TensorFlow ')
# Info about the constant
print(hello)

# Intro tf session
with tf.Session() as sess:
    # Run the op
    print(sess.run(hello))


#######################################
### FEEDING STRING TO A PLACEHOLDER ###
#######################################

# 1) Create a placeholder
my_placeholder = tf.placeholder(dtype=tf.string)

# 2) Create my operation
my_string = my_placeholder

# 3) Intro tf session
with tf.Session() as sess:
    # Run the op
    print(sess.run(my_string, feed_dict={my_placeholder: "hello"}))


################
### EXERCISE ###
################
"""
Print the concatenation of the hello constant and your name:
 => you only have to create an operation between 'hello' and 'placeholder'
"""

# Placeholder and the constant
my_placeholder = tf.placeholder(dtype=tf.string)
hello = tf.constant('Welcome to TensorFlow ')

# Create your own operation (only one line...)
welcome = tf.add(my_placeholder, hello)

# Intro tf session
with tf.Session() as sess:
    # Run the op
    print(sess.run(welcome, feed_dict={my_placeholder: "Fabien"}))
    print(sess.run(welcome, feed_dict={my_placeholder: "Jean"}))
    print(sess.run(welcome, feed_dict={my_placeholder: "Romaric"}))



