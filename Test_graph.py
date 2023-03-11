import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#面積的公式等於長乘以寬
width = tf.placeholder("int32",name='width')
height = tf.placeholder("int32",name='height')
area = tf.multiply(width,height,name='area')
#定義Session
with tf.Session() as sess:
     init = tf.global_variables_initializer()
     sess.run(init)
     print('[Alyson log] area=',sess.run(area,feed_dict={width: 6,height: 8}))

from datetime import datetime 
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")               
root_logdir = "tf_logs"     
logdir = "{}/run-{}/".format(root_logdir, now) 

tf.summary.merge_all()
file_writer =tf.summary.FileWriter(logdir,tf.get_default_graph())


file_writer.close() 

print("suc")