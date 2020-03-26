import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer as xavier
import numpy as np
from base_model import base_model

'''model_dir=r'../HNRE/outputs/ckpt/pcnn_att/lr_0.2/'
checkpoint_path = os.path.join(model_dir, "pcnn-34808")# 从checkpoint中读出数据
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)# reader = tf.train.NewCheckpointReader(checkpoint_path) # 用tf.train中的NewCheckpointReader方法
var_to_shape_map = reader.get_variable_to_shape_map()# 输出权重tensor名字和值
conv = reader.get_tensor('sentence-encoder/conv2d/kernel')
conv_bias = reader.get_tensor('sentence-encoder/conv2d/bias')
word_vec = reader.get_tensor('embedding-lookup/temp_word_embedding')
unknown_vec = reader.get_tensor('embedding-lookup/unk_embedding')
pos1_vec = reader.get_tensor('embedding-lookup/temp_pos1_embedding')
pos2_vec = reader.get_tensor('embedding-lookup/temp_pos2_embedding')
rel_vec = reader.get_tensor('sentence-level-attention/relation_matrix')
bias_vec = reader.get_tensor('loss/bias')'''

'''init_file = '../HNRE/data/initial_vectors/init_vec_pcnn'
init_vec = pickle.load(open(init_file, 'rb'))
conv = init_vec['convkernel']
conv_bias = init_vec['convbias']
word_vec = init_vec['wordvec']
unknown_vec = init_vec['unkvec']
pos1_vec = init_vec['pos1vec']
pos2_vec = init_vec['pos2vec']'''
#conv = None
#conv_bias = None
#word_vec = np.load('../ori_data/txt_processed/vec.npy')
#pos1_vec = None
#pos2_vec = None
#rel_vec = None
#bias_vec = None

FLAGS = tf.flags.FLAGS

class PCNN(base_model):
    def __init__(self,sess):
        super(PCNN, self).__init__(sess=sess)

    def bulid(self,init_vec):

        with tf.variable_scope("embedding-lookup", initializer=xavier(), dtype=tf.float32):

            temp_word_embedding = self._GetVar(init_vec=init_vec, key='wordvec', name='temp_word_embedding',
                shape=[FLAGS.vocabulary_size, FLAGS.word_size],trainable=True)
            unk_word_embedding = self._GetVar(init_vec=init_vec, key='unkvec', name='unk_embedding',shape=[FLAGS.word_size])
            word_embedding = tf.concat([temp_word_embedding, tf.reshape(unk_word_embedding,[1,FLAGS.word_size]),
               tf.reshape(tf.constant(np.zeros(FLAGS.word_size),dtype=tf.float32),[1,FLAGS.word_size])],0)
            temp_pos1_embedding = self._GetVar(init_vec=init_vec, key='pos1vec', name='temp_pos1_embedding',shape=[FLAGS.pos_num,FLAGS.pos_size])
            temp_pos2_embedding = self._GetVar(init_vec=init_vec, key='pos2vec', name='temp_pos2_embedding',shape=[FLAGS.pos_num,FLAGS.pos_size])
            pos1_embedding = tf.concat([temp_pos1_embedding,tf.reshape(tf.constant(np.zeros(FLAGS.pos_size,dtype=np.float32)),[1, FLAGS.pos_size])],0)
            pos2_embedding = tf.concat([temp_pos2_embedding,tf.reshape(tf.constant(np.zeros(FLAGS.pos_size,dtype=np.float32)),[1, FLAGS.pos_size])],0)

            input_word = tf.nn.embedding_lookup(word_embedding, self.word)  # N,max_len,d
            input_pos1 = tf.nn.embedding_lookup(pos1_embedding, self.pos1)
            input_pos2 = tf.nn.embedding_lookup(pos2_embedding, self.pos2)
            input_embedding = tf.concat(values = [input_word, input_pos1, input_pos2], axis = -1)

            temp_type_embedding = tf.get_variable('type_embedding', shape=[FLAGS.type_num,FLAGS.type_dim] ,initializer=xavier(), dtype=tf.float32)
            type_embedding = tf.concat([tf.reshape(tf.constant(np.zeros(FLAGS.type_dim),dtype=tf.float32),[1,FLAGS.type_dim]),temp_type_embedding],0)

            en1_type = tf.nn.embedding_lookup(type_embedding, self.en1_type)    # batchsize,max_type_num,type_dim
            en2_type = tf.nn.embedding_lookup(type_embedding, self.en2_type)
            #en1_type = tf.divide(tf.reduce_sum(en1_type, axis=1), tf.expand_dims(self.en1_type_len, axis=1))
            #en2_type = tf.divide(tf.reduce_sum(en2_type, axis=1), tf.expand_dims(self.en2_type_len, axis=1))
            x_type = tf.concat([en1_type, en2_type], -1)

            '''#att_type = tf.get_variable('att_type', [FLAGS.type_dim,1],initializer=xavier())
            att_1_type = tf.get_variable('att_1_type', [FLAGS.type_dim,50],initializer=xavier())
            att_2_type = tf.get_variable('att_2_type', [50,1],initializer=xavier())
            padding = tf.constant(np.zeros(FLAGS.max_type_num)*(-1e8),dtype=tf.float32)
            en1_type_stack, en2_type_stack = [],[]
            for i in range(FLAGS.batch_size):
                #temp_alpha_1 = tf.squeeze(en1_type[i] @ att_type , -1)  # max_type_num,type_dim * type_dim,1 = max_type_num,1
                #temp_alpha_2 = tf.squeeze(en2_type[i] @ att_type , -1)
                temp_alpha_1 = tf.squeeze(tf.nn.tanh(en1_type[i] @ att_1_type ) @ att_2_type, -1)
                temp_alpha_2 = tf.squeeze(tf.nn.tanh(en2_type[i] @ att_1_type ) @ att_2_type, -1)
                temp_alpha_1 = tf.where(tf.equal(self.en1_type_mask[i], 1), temp_alpha_1, padding)
                temp_alpha_2 = tf.where(tf.equal(self.en2_type_mask[i], 1), temp_alpha_2, padding) # max_type_num
                temp_alpha_1 = tf.nn.softmax(temp_alpha_1)
                temp_alpha_2 = tf.nn.softmax(temp_alpha_2)
                en1_type_stack.append(tf.squeeze(tf.expand_dims(temp_alpha_1,0) @ en1_type[i],0)) # 1,max_type_num * max_type_num,type_dim = 1,type_dim = type_dim
                en2_type_stack.append(tf.squeeze(tf.expand_dims(temp_alpha_2,0) @ en2_type[i],0))
            en1_type_stack = tf.stack(en1_type_stack)
            en2_type_stack = tf.stack(en2_type_stack)
            x_type = tf.concat([en1_type_stack, en2_type_stack], -1)'''

        with tf.variable_scope("encoder"):

            input_dim = input_embedding.shape[-1]
            mask_embedding = tf.constant([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=np.float32)
            pcnn_mask = tf.nn.embedding_lookup(mask_embedding, self.mask)
            input_sentence = tf.expand_dims(input_embedding, axis=1)
            with tf.variable_scope("conv2d"):
                conv_kernel = self._GetVar(init_vec=init_vec,key='convkernel',name='kernel',
                    shape=[1,3,input_dim,FLAGS.hidden_size],trainable=True)
                conv_bias = self._GetVar(init_vec=init_vec,key='convbias',name='bias',shape=[FLAGS.hidden_size],trainable=True)
            x = tf.layers.conv2d(inputs = input_sentence, filters=FLAGS.hidden_size,
                kernel_size=[1,3], strides=[1, 1], padding='same', reuse=tf.AUTO_REUSE)

            sequence = tf.reshape(x, [-1, FLAGS.max_len, FLAGS.hidden_size])

            x = tf.reshape(x, [-1, FLAGS.max_len, FLAGS.hidden_size, 1])
            x = tf.reduce_max(tf.reshape(pcnn_mask, [-1, 1, FLAGS.max_len, 3]) * tf.transpose(x,[0, 2, 1, 3]), axis = 2)
            x = tf.nn.relu(tf.reshape(x, [-1, FLAGS.hidden_size * 3]))

        with tf.variable_scope("selector"):

            attention_1 = tf.get_variable('attention_1', [self.hidden_size,300],initializer=xavier())
            attention_2 = tf.get_variable('attention_2', [300,1],initializer=xavier())
            alpha = tf.squeeze(tf.nn.tanh(x @ attention_1) @ attention_2 , -1)

            bag_repre = []
            for i in range(FLAGS.batch_size):
                m = x[self.scope[i]:self.scope[i+1]]# (n , hidden_size)
                sent_score = tf.nn.softmax(alpha[self.scope[i]:self.scope[i+1]])
                #m = x[self.scope[i][0]:self.scope[i][1]]# (n , hidden_size)
                #sent_score = tf.nn.softmax(alpha[self.scope[i][0]:self.scope[i][1]])
                bag_repre.append(tf.squeeze(tf.matmul(tf.expand_dims(sent_score,0), m)))
            bag_repre = tf.layers.dropout(tf.stack(bag_repre), rate = 1 - self.keep_prob, training = self.istrain)

        with tf.variable_scope("loss"):

            discrimitive_matrix = self._GetVar(init_vec=init_vec, key='disckernel',
                name='discrimitive_matrix', shape=[53, self.hidden_size + FLAGS.type_dim *2])
            bias = self._GetVar(init_vec=init_vec, key='discbias',
                name='bias', shape=[53], initializer=tf.zeros_initializer())

            bag_repre_type = tf.concat([bag_repre,x_type],-1)

            self.logit = tf.matmul(bag_repre_type, discrimitive_matrix, transpose_b=True) + bias
            self.output = tf.nn.softmax(self.logit,-1)

            label_onehot = tf.one_hot(indices=self.label, depth=FLAGS.num_classes, dtype=tf.int32)

            regularizer = tf.contrib.layers.l2_regularizer(0.00001)
            l2_loss = tf.contrib.layers.apply_regularization(regularizer=regularizer, weights_list=tf.trainable_variables())
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label_onehot,logits=self.logit)) + l2_loss

