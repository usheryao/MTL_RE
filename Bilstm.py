import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer as xavier
import numpy as np
import pickle,random,time,sys,os
from tensorflow.python import pywrap_tensorflow
from sklearn.metrics import average_precision_score

FLAGS = tf.flags.FLAGS

class Bilstm():
    def __init__(self,sess,optimizer=tf.train.GradientDescentOptimizer):

        self.sess = sess

        self.place_holder()

        init_file = '../raw_HNRE/data/initial_vectors/init_vec_pcnn'
        init_vec = pickle.load(open(init_file, 'rb'))
        self.bulid(init_vec)
        self.sess.run(tf.global_variables_initializer())

        self.opt = optimizer(FLAGS.learning_rate)
        #tf.group(train_op_slow, train_op_fast)

    def place_holder(self):

        #self.global_step = tf.Variable(0,trainable=False,name='global_step')
        self.istrain = tf.placeholder(dtype=tf.bool,name='istrain')
        self.keep_prob = tf.placeholder(dtype=tf.float32,name='keep_prob')

        #self.scope = tf.placeholder(dtype=tf.int32, shape=[None, 2], name='scope')
        self.scope = tf.placeholder(dtype=tf.int32, shape=[FLAGS.batch_size + 1], name='scope')
        self.word = tf.placeholder(dtype=tf.int32,shape=[None,FLAGS.max_len],name='word')
        self.pos1 = tf.placeholder(dtype=tf.int32,shape=[None,FLAGS.max_len],name='pos1')
        self.pos2 = tf.placeholder(dtype=tf.int32,shape=[None,FLAGS.max_len],name='pos2')
        self.label = tf.placeholder(dtype=tf.int32, shape=[FLAGS.batch_size], name='label')
        #self.label = tf.placeholder(dtype=tf.int32, shape=[None], name='label')
        self.mask = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.max_len], name="mask")
        #self.label_idx = tf.placeholder(dtype=tf.int32,shape=[None],name='label_idx')
        self.len = tf.placeholder(dtype=tf.int32,shape=[None],name='len')
        self.en1_type = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size,FLAGS.type_num+1], name='en1_type')
        self.en2_type = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size,FLAGS.type_num+1], name='en2_type')
        #self.en1_type = tf.placeholder(dtype=tf.int32, shape=[FLAGS.batch_size], name='en1_type')
        #self.en2_type = tf.placeholder(dtype=tf.int32, shape=[FLAGS.batch_size], name='en2_type')
        #self.en1_type_len = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size], name='en1_len')
        #self.en2_type_len = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size], name='en2_len')
        #self.en1_type_mask = tf.placeholder(dtype=tf.int32, shape=[FLAGS.batch_size, 5], name='en1_type_mask')
        #self.en2_type_mask = tf.placeholder(dtype=tf.int32, shape=[FLAGS.batch_size, 5], name='en2_type_mask')
        #self.en1_type = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.max_type_num], name='en1_type')
        #self.en2_type = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.max_type_num], name='en2_type')
        #self.en1_type_len = tf.placeholder(dtype=tf.float32, shape=[None], name='en1_len')
        #self.en2_type_len = tf.placeholder(dtype=tf.float32, shape=[None], name='en2_len')
        self.en1_word = tf.placeholder(dtype=tf.int32,shape=[FLAGS.batch_size],name='ent1_word')
        self.en2_word = tf.placeholder(dtype=tf.int32,shape=[FLAGS.batch_size],name='ent2_word')
        #self.word_type = tf.placeholder(dtype=tf.int32,shape=[FLAGS.batch_size,FLAGS.max_len],name='word_type')
        #self.pos1_type = tf.placeholder(dtype=tf.int32,shape=[FLAGS.batch_size,FLAGS.max_len],name='pos1_type')
        #self.pos2_type = tf.placeholder(dtype=tf.int32,shape=[FLAGS.batch_size,FLAGS.max_len],name='pos2_type')
        #self.mask_type = tf.placeholder(dtype=tf.int32, shape=[FLAGS.batch_size, FLAGS.max_len], name="mask_type")
        #self.len_type = tf.placeholder(dtype=tf.int32,shape=[FLAGS.batch_size],name='len_type')

    def bulid(self,init_vec):

        with tf.variable_scope("embedding-lookup", initializer=xavier(), dtype=tf.float32):

            temp_word_embedding = self._GetVar(init_vec=init_vec, key='wordvec', name='temp_word_embedding',
                shape=[FLAGS.vocabulary_size, FLAGS.word_size],trainable=True)
            unk_word_embedding = self._GetVar(init_vec=init_vec, key='unkvec', name='unk_embedding',shape=[FLAGS.word_size],trainable=True)
            word_embedding = tf.concat([temp_word_embedding, tf.reshape(unk_word_embedding,[1,FLAGS.word_size]),
               tf.reshape(tf.constant(np.zeros(FLAGS.word_size),dtype=tf.float32),[1,FLAGS.word_size])],0)
            temp_pos1_embedding = self._GetVar(init_vec=init_vec, key='pos1vec', name='temp_pos1_embedding',shape=[FLAGS.pos_num,FLAGS.pos_size],trainable=True)
            temp_pos2_embedding = self._GetVar(init_vec=init_vec, key='pos2vec', name='temp_pos2_embedding',shape=[FLAGS.pos_num,FLAGS.pos_size],trainable=True)
            pos1_embedding = tf.concat([temp_pos1_embedding,tf.reshape(tf.constant(np.zeros(FLAGS.pos_size,dtype=np.float32)),[1, FLAGS.pos_size])],0)
            pos2_embedding = tf.concat([temp_pos2_embedding,tf.reshape(tf.constant(np.zeros(FLAGS.pos_size,dtype=np.float32)),[1, FLAGS.pos_size])],0)

            input_word = tf.nn.embedding_lookup(word_embedding, self.word)  # N,max_len,d
            input_pos1 = tf.nn.embedding_lookup(pos1_embedding, self.pos1)
            input_pos2 = tf.nn.embedding_lookup(pos2_embedding, self.pos2)

        with tf.variable_scope("entity_typing"):

            input_type_1 = tf.concat(values = [input_word, input_pos1], axis = -1)
            input_type_2 = tf.concat(values = [input_word, input_pos2], axis = -1)

            input_type_1 = tf.concat(values = [input_type_1, input_type_2], axis = 0)

            lstm_cell_forward = tf.contrib.rnn.BasicLSTMCell(FLAGS.rnn_size)
            lstm_cell_backward = tf.contrib.rnn.BasicLSTMCell(FLAGS.rnn_size)

            len = tf.concat([self.len,self.len],0)
            (all_states, last_states) = tf.nn.bidirectional_dynamic_rnn(lstm_cell_forward,lstm_cell_backward,input_type_1,dtype=tf.float32,sequence_length=len)
            (fw_outputs,bw_outputs) = (all_states)  # N,max_len,grusize
            outputs_1 = tf.concat([fw_outputs,bw_outputs],-1) # N,max_len,grusize*2

            ET_att_1 = tf.get_variable('ET_att_1', [FLAGS.rnn_size*2,128],initializer=xavier())
            ET_att_2 = tf.get_variable('ET_att_2', [128,1],initializer=xavier())
            padding_1 = tf.ones_like(self.mask,dtype=tf.float32) * tf.constant([-1e8])
            padding = tf.concat([padding_1,padding_1],0)
            mask = tf.concat([self.mask,self.mask],0)

            outputs_1_ = tf.reshape(outputs_1,[-1,FLAGS.rnn_size*2])
            temp_alpha_1 = tf.reshape(tf.nn.relu(outputs_1_ @ ET_att_1) @ ET_att_2, [-1,FLAGS.max_len])
            temp_alpha_1 = tf.where(tf.equal(mask, 0), padding, temp_alpha_1)
            alpha_1 = tf.nn.softmax(temp_alpha_1,-1)    # N,max_len
            outputs_1 = tf.reshape(tf.expand_dims(alpha_1,1) @ outputs_1, [-1,FLAGS.rnn_size*2])

            ET_sent_att_1 = tf.get_variable('ET_sent_att_1', [FLAGS.rnn_size*2,128],initializer=xavier())
            ET_sent_att_2 = tf.get_variable('ET_sent_att_2', [128,1],initializer=xavier())
            alpha_type_sent_1 = tf.squeeze(tf.nn.tanh(outputs_1 @ ET_sent_att_1) @ ET_sent_att_2 , -1)

            type_repre_1 = []
            type_repre_2 = []
            for i in range(FLAGS.batch_size):
                m = outputs_1[self.scope[i]:self.scope[i+1]]# (n , hidden_size)
                sent_score = tf.nn.softmax(alpha_type_sent_1[self.scope[i]:self.scope[i+1]])
                type_repre_1.append(tf.squeeze(tf.matmul(tf.expand_dims(sent_score,0), m)))

            for i in range(FLAGS.batch_size):
                m = outputs_1[self.scope[i]+FLAGS.batch_size:self.scope[i+1]+FLAGS.batch_size]# (n , hidden_size)
                sent_score = tf.nn.softmax(alpha_type_sent_1[self.scope[i]+FLAGS.batch_size:self.scope[i+1]+FLAGS.batch_size])
                type_repre_2.append(tf.squeeze(tf.matmul(tf.expand_dims(sent_score,0), m)))

            type_repre_1 = tf.layers.dropout(tf.stack(type_repre_1), rate = 1 - self.keep_prob, training = self.istrain)
            type_repre_2 = tf.layers.dropout(tf.stack(type_repre_2), rate = 1 - self.keep_prob, training = self.istrain)

            ent1_word = tf.nn.embedding_lookup(word_embedding, self.en1_word)
            ent2_word = tf.nn.embedding_lookup(word_embedding, self.en2_word)

            en1_outputs = tf.concat([type_repre_1,ent1_word],-1)
            en2_outputs = tf.concat([type_repre_2,ent2_word],-1)

            ET_matrix = self._GetVar(init_vec=init_vec, key='disckernel',
                name='ET_matrix', shape=[39, FLAGS.rnn_size*2 + FLAGS.word_size])
            ET_bias = self._GetVar(init_vec=init_vec, key='discbias',
                name='ET_bias', shape=[39], initializer=tf.zeros_initializer())

            logits_1 = tf.matmul(en1_outputs, ET_matrix, transpose_b=True) + ET_bias
            logits_2 = tf.matmul(en2_outputs, ET_matrix, transpose_b=True) + ET_bias

            loss_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.en1_type,logits=logits_1))
            loss_2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.en2_type,logits=logits_2))

            output_1 = tf.nn.sigmoid(logits_1) # batchsize, 39
            output_2 = tf.nn.sigmoid(logits_2)
            ones = tf.ones_like(logits_1)
            zeros = tf.zeros_like(logits_1)
            self.output_1 = tf.where(tf.greater(output_1, 0.5), ones, zeros)    # batch_size, 39
            self.output_2 = tf.where(tf.greater(output_2, 0.5), ones, zeros)

        with tf.variable_scope("loss"):

            regularizer = tf.contrib.layers.l2_regularizer(0.00001)
            l2_loss = tf.contrib.layers.apply_regularization(regularizer=regularizer, weights_list=tf.trainable_variables())
            self.loss = l2_loss + loss_1 + loss_2

    def train(self):

        #ET_idx = np.load(FLAGS.data_path+'ET_train_idx.npy')

        #data_path = '../raw_HNRE/data/'
        train_scope = np.load(FLAGS.data_path+'train_scope.npy')
        train_word = np.load(FLAGS.data_path+'train_word.npy')
        train_pos1 = np.load(FLAGS.data_path+'train_pos1.npy')
        train_pos2 = np.load(FLAGS.data_path+'train_pos2.npy')
        train_mask = np.load(FLAGS.data_path+'train_mask.npy')
        train_len = np.load(FLAGS.data_path+'train_len.npy')
        train_label = np.load(FLAGS.data_path+'train_label.npy')
        #train_en1_type = np.load(FLAGS.data_path+'train_en1_type.npy')
        #train_en2_type = np.load(FLAGS.data_path+'train_en2_type.npy')
        train_en1_type = np.load(FLAGS.data_path+'train_en1_type_nhot.npy')
        train_en2_type = np.load(FLAGS.data_path+'train_en2_type_nhot.npy')


        #train_en1_type_len = np.load(FLAGS.data_path+'train_en1_len.npy')
        #train_en2_type_len = np.load(FLAGS.data_path+'train_en2_len.npy')
        train_en1_word = np.load(FLAGS.data_path+'train_en1_word.npy')
        train_en2_word = np.load(FLAGS.data_path+'train_en2_word.npy')
        #train_word = train_word[ET_idx]
        #train_pos1 = train_pos1[ET_idx]
        #train_pos2 = train_pos2[ET_idx]
        #train_mask = train_mask[ET_idx]
        #train_len = train_len[ET_idx]
        #train_en1_word = train_en1_word[ET_idx]
        #train_en2_word = train_en2_word[ET_idx]
        #print(train_en1_type.shape)
        #train_en1_type = train_en1_type[ET_idx][:,0]
        #train_en2_type = train_en2_type[ET_idx][:,0]
        #train_en1_type = train_en1_type[:,0]
        #train_en2_type = train_en2_type[:,0]

        test_scope = np.load(FLAGS.data_path+'test_scope.npy')
        test_word = np.load(FLAGS.data_path+'test_word.npy')
        test_pos1 = np.load(FLAGS.data_path+'test_pos1.npy')
        test_pos2 = np.load(FLAGS.data_path+'test_pos2.npy')
        test_mask = np.load(FLAGS.data_path+'test_mask.npy')
        test_len = np.load(FLAGS.data_path+'test_len.npy')
        test_label = np.load(FLAGS.data_path+'test_label.npy')
        exclude_na_flatten_label = np.load(FLAGS.data_path+'all_true_label.npy')
        #test_en1_type = np.load(FLAGS.data_path+'test_en1_type.npy')
        #test_en2_type = np.load(FLAGS.data_path+'test_en2_type.npy')
        test_en1_type = np.load(FLAGS.data_path+'test_en1_type_nhot.npy')
        test_en2_type = np.load(FLAGS.data_path+'test_en2_type_nhot.npy')
        test_en1_word = np.load(FLAGS.data_path+'test_en1_word.npy')
        test_en2_word = np.load(FLAGS.data_path+'test_en2_word.npy')
        #test_en1_type_len = np.load(FLAGS.data_path+'test_en1_len.npy')
        #test_en2_type_len = np.load(FLAGS.data_path+'test_en2_len.npy')
        #test_word = test_word[ET_idx]
        #test_pos1 = test_pos1[ET_idx]
        #test_pos2 = test_pos2[ET_idx]
        #test_len = test_len[ET_idx]
        #test_en1_type = test_en1_type[:,0]
        #test_en2_type = test_en2_type[:,0]

        saver = tf.train.Saver(max_to_keep=40)
        train_op = self.opt.minimize(self.loss)

        best_auc,best_acc = 0,0
        not_best_count = 0
        early_stop_steps = 10

        #print(test_scope.shape)

        # test data padding
        test_scope_idx = list(range(len(test_scope)))
        test_ite = int(len(test_scope_idx)/FLAGS.batch_size)
        if test_ite * FLAGS.batch_size < len(test_scope_idx):
            test_ite += 1
            padding = test_ite * FLAGS.batch_size - len(test_scope_idx)
            test_word = np.concatenate([test_word,np.zeros((padding,FLAGS.max_len))],0)
            test_pos1 = np.concatenate([test_pos1,np.zeros((padding,FLAGS.max_len))],0)
            test_pos2 = np.concatenate([test_pos2,np.zeros((padding,FLAGS.max_len))],0)
            test_mask = np.concatenate([test_mask,np.zeros((padding,FLAGS.max_len))],0)
            test_len = np.concatenate([test_len,np.zeros((padding))],0)
            test_label = np.concatenate([test_label,np.zeros((padding))],0)
            test_en1_type = np.concatenate([test_en1_type,np.zeros((padding,39))],0)
            test_en2_type = np.concatenate([test_en2_type,np.zeros((padding,39))],0)
            test_en1_word = np.concatenate([test_en1_word,np.zeros((padding))],0)
            test_en2_word = np.concatenate([test_en2_word,np.zeros((padding))],0)
            #test_en1_type_len = np.concatenate([test_en1_type_len,np.ones((padding))],0)
            #test_en2_type_len = np.concatenate([test_en2_type_len,np.ones((padding))],0)
            for i in range(padding):
                test_scope = np.concatenate([test_scope,[[test_scope[-1][1]+1,test_scope[-1][1]+1]]],0)

        # one epoch
        sv_path = FLAGS.model_dir + 'lr_'+str(FLAGS.learning_rate)+'/'+'type_'+str(FLAGS.type_dim)+'/'
        f = open(sv_path+'config.txt','a+')
        f.write('lr_'+str(FLAGS.learning_rate)+'\t'+'type_'+str(FLAGS.type_dim)+'\n')
        for epoch in range(FLAGS.max_epoch):

            #saver.restore(self.sess, './checkpoint/MTL/lr_0.5/type_300/pcnn_MTL')
            #print('restored')

            print('###### Epoch ' + str(epoch) + ' ######')

            train_scope_idx = list(range(len(train_scope)))
            random.shuffle(train_scope_idx)

            tot, tot_not_na, tot_ner = 0.00001, 0.00001, 0.00001
            tot_correct, tot_correct_1, tot_correct_2, tot_not_na_correct, tot_ner_correct = 0,0,0,0,0
            time_sum = 0
            tot_loss, ner_tot_loss = 0, 0
            train_en1_type_tot,train_en2_type_tot,train_en1_output_tot,train_en2_output_tot,test_en1_type_tot,test_en2_type_tot,test_en1_output_tot,test_en2_output_tot = [],[],[],[],[],[],[],[]

            # one batch
            for i in range(int(len(train_scope_idx)/FLAGS.batch_size)):
            #for i in range(2):
                time_start = time.time()
                index,index_type,index_1, batch_label, batch_en1_type, batch_en2_type, batch_en1_type_len, batch_en2_type_len  = [],[],[],[],[],[],[],[]
                batch_scope = [0]
                scopes = train_scope[train_scope_idx[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]]
                for j,scope in enumerate(scopes):
                    index = index + list(range(scope[0], scope[1]+1))
                    index_1.append(scope[0])
                    batch_scope.append(batch_scope[len(batch_scope)-1] + scope[1] - scope[0] + 1)

                batch_label = train_label[index_1]
                batch_en1_type = train_en1_type[index_1]
                batch_en2_type = train_en2_type[index_1]
                feed_dict = {
                    self.scope : np.array(batch_scope),
                    self.word : train_word[index,:],
                    self.pos1 : train_pos1[index,:],
                    self.pos2 : train_pos2[index,:],
                    self.len  : train_len[index],
                    self.mask : train_mask[index,:],
                    self.label : batch_label,
                    self.en1_type : batch_en1_type,
                    self.en2_type : batch_en2_type,
                    self.en1_word : train_en1_word[index_1],
                    self.en2_word : train_en2_word[index_1],
                    #self.word_type : train_word[index_type,:],
                    #self.pos1_type : train_pos1[index_type,:],
                    #self.pos2_type : train_pos2[index_type,:],
                    #self.mask_type : train_mask[index_type,:],
                    #self.len_type : train_len[index_type],
                    #self.en1_type_len : np.array(batch_en1_type_len),
                    #self.en2_type_len : np.array(batch_en2_type_len),
                    #self.en1_type_mask : batch_en1_type_mask,
                    #self.en2_type_mask : batch_en2_type_mask,
                    #self.label_idx : self.train_label[index],
                    self.keep_prob : 0.5,
                    self.istrain : True
                }

                train_output_1,train_output_2,iter_loss,_ = self.sess.run([self.output_1,self.output_2,self.loss,train_op],feed_dict)

                tot_loss += iter_loss
                tot += batch_label.shape[0]
                tot_not_na += (batch_label != 0).sum()
                time_end = time.time()
                t = time_end - time_start
                time_sum += t

                train_en1_type_tot.append(batch_en1_type)
                train_en2_type_tot.append(batch_en2_type)
                train_en1_output_tot.append(train_output_1)
                train_en2_output_tot.append(train_output_2)

                sys.stdout.write("epoch %d step %d time %.2f | loss: %f\r"
                             % (epoch, i, t, iter_loss))
                sys.stdout.flush()

            train_en1_type_tot = np.concatenate(train_en1_type_tot, axis=0)
            train_en2_type_tot = np.concatenate(train_en2_type_tot, axis=0)
            train_en1_type_tot = np.reshape(train_en1_type_tot[:],(-1))
            train_en2_type_tot = np.reshape(train_en2_type_tot[:],(-1))
            train_en1_output_tot = np.concatenate(train_en1_output_tot, axis=0)
            train_en2_output_tot = np.concatenate(train_en2_output_tot, axis=0)
            train_en1_output_tot = np.reshape(train_en1_output_tot[:],(-1))
            train_en2_output_tot = np.reshape(train_en2_output_tot[:],(-1))
            train_en1_auc = average_precision_score(train_en1_type_tot, train_en1_output_tot)
            train_en2_auc = average_precision_score(train_en2_type_tot, train_en2_output_tot)

            s = ("epoch %d step %d time %.2f | loss: %f | en1_type auc: %f, en2_type auc: %f\r"
                             % (epoch, i, t, iter_loss,train_en1_auc,  train_en2_auc))
            print(s)
            f.write(s+'\n')
            print("\nAverage iteration time: %f" % (time_sum / i))
            not_na_acc = float(tot_not_na_correct) / tot_not_na
            #tf.summary.scalar('tot_acc', acc)
            #tf.summary.scalar('not_na_acc', not_na_acc)

            print("Testing...")
            tot_not_na, tot_correct,tot_ner,tot_correct_1,tot_correct_2 = 0.000001, 0.000001, 0.000001, 0.000001, 0.000001
            tot, tot_not_na_correct, tot_ner_correct = 0, 0 ,0
            time_sum = 0
            stack_output = []

            for i in range(test_ite):
            #for i in range(20):
                time_start = time.time()
                scopes = test_scope[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]
                index,index_type,index_1, batch_label, batch_en1_type, batch_en2_type, batch_en1_type_len, batch_en2_type_len = [],[],[],[],[],[],[],[]
                #test_idx = list(range(len(test_word)))
                #index = test_idx[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]
                batch_scope = [0]
                #batch_en1_type_mask= batch_en2_type_mask = np.zeros((FLAGS.batch_size,FLAGS.max_type_num))
                #batch_scope = []
                #cur_pos = 0

                for j,scope in enumerate(scopes):
                    index = index + list(range(scope[0], scope[1]+1))
                    index_1.append(scope[0])
                    batch_scope.append(batch_scope[len(batch_scope)-1] + scope[1] - scope[0] + 1)

                batch_label = test_label[index_1]
                batch_en1_type = test_en1_type[index_1]
                batch_en2_type = test_en2_type[index_1]

                feed_dict = {

                    self.istrain : False,
                    self.keep_prob : 1.0,

                    self.scope : np.array(batch_scope),
                    self.word : test_word[index,:],
                    self.pos1 : test_pos1[index,:],
                    self.pos2 : test_pos2[index,:],
                    self.len  : test_len[index],
                    self.mask : test_mask[index,:],
                    self.en1_type : batch_en1_type,
                    self.en2_type : batch_en2_type,
                    self.en1_word : test_en1_word[index_1],
                    self.en2_word : test_en2_word[index_1],
                    self.label : batch_label
                }

                test_output_1, test_output_2 = self.sess.run([self.output_1,self.output_2],feed_dict)

                tot += batch_label.shape[0]
                tot_not_na += (batch_label != 0).sum()
                time_end = time.time()
                t = time_end - time_start
                time_sum += t

                test_en1_type_tot.append(batch_en1_type)
                test_en2_type_tot.append(batch_en2_type)
                test_en1_output_tot.append(test_output_1)
                test_en2_output_tot.append(test_output_2)

                sys.stdout.write("[TEST] step %d |\r"
                                 % (i))
                sys.stdout.flush()

            test_en1_type_tot = np.concatenate(test_en1_type_tot, axis=0)
            test_en2_type_tot = np.concatenate(test_en2_type_tot, axis=0)
            test_en1_type_tot = np.reshape(test_en1_type_tot[:],(-1))
            test_en2_type_tot = np.reshape(test_en2_type_tot[:],(-1))
            test_en1_output_tot = np.concatenate(test_en1_output_tot, axis=0)
            test_en2_output_tot = np.concatenate(test_en2_output_tot, axis=0)
            test_en1_output_tot = np.reshape(test_en1_output_tot[:],(-1))
            test_en2_output_tot = np.reshape(test_en2_output_tot[:],(-1))
            test_en1_auc = average_precision_score(test_en1_type_tot, test_en1_output_tot)
            test_en2_auc = average_precision_score(test_en2_type_tot, test_en2_output_tot)


            s = ("[TEST] step %d | en1_type auc: %f, en1_type auc: %f\r"
                                 % (i, test_en1_auc, test_en2_auc))
            print(s)
            f.write(s+'\n')

            auc = (test_en1_auc + test_en2_auc) / 2

            if auc > best_auc:
                best_auc = auc
                print('best model , saving...')
                path = saver.save(self.sess,sv_path + FLAGS.model)

            else:
                not_best_count += 1

            if not_best_count > early_stop_steps:
                break

        print("######")
        print("Finish training " )
        print("Best epoch acc = %f" % (best_auc))


        s = str(best_auc)
        f.write(s)
        f.close()

    def _GetVar(self, init_vec, key, name, shape=None, initializer=None, trainable=True):

        if init_vec is not None and key in init_vec:
            print('using pretrained {} and is {}'.format(key, 'trainable' if trainable else 'not trainable'))
            return tf.get_variable(name = name, initializer = init_vec[key], trainable = trainable)
        else:
            print('{} initialized without pretrained'.format(key))
            return tf.get_variable(name = name, shape = shape, initializer = initializer, trainable = trainable)

