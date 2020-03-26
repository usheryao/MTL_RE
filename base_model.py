import tensorflow as tf
import pickle
import numpy as np
import random,time,sys
from sklearn.metrics import average_precision_score

FLAGS = tf.flags.FLAGS

class base_model():

    def __init__(self,sess,optimizer=tf.train.GradientDescentOptimizer):

        self.sess = sess
        if FLAGS.model[:4] == 'pcnn':
            self.hidden_size = FLAGS.hidden_size * 3
        else:
            self.hidden_size = FLAGS.hidden_size

        self.place_holder()

        init_file = '../raw_HNRE/data/initial_vectors/init_vec_pcnn'
        init_vec = pickle.load(open(init_file, 'rb'))
        self.bulid(init_vec)
        self.sess.run(tf.global_variables_initializer())
        self.opt = optimizer(FLAGS.learning_rate)

    def place_holder(self):

        self.global_step = tf.Variable(0,trainable=False,name='global_step')
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
        #self.en1_type = tf.placeholder(dtype=tf.int32, shape=[FLAGS.batch_size, 5], name='en1_type')
        #self.en2_type = tf.placeholder(dtype=tf.int32, shape=[FLAGS.batch_size, 5], name='en2_type')
        self.en1_type = tf.placeholder(dtype=tf.int32, shape=[FLAGS.batch_size], name='en1_type')
        self.en2_type = tf.placeholder(dtype=tf.int32, shape=[FLAGS.batch_size], name='en2_type')
        #self.en1_type_len = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size], name='en1_len')
        ##self.en2_type_len = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size], name='en2_len')
        self.en1_type_mask = tf.placeholder(dtype=tf.int32, shape=[FLAGS.batch_size, 5], name='en1_type_mask')
        self.en2_type_mask = tf.placeholder(dtype=tf.int32, shape=[FLAGS.batch_size, 5], name='en2_type_mask')
        #self.en1_type = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.max_type_num], name='en1_type')
        #self.en2_type = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.max_type_num], name='en2_type')
        #self.en1_type_len = tf.placeholder(dtype=tf.float32, shape=[None], name='en1_len')
        #self.en2_type_len = tf.placeholder(dtype=tf.float32, shape=[None], name='en2_len')

    def bulid(self,init_vec):

        raise NotImplementedError("Each Model must re-implement this method.")

    def train(self):

        #data_path = '../raw_HNRE/data/'
        train_scope = np.load(FLAGS.data_path+'train_scope.npy')
        train_word = np.load(FLAGS.data_path+'train_word.npy')
        train_pos1 = np.load(FLAGS.data_path+'train_pos1.npy')
        train_pos2 = np.load(FLAGS.data_path+'train_pos2.npy')
        train_mask = np.load(FLAGS.data_path+'train_mask.npy')
        train_len = np.load(FLAGS.data_path+'train_len.npy')
        train_label = np.load(FLAGS.data_path+'train_label.npy')
        train_en1_type = np.load(FLAGS.data_path+'train_en1_type.npy')
        train_en2_type = np.load(FLAGS.data_path+'train_en2_type.npy')
        train_en1_type_len = np.load(FLAGS.data_path+'train_en1_len.npy')
        train_en2_type_len = np.load(FLAGS.data_path+'train_en2_len.npy')

        test_scope = np.load(FLAGS.data_path+'test_scope.npy')
        test_word = np.load(FLAGS.data_path+'test_word.npy')
        test_pos1 = np.load(FLAGS.data_path+'test_pos1.npy')
        test_pos2 = np.load(FLAGS.data_path+'test_pos2.npy')
        test_mask = np.load(FLAGS.data_path+'test_mask.npy')
        test_len = np.load(FLAGS.data_path+'test_len.npy')
        test_label = np.load(FLAGS.data_path+'test_label.npy')
        exclude_na_flatten_label = np.load(FLAGS.data_path+'all_true_label.npy')
        test_en1_type = np.load(FLAGS.data_path+'test_en1_type.npy')
        test_en2_type = np.load(FLAGS.data_path+'test_en2_type.npy')
        test_en1_type_len = np.load(FLAGS.data_path+'test_en1_len.npy')
        test_en2_type_len = np.load(FLAGS.data_path+'test_en2_len.npy')

        saver = tf.train.Saver(max_to_keep=40)
        train_op = self.opt.minimize(self.loss)

        best_auc = 0
        not_best_count = 0
        early_stop_steps = 10

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
            test_en1_type = np.concatenate([test_en1_type,np.zeros((padding,FLAGS.max_type_num))],0)
            test_en2_type = np.concatenate([test_en2_type,np.zeros((padding,FLAGS.max_type_num))],0)
            test_en1_type_len = np.concatenate([test_en1_type_len,np.ones((padding))],0)
            test_en2_type_len = np.concatenate([test_en2_type_len,np.ones((padding))],0)

            for i in range(padding):
                test_scope = np.concatenate([test_scope,[[test_scope[-1][1]+1,test_scope[-1][1]+1]]],0)

        # one epoch
        for epoch in range(FLAGS.max_epoch):

            print('###### Epoch ' + str(epoch) + ' ######')

            train_scope_idx = list(range(len(train_scope)))
            random.shuffle(train_scope_idx)

            tot, tot_not_na, tot_ner = 0.00001, 0.00001, 0.00001
            tot_correct, tot_not_na_correct, tot_ner_correct = 0, 0, 0
            time_sum = 0
            tot_loss, ner_tot_loss = 0, 0

            # one batch
            for i in range(int(len(train_scope_idx)/FLAGS.batch_size)):
            #for i in range(2):
                time_start = time.time()
                index, batch_label, batch_en1_type, batch_en2_type, batch_en1_type_len, batch_en2_type_len  = [],[],[],[],[],[]
                batch_scope = [0]
                batch_en1_type_mask= batch_en2_type_mask = np.zeros((FLAGS.batch_size,FLAGS.max_type_num))
                #batch_scope = []
                #cur_pos = 0
                scopes = train_scope[train_scope_idx[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]]
                for j,scope in enumerate(scopes):
                    index = index + list(range(scope[0], scope[1]+1))
                    batch_label.append(train_label[scope[0]])
                    #batch_en1_type.append(train_en1_type[scope[0]])
                    #batch_en2_type.append(train_en2_type[scope[0]])
                    batch_en1_type.append(train_en1_type[scope[0]][0])
                    batch_en2_type.append(train_en2_type[scope[0]][0])
                    #batch_en1_type_len.append(train_en1_type_len[scope[0]])
                    #batch_en2_type_len.append(train_en2_type_len[scope[0]])
                    batch_scope.append(batch_scope[len(batch_scope)-1] + scope[1] - scope[0] + 1)
                    for k in range(int(train_en1_type_len[scope[0]])):
                        batch_en1_type_mask[j][k] = 1
                    for k in range(int(train_en2_type_len[scope[0]])):
                        batch_en2_type_mask[j][k] = 1
                    #batch_scope.append([cur_pos, cur_pos + scope[1] - scope[0] + 1])
                    #cur_pos += scope[1] - scope[0] + 1


                batch_label = np.array(batch_label)

                feed_dict = {
                    self.scope : np.array(batch_scope),
                    self.word : train_word[index,:],
                    self.pos1 : train_pos1[index,:],
                    self.pos2 : train_pos2[index,:],
                    self.mask : train_mask[index,:],
                    self.label : batch_label,
                    self.en1_type : np.array(batch_en1_type),
                    self.en2_type : np.array(batch_en2_type),
                    #self.en1_type_len : np.array(batch_en1_type_len),
                    #self.en2_type_len : np.array(batch_en2_type_len),
                    self.en1_type_mask : batch_en1_type_mask,
                    self.en2_type_mask : batch_en2_type_mask,
                    #self.label_idx : self.train_label[index],
                    self.keep_prob : 0.5,
                    self.istrain : True
                }

                train_output,iter_loss,_ = self.sess.run([self.output,self.loss,train_op],feed_dict)

                tot_loss += iter_loss
                iter_output = train_output.argmax(-1)
                iter_correct = (iter_output == batch_label).sum()
                iter_not_na_correct = np.logical_and(iter_output == batch_label, batch_label != 0).sum()
                tot_correct += iter_correct
                tot_not_na_correct += iter_not_na_correct
                tot += batch_label.shape[0]
                tot_not_na += (batch_label != 0).sum()
                time_end = time.time()
                t = time_end - time_start
                time_sum += t

                sys.stdout.write("epoch %d step %d time %.2f | loss: %f, not NA accuracy: %f, accuracy: %f\r"
                             % (epoch, i, t, iter_loss, float(tot_not_na_correct) / tot_not_na, float(tot_correct) / tot))
                sys.stdout.flush()

            print("\nAverage iteration time: %f" % (time_sum / i))
            acc = float(tot_correct) / tot
            not_na_acc = float(tot_not_na_correct) / tot_not_na
            tf.summary.scalar('tot_acc', acc)
            tf.summary.scalar('not_na_acc', not_na_acc)

            print("Testing...")
            tot_not_na, tot_correct,tot_ner = 0.000001, 0.000001, 0.000001
            tot, tot_not_na_correct, tot_ner_correct = 0, 0 ,0
            time_sum = 0
            stack_output = []

            for i in range(test_ite):

                time_start = time.time()
                scopes = test_scope[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]
                index, batch_label, batch_en1_type, batch_en2_type, batch_en1_type_len, batch_en2_type_len = [],[],[],[],[],[]
                batch_scope = [0]
                batch_en1_type_mask= batch_en2_type_mask = np.zeros((FLAGS.batch_size,FLAGS.max_type_num))
                #batch_scope = []
                #cur_pos = 0

                for j,scope in enumerate(scopes):
                    index = index + list(range(scope[0], scope[1]+1))
                    batch_label.append(test_label[scope[0]])
                    #batch_en1_type.append(test_en1_type[scope[0]])
                    #batch_en2_type.append(test_en2_type[scope[0]])
                    batch_en1_type.append(test_en1_type[scope[0]][0])
                    batch_en2_type.append(test_en2_type[scope[0]][0])
                    #batch_en1_type_len.append(test_en1_type_len[scope[0]])
                    #batch_en2_type_len.append(test_en2_type_len[scope[0]])
                    batch_scope.append(batch_scope[len(batch_scope)-1] + scope[1] - scope[0] + 1)
                    for k in range(int(train_en1_type_len[scope[0]])):
                        batch_en1_type_mask[j][k] = 1
                    for k in range(int(train_en2_type_len[scope[0]])):
                        batch_en2_type_mask[j][k] = 1
                    #batch_scope.append([cur_pos, cur_pos + scope[1] - scope[0] + 1])
                    #cur_pos += scope[1] - scope[0] + 1

                batch_label = np.array(batch_label)
                feed_dict = {

                    self.istrain : False,
                    self.keep_prob : 1.0,

                    self.scope : np.array(batch_scope),
                    self.word : test_word[index,:],
                    self.pos1 : test_pos1[index,:],
                    self.pos2 : test_pos2[index,:],
                    self.mask : test_mask[index,:],
                    self.en1_type : np.array(batch_en1_type),
                    self.en2_type : np.array(batch_en2_type),
                    #self.en1_type_len : np.array(batch_en1_type_len),
                    #self.en2_type_len : np.array(batch_en2_type_len),
                    self.en1_type_mask : batch_en1_type_mask,
                    self.en2_type_mask : batch_en2_type_mask,
                    self.label : batch_label
                }

                test_output = self.sess.run([self.output],feed_dict)[0]
                stack_output.append(test_output)

                iter_output = test_output.argmax(-1)
                iter_correct = (iter_output == batch_label).sum()
                iter_not_na_correct = np.logical_and(iter_output == batch_label, batch_label != 0).sum()
                tot_correct += iter_correct
                tot_not_na_correct += iter_not_na_correct
                tot += batch_label.shape[0]
                tot_not_na += (batch_label != 0).sum()
                time_end = time.time()
                t = time_end - time_start
                time_sum += t
                sys.stdout.write("[TEST] step %d | not NA accuracy: %f, accuracy: %f\r" % (i, float(tot_not_na_correct) / tot_not_na, float(tot_correct) / tot))
                sys.stdout.flush()

            stack_output = np.concatenate(stack_output, axis=0)
            exclude_na_flatten_output = np.reshape(stack_output[:,1:],(-1))
            exclude_na_flatten_output = exclude_na_flatten_output[:len(exclude_na_flatten_label)]

            m = average_precision_score(exclude_na_flatten_label, exclude_na_flatten_output)
            print()
            print(m)
            if m > best_auc:
                best_auc = m
                print('best model , saving...')
                sv_path = FLAGS.model_dir + 'lr_'+str(FLAGS.learning_rate)+'/'+'type_'+str(FLAGS.type_dim)+'/'
                #path = saver.save(self.sess,sv_path + FLAGS.model)
            else:
                not_best_count += 1

            if not_best_count > early_stop_steps:
                break

        print("######")
        print("Finish training " )
        print("Best epoch auc = %f" % (best_auc))
        #f = open(sv_path+'config.txt','w')
        #f.write('lr_'+str(FLAGS.learning_rate)+'\t'+'type_'+str(FLAGS.type_dim)+'\n')
        #s = str(m)
        #f.write(s)
        #f.close()

    def _GetVar(self, init_vec, key, name, shape=None, initializer=None, trainable=True):

        if init_vec is not None and key in init_vec:
            print('using pretrained {} and is {}'.format(key, 'trainable' if trainable else 'not trainable'))
            return tf.get_variable(name = name, initializer = init_vec[key], trainable = trainable)
        else:
            print('{} initialized without pretrained'.format(key))
            return tf.get_variable(name = name, shape = shape, initializer = initializer, trainable = trainable)
