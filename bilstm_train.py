import numpy as np
import tensorflow as tf
import random,os
from pcnn import PCNN
from MTL import MTL
from Bilstm import Bilstm
from transformer import Transformer

FLAGS = tf.flags.FLAGS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# overall
tf.flags.DEFINE_string('model', 'pcnn', 'neural models to encode sentences')
tf.flags.DEFINE_string('gpu', '0', 'gpu to use')
tf.flags.DEFINE_bool('allow_growth', True, 'memory growth')
tf.flags.DEFINE_string('data_path', '../ori_data/txt_processed/', 'path to load data')
tf.flags.DEFINE_string('model_dir','./checkpoint/','path to store model')
tf.flags.DEFINE_string('summary_dir','./summary','path to store summary_dir')
tf.flags.DEFINE_integer('batch_size',160,'entity numbers used each training time')
tf.flags.DEFINE_integer('random_seed',27,'random_seed')

# training
tf.flags.DEFINE_integer('max_epoch',40,'maximum of training epochs')
tf.flags.DEFINE_integer('save_epoch',1,'frequency of training epochs')
tf.flags.DEFINE_float('weight_decay',0.00001,'l2_weight_decay')
tf.flags.DEFINE_float('keep_prob',0.5,'dropout rate')
tf.flags.DEFINE_float('learning_rate',0.2,'learning rate')

# parameters
tf.flags.DEFINE_integer('word_size', 50,'maximum of relations')
tf.flags.DEFINE_integer('hidden_size',230,'hidden feature size')
tf.flags.DEFINE_integer('pos_size',5,'position embedding size')
tf.flags.DEFINE_integer('type_num',0,'num of entity type')  # 38
tf.flags.DEFINE_integer('type_dim',0,'dim of entity type')  # 100
tf.flags.DEFINE_integer('max_type_num',5,'max_num of entity type')  # 5
tf.flags.DEFINE_integer('rnn_size',200,'hidden feature size')

# statistics
tf.flags.DEFINE_integer('max_len',120,'maximum of number of words in one sentence')
tf.flags.DEFINE_integer('pos_num', 100 * 2 + 1,'number of position embedding vectors')
tf.flags.DEFINE_integer('num_classes', 53,'maximum of relations')
tf.flags.DEFINE_integer('vocabulary_size', 114042,'maximum of relations')

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
makedirs(FLAGS.model_dir)
makedirs(FLAGS.model_dir+'lr_'+str(FLAGS.learning_rate))
if FLAGS.type_dim > 0 :
    makedirs(FLAGS.model_dir+'lr_'+str(FLAGS.learning_rate)+'/'+'type_'+str(FLAGS.type_dim))

# model
np.random.seed(FLAGS.random_seed)
random.seed(FLAGS.random_seed)
tf.set_random_seed(FLAGS.random_seed)

# init
config = tf.ConfigProto(allow_soft_placement=True,)  # allow cpu computing if there is no gpu available
config.gpu_options.allow_growth = FLAGS.allow_growth
sess = tf.Session(config=config)
#model = PCNN(sess)
#model = MTL(sess)
model = Bilstm(sess)
model.train()
