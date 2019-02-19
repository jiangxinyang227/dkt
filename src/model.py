import tensorflow as tf
import numpy as np


class TensorFlowDKT(object):
    def __init__(self, config):
        # 导入配置好的参数
        self.hiddens = hiddens = config.modelConfig.hidden_layers
        self.num_skills = num_skills = config.num_skills
        self.input_size = input_size = config.input_size
        self.keep_prob_value = config.modelConfig.dropout_keep_prob

        # 定义需要喂给模型的参数
        self.max_steps = tf.placeholder(tf.int32, name="max_steps")  # 当前batch中最大序列长度
        self.input_data = tf.placeholder(tf.float32, [None, None, input_size], name="input_x")

        self.sequence_len = tf.placeholder(tf.int32, [None], name="sequence_len")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        self.target_id = tf.placeholder(tf.int32, [None, None], name="target_id")
        self.target_correctness = tf.placeholder(tf.float32, [None, None], name="target_correctness")

        self.batch_size = tf.placeholder(tf.int32, name="batch_size")

        self.flat_target_correctness = None

        # 构建lstm模型结构
        hidden_layers = []
        for idx, hidden_size in enumerate(hiddens):
            lstm_layer = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True)
            hidden_layer = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_layer,
                                                         output_keep_prob=self.keep_prob)
            hidden_layers.append(hidden_layer)
        self.hidden_cell = tf.nn.rnn_cell.MultiRNNCell(cells=hidden_layers, state_is_tuple=True)

        # 采用动态rnn，动态输入序列的长度
        outputs, self.current_state = tf.nn.dynamic_rnn(cell=self.hidden_cell,
                                                        inputs=self.input_data,
                                                        sequence_length=self.sequence_len,
                                                        dtype=tf.float32)

        # 隐层到输出层的权重系数[最后隐层的神经元数量，知识点数]
        output_w = tf.get_variable("W", [hiddens[-1], num_skills])
        output_b = tf.get_variable("b", [num_skills])

        self.output = tf.reshape(outputs, [-1, hiddens[-1]])
        # 因为权值共享的原因，对生成的矩阵[batch_size * self.max_steps, num_skills]中的每一行都加上b
        self.logits = tf.matmul(self.output, output_w) + output_b

        self.mat_logits = tf.reshape(self.logits, [-1, self.max_steps, num_skills])

        # 对每个batch中每个序列中的每个时间点的输出中的每个值进行sigmoid计算，这里的值表示对某个知识点的掌握情况，
        # 每个时间点都会输出对所有知识点的掌握情况
        self.pred_all = tf.sigmoid(self.mat_logits, name="pred_all")

        # 计算损失loss
        flat_logits = tf.reshape(self.logits, [-1])

        flat_target_correctness = tf.reshape(self.target_correctness, [-1])
        self.flat_target_correctness = flat_target_correctness

        flat_base_target_index = tf.range(self.batch_size * self.max_steps) * num_skills

        # 因为flat_logits的长度为batch_size * num_steps * num_skills，我们要根据每一步的target_id将其长度变成batch_size * num_steps
        flat_base_target_id = tf.reshape(self.target_id, [-1])

        flat_target_id = flat_base_target_id + flat_base_target_index
        # gather是从一个tensor中切片一个子集
        flat_target_logits = tf.gather(flat_logits, flat_target_id)

        # 对切片后的数据进行sigmoid转换
        self.pred = tf.sigmoid(tf.reshape(flat_target_logits, [-1, self.max_steps]), name="pred")
        # 将sigmoid后的值表示为0或1
        self.binary_pred = tf.cast(tf.greater_equal(self.pred, 0.5), tf.float32, name="binary_pred")

        # 定义损失函数
        with tf.name_scope("loss"):
            # flat_target_logits_sigmoid = tf.nn.log_softmax(flat_target_logits)
            # self.loss = -tf.reduce_mean(flat_target_correctness * flat_target_logits_sigmoid)
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=flat_target_correctness,
                                                                               logits=flat_target_logits))


