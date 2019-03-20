import tensorflow as tf
import numpy as np


class TensorFlowDKT(object):
    def __init__(self, config, is_add_loss=False):

        self.is_add_loss = is_add_loss

        # 导入配置好的参数
        self.lambda_o = config.lambda_o
        self.lambda_w1 = config.lambda_w1
        self.lambda_w2 = config.lambda_w2

        self.hiddens = hiddens = config.modelConfig.hidden_layers
        self.num_skills = num_skills = config.num_skills
        self.input_size = input_size = config.input_size
        self.keep_prob_value = config.modelConfig.dropout_keep_prob

        # 定义需要喂给模型的参数
        self.max_steps = tf.placeholder(tf.int32, name="max_steps")  # 当前batch中最大序列长度
        self.input_data = tf.placeholder(tf.float32, [None, None, input_size], name="input_x")

        self.sequence_len = tf.placeholder(tf.int32, [None], name="sequence_len")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        # 下面两个输入主要是为了计算当前的预测的结果和下一道题真实结果的损失
        # [1:]序列中的题目所属知识点ID
        self.target_id = tf.placeholder(tf.int32, [None, None], name="target_id")
        # [1:]序列的做题结果，可以看作真实的y值
        self.target_correctness = tf.placeholder(tf.float32, [None, None], name="target_correctness")

        # 下面两个输入主要是为了计算当前预测的结果和当前这道题真实结果的损失
        self.source_id = tf.placeholder(tf.int32, [None, None], name="source_id")
        self.source_correctness = tf.placeholder(tf.float32, [None, None], name="source_correctness")

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
        self.pred_all = tf.sigmoid(self.mat_logits, name="pred_all")  # [batch_size, num_steps, num_skills]

        flat_logits = tf.reshape(self.logits, [-1])

        flat_target_correctness = tf.reshape(self.target_correctness, [-1])
        flat_source_correctness = tf.reshape(self.source_correctness, [-1])

        flat_base_index = tf.range(self.batch_size * self.max_steps) * num_skills

        flat_target_logits = self.get_slice_logits(self.target_id, flat_base_index, flat_logits)  # [batch_size, num_steps]
        flat_source_logits = self.get_slice_logits(self.source_id, flat_base_index, flat_logits)  # [batch_size, num_steps]

        # 对切片后的数据进行sigmoid转换
        self.pred = tf.sigmoid(tf.reshape(flat_target_logits, [-1, self.max_steps]), name="pred")
        # 将sigmoid后的值表示为0或1
        self.binary_pred = tf.cast(tf.greater_equal(self.pred, 0.5), tf.float32, name="binary_pred")

        # 定义损失函数
        with tf.name_scope("loss"):

            self.loss = self.get_cross_entropy_loss(flat_target_correctness, flat_target_logits)

            if self.is_add_loss:
                r_loss = self.get_cross_entropy_loss(flat_source_correctness, flat_source_logits)
                l1_loss, l2_loss = self.get_reg_loss()

                other_loss = self.lambda_o + r_loss + l1_loss + l2_loss

                self.loss += other_loss

    def get_slice_logits(self, ids, flat_base_index, flat_logits):
        """
        模型的返回结果是[batch_size, num_steps, num_skills]，但本质上我们是将每个时间步都看作一个二分类问题，因此
        需要获得每个时间步对应的知识点的输出，是的最后的维度是[batch_size, num_steps]
        :param ids: 知识点id，我们要根据这个id取相应的值，[batch_size, num_steps]
        :param flat_base_index: [batch_size, num_steps, num_skills]摊平的一维向量，里面的值为[0, 1, ...]
        :param flat_logits: [batch_size, num_steps, num_skills]摊平的预测结果
        :return:
        """
        # 因为flat_logits的长度为batch_size * num_steps * num_skills，
        # 我们要根据每一步的target_id将其长度变成batch_size * num_steps
        flat_base__id = tf.reshape(ids, [-1])

        flat_id = flat_base__id + flat_base_index
        # gather是从一个tensor中切片一个子集
        flat_target_logits = tf.gather(flat_logits, flat_id)

        return flat_target_logits

    def get_cross_entropy_loss(self, labels, logits):
        """
        计算交叉熵损失函数
        :param labels:
        :param logits:
        :return:
        """
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                                                      logits=logits))
        return loss

    def get_reg_loss(self):
        """
        计算正则项的损失，加入L1和L2正则项来解决序列中知识点掌握度的波动现象
        :return:
        """
        # 计算一个batch中所有的真实序列长度和
        total_num_steps = tf.cast(tf.reduce_sum(self.sequence_len), tf.float32)
        # L1正则项
        waviness_norm_l1 = tf.abs(self.mat_logits[:, 1:, :] - self.mat_logits[:, :-1, :])

        waviness_l1 = tf.reduce_sum(waviness_norm_l1) / total_num_steps / self.num_skills
        l1_loss = self.lambda_w1 * waviness_l1

        # L2正则项
        waviness_norm_l2 = tf.square(self.mat_logits[:, 1:, :] - self.mat_logits[:, :-1, :])
        waviness_l2 = tf.reduce_sum(waviness_norm_l2) / total_num_steps / self.num_skills
        l2_loss = self.lambda_w2 * waviness_l2

        return l1_loss, l2_loss
