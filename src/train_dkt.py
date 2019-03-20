import time
import datetime
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, accuracy_score

from model import TensorFlowDKT
from data_process import DataGenerator
from config import Config


def mean(item):
    return sum(item) / len(item)


def gen_metrics(sequence_len, binary_pred, pred, target_correctness):
    """
    生成auc和accuracy的指标值
    :param sequence_len: 每一个batch中各序列的长度组成的列表
    :param binary_pred:
    :param pred:
    :param target_correctness:
    :return:
    """
    binary_preds = []
    preds = []
    target_correctnesses = []
    for seq_idx, seq_len in enumerate(sequence_len):
        binary_preds.append(binary_pred[seq_idx, :seq_len])
        preds.append(pred[seq_idx, :seq_len])
        target_correctnesses.append(target_correctness[seq_idx, :seq_len])

    new_binary_pred = np.concatenate(binary_preds)
    new_pred = np.concatenate(preds)
    new_target_correctness = np.concatenate(target_correctnesses)

    auc = roc_auc_score(new_target_correctness, new_pred)
    accuracy = accuracy_score(new_target_correctness, new_binary_pred)

    return auc, accuracy


class DKTEngine(object):

    def __init__(self):
        self.config = Config()
        self.train_dkt = None
        self.test_dkt = None
        self.sess = None
        self.global_step = 0

    def add_gradient_noise(self, grad, stddev=1e-3, name=None):
        """
        Adds gradient noise as described in http://arxiv.org/abs/1511.06807 [2].
        """
        with tf.op_scope([grad, stddev], name, "add_gradient_noise") as name:
            grad = tf.convert_to_tensor(grad, name="grad")
            gn = tf.random_normal(tf.shape(grad), stddev=stddev)
            return tf.add(grad, gn, name=name)

    def train_step(self, params, train_op, train_summary_op, train_summary_writer):
        """
        A single training step
        """
        dkt = self.train_dkt
        sess = self.sess
        global_step = self.global_step

        feed_dict = {dkt.input_data: params['input_x'],
                     dkt.target_id: params['target_id'],
                     dkt.target_correctness: params['target_correctness'],
                     dkt.source_id: params["source_id"],
                     dkt.source_correctness: params["source_correctness"],
                     dkt.max_steps: params['max_len'],
                     dkt.sequence_len: params['seq_len'],
                     dkt.keep_prob: self.config.modelConfig.dropout_keep_prob,
                     dkt.batch_size: self.config.batch_size}

        _, step, summaries, loss, binary_pred, pred, target_correctness = sess.run(
            [train_op, global_step, train_summary_op, dkt.loss, dkt.binary_pred, dkt.pred, dkt.target_correctness],
            feed_dict)

        auc, accuracy = gen_metrics(params['seq_len'], binary_pred, pred, target_correctness)

        time_str = datetime.datetime.now().isoformat()
        print("train: {}: step {}, loss {}, acc {}, auc: {}".format(time_str, step, loss, accuracy, auc))
        train_summary_writer.add_summary(summaries, step)

    def dev_step(self, params, dev_summary_op, writer=None):
        """
        Evaluates model on a dev set
        """
        dkt = self.test_dkt
        sess = self.sess
        global_step = self.global_step

        feed_dict = {dkt.input_data: params['input_x'],
                     dkt.target_id: params['target_id'],
                     dkt.target_correctness: params['target_correctness'],
                     dkt.source_id: params["source_id"],
                     dkt.source_correctness: params["source_correctness"],
                     dkt.max_steps: params['max_len'],
                     dkt.sequence_len: params['seq_len'],
                     dkt.keep_prob: 1.0,
                     dkt.batch_size: len(params["seq_len"])}
        step, summaries, loss, pred, binary_pred, target_correctness = sess.run(
            [global_step, dev_summary_op, dkt.loss, dkt.pred, dkt.binary_pred, dkt.target_correctness],
            feed_dict)

        auc, accuracy = gen_metrics(params['seq_len'], binary_pred, pred, target_correctness)
        # precision, recall, f_score = precision_recall_fscore_support(target_correctness, binary_pred)

        if writer:
            writer.add_summary(summaries, step)

        return loss, accuracy, auc

    def run_epoch(self, fileName):
        """
        训练模型
        :param filePath:
        :return:
        """

        # 实例化配置参数对象
        config = Config()

        # 实例化数据生成对象
        dataGen = DataGenerator(fileName, config)
        dataGen.gen_attr()  # 生成训练集和测试集

        train_seqs = dataGen.train_seqs
        test_seqs = dataGen.test_seqs

        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False
        )
        sess = tf.Session(config=session_conf)
        self.sess = sess

        with sess.as_default():
            # 实例化dkt模型对象
            with tf.name_scope("train"):
                with tf.variable_scope("dkt", reuse=None):
                    train_dkt = TensorFlowDKT(config, is_add_loss=config.is_add_loss)

            with tf.name_scope("test"):
                with tf.variable_scope("dkt", reuse=True):
                    test_dkt = TensorFlowDKT(config, is_add_loss=config.is_add_loss)

            self.train_dkt = train_dkt
            self.test_dkt = test_dkt

            global_step = tf.Variable(0, name="global_step", trainable=False)
            self.global_step = global_step

            # 定义一个优化器
            optimizer = tf.train.AdamOptimizer(config.trainConfig.learning_rate)
            grads_and_vars = optimizer.compute_gradients(train_dkt.loss)

            # 对梯度进行截断，并且加上梯度噪音
            grads_and_vars = [(tf.clip_by_norm(g, config.trainConfig.max_grad_norm), v)
                              for g, v in grads_and_vars if g is not None]
            # grads_and_vars = [(self.add_gradient_noise(g), v) for g, v in grads_and_vars]

            # 定义图中最后的节点
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step, name="train_op")

            # 保存各种变量或结果的值
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("writing to {}".format(out_dir))

            # 训练时的 Summaries
            train_loss_summary = tf.summary.scalar("loss", train_dkt.loss)
            train_summary_op = tf.summary.merge([train_loss_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # 测试时的 summaries
            test_loss_summary = tf.summary.scalar("loss", test_dkt.loss)
            dev_summary_op = tf.summary.merge([test_loss_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            saver = tf.train.Saver(tf.global_variables())

            sess.run(tf.global_variables_initializer())

            print("初始化完毕，开始训练")
            for i in range(config.trainConfig.epochs):
                np.random.shuffle(train_seqs)
                for params in dataGen.next_batch(train_seqs, "train"):
                    # 批次获得训练集，训练模型
                    self.train_step(params, train_op, train_summary_op, train_summary_writer)

                    current_step = tf.train.global_step(sess, global_step)
                    # train_step.run(feed_dict={x: batch_train[0], y_actual: batch_train[1], keep_prob: 0.5})
                    # 对结果进行记录
                    if current_step % config.trainConfig.evaluate_every == 0:
                        print("\nEvaluation:")
                        # 获得测试数据

                        losses = []
                        accuracys = []
                        aucs = []
                        for params in dataGen.next_batch(test_seqs, "test"):
                            loss, accuracy, auc = self.dev_step(params, dev_summary_op, writer=None)
                            losses.append(loss)
                            accuracys.append(accuracy)
                            aucs.append(auc)

                        time_str = datetime.datetime.now().isoformat()
                        print("dev: {}, step: {}, loss: {}, acc: {}, auc: {}".
                              format(time_str, current_step, mean(losses), mean(accuracys), mean(aucs)))

                    # 保存为checkpoint模型文件
                    if current_step % config.trainConfig.checkpoint_every == 0:
                        path = saver.save(sess, "model/add_loss/my-model", global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))

            # 保存为pb模型文件
            # builder = tf.saved_model.builder.SavedModelBuilder("./sevenSkillModel")
            # inputs = {"input_x": tf.saved_model.utils.build_tensor_info(self.train_dkt.input_data),
            #           "target_id": tf.saved_model.utils.build_tensor_info(self.train_dkt.target_id),
            #           "max_steps": tf.saved_model.utils.build_tensor_info(self.train_dkt.max_steps),
            #           "sequence_len": tf.saved_model.utils.build_tensor_info(self.train_dkt.sequence_len),
            #           "keep_prob": tf.saved_model.utils.build_tensor_info(self.train_dkt.keep_prob),
            #           "batch_size": tf.saved_model.utils.build_tensor_info(self.train_dkt.batch_size)}
            #
            # outputs = {"pred_all": tf.saved_model.utils.build_tensor_info(self.train_dkt.pred_all)}
            #
            # prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(inputs=inputs,
            #                                                                               outputs=outputs,
            #                                                                               method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
            # legacy_init_op = tf.group(tf.tables_initializer(), name="legacy_init_op")
            # builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
            #                                      signature_def_map={"predict": prediction_signature},
            #                                      legacy_init_op=legacy_init_op)
            #
            # builder.save()


if __name__ == "__main__":
    fileName = "../data/assistments.txt"
    dktEngine = DKTEngine()
    dktEngine.run_epoch(fileName)

