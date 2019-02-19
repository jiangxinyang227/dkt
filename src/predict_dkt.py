import tensorflow as tf

from data_process import DataGenerator
from config import Config
from train_dkt import gen_metrics, mean


def load_model(fileName):
    # 实例化配置参数对象
    config = Config()

    # 实例化数据生成对象
    dataGen = DataGenerator(fileName, config)
    dataGen.gen_attr()  # 生成训练集和测试集

    test_seqs = dataGen.test_seqs

    with tf.Session() as sess:

        accuracys = []
        aucs = []
        step = 1

        for params in dataGen.next_batch(test_seqs, "test"):
            print("step: {}".format(step))

            checkpoint_file = tf.train.latest_checkpoint("model/")
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # 获得默认的计算图结构
            graph = tf.get_default_graph()

            # 获得需要喂给模型的参数，输出的结果依赖的输入值
            input_x = graph.get_operation_by_name("test/dkt/input_x").outputs[0]
            target_id = graph.get_operation_by_name("test/dkt/target_id").outputs[0]
            keep_prob = graph.get_operation_by_name("test/dkt/keep_prob").outputs[0]
            max_steps = graph.get_operation_by_name("test/dkt/max_steps").outputs[0]
            sequence_len = graph.get_operation_by_name("test/dkt/sequence_len").outputs[0]
            batch_size = graph.get_operation_by_name("test/dkt/batch_size").outputs[0]

            # 获得输出的结果
            pred_all = graph.get_tensor_by_name("test/dkt/pred_all:0")
            pred = graph.get_tensor_by_name("test/dkt/pred:0")
            binary_pred = graph.get_tensor_by_name("test/dkt/binary_pred:0")

            target_correctness = params['target_correctness']
            pred_all, pred, binary_pred = sess.run([pred_all, pred, binary_pred],
                                                   feed_dict={input_x: params["input_x"],
                                                              target_id: params["target_id"],
                                                              keep_prob: 1.0,
                                                              max_steps: params["max_len"],
                                                              sequence_len: params["seq_len"],
                                                              batch_size: len(params["seq_len"])})

            auc, acc = gen_metrics(params["seq_len"], binary_pred, pred, target_correctness)
            print(auc, acc)
            accuracys.append(acc)
            aucs.append(auc)
            step += 1

        aucMean = mean(aucs)
        accMean = mean(accuracys)

        print("inference  auc: {}  acc: {}".format(aucMean, accMean))


if __name__ == "__main__":
    fileName = "../data/assistments.txt"
    load_model(fileName)