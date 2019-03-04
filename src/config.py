class TrainConfig(object):
    epochs = 5
    decay_rate = 0.92
    learning_rate = 0.01
    evaluate_every = 100
    checkpoint_every = 100
    max_grad_norm = 3.0


class ModelConfig(object):
    hidden_layers = [200]
    dropout_keep_prob = 0.6


class Config(object):
    batch_size = 10
    num_skills = 267  # 训练集所包含的知识点的数量，针对自己的数据集，需要修改这个值
    input_size = num_skills * 2

    trainConfig = TrainConfig()
    modelConfig = ModelConfig()
