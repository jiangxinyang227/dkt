class TrainConfig(object):
    epochs = 5
    learning_rate = 0.01
    evaluate_every = 100
    checkpoint_every = 100
    max_grad_norm = 3.0


class ModelConfig(object):
    hidden_layers = [512]
    dropout_keep_prob = 0.6


class Config(object):
    batch_size = 16
    num_skills = 267  # 训练集所包含的知识点的数量，针对自己的数据集，需要修改这个值
    input_size = num_skills * 2

    # 增加的三个损失的惩罚系数
    lambda_o = 0.1
    lambda_w1 = 0.003
    lambda_w2 = 3.0

    # 是否添加三个损失来控制序列中的输出结果的稳定性
    is_add_loss = True

    trainConfig = TrainConfig()
    modelConfig = ModelConfig()
