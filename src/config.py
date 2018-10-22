class TrainConfig(object):
    epochs = 10
    decay_rate = 0.95
    learning_rate = 0.01
    evaluate_every = 100
    checkpoint_every = 100
    max_grad_norm = 3.0


class ModelConfig(object):
    hidden_layers = [200]
    dropout_keep_prob = 0.5


class Config(object):
    batch_size = 10
    num_skills = 124
    input_size = num_skills * 2

    trainConfig = TrainConfig()
    modelConfig = ModelConfig()