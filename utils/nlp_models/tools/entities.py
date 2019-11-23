# --*-- coding: utf-8 --*--


class TrainOptions(object):
    def __init__(self, max_epoch=1, batch_size=128, check_steps=1, patient=3, dropout_keep_prob=1.0, over_sample=None):
        # maximum epoch during training
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        # early stop config. every check_steps to calculate the criterion
        self.check_steps = check_steps
        # early stop config. patient times not improved, stop.
        self.patient = patient
        self.dropout_keep_prob = dropout_keep_prob
        self.over_sample = over_sample