import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, GaussianNoise


class Model:
    def __init__(self, labeled_batch_size, do_ssl=True):
        """
        Sets up the Tensorflow Graph for the Mean Teacher method

        In case you don't want to do Semi Supervised learning, set do_ssl=False. In that case, consistency_coeff_set
        will be 0., so that the consisentency loss of the Mean Teacher method is not used during training.

        :param labeled_batch_size:
        :param do_ssl:
        """
        # Some hyperparameters
        ramp_end_step = 2000.
        consistency_coeff_set = 1E-2 if do_ssl else 0.  # 1E-1
        l2_loss_coeff = 1E-5  # 1E-5
        learning_rate = 1E-3  # 1E-3

        # input placeholders
        self.input = tf.placeholder(dtype=tf.float32, shape=[None, 2], name="inputs")
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None], name="labels")
        self.dropout_rate = tf.placeholder(dtype=tf.float32, name="dropout_rate")

        global_step = tf.Variable(0, trainable=False)

        def representation_layers(scope_name, input_tensor):
            # Generate the representation learner in two different variable scopes. This simplifies the subsequent
            # routing of gradients.
            with tf.variable_scope(scope_name):
                x1 = Dropout(self.dropout_rate)(Dense(32, activation=tf.nn.selu)(input_tensor))
                x2 = Dropout(self.dropout_rate)(Dense(32, activation=tf.nn.selu)(x1))
                x3 = Dropout(self.dropout_rate)(Dense(2, activation=tf.nn.selu)(x2))
                return x3

        # The student learns the representations. The teacher only copies the student, so we call stop_gradient()
        self.representation_student = representation_layers('student', self.input)
        self.representation_teacher = tf.stop_gradient(
            representation_layers('teacher', self.noisy(self.input, 0.05)))

        # Predictions are based on only the representations from the student
        logits = tf.squeeze(Dense(1)(self.representation_student))

        self.predictions = tf.nn.sigmoid(logits)

        # TODO (rob) this paper uses EMA coefficient of 0.95 ??
        #  https://colinraffel.com/publications/nips2018realistic.pdf
        ema = tf.train.ExponentialMovingAverage(0.9)
        tf.add_to_collection('ema_op', ema.apply(tf.trainable_variables('student')))

        # We maintain an EMA of the student parameters. In the next lines we bind the teacher to the student
        # On each call to the teacher variables, the tf.assign() gets called which "copies" the EMA's of each parameter
        # to the corresponding variable in the teacher
        for var_student, var_teacher in zip(tf.trainable_variables('student'), tf.trainable_variables('teacher')):
            tf.add_to_collection(var_teacher.op.name, tf.assign(var_teacher, ema.average(var_student)))

        # We ramp up the coefficient for the consistency loss in the first 500 steps.
        # According to Appendix C in https://colinraffel.com/publications/nips2018realistic.pdf
        consistency_coeff = tf.clip_by_value(consistency_coeff_set / ramp_end_step *
                                             tf.cast(global_step, tf.float32),
                                             0.,
                                             consistency_coeff_set)

        # Calculate our three losses:
        # 1. The consistency loss
        const_loss = self.consistency_loss(self.representation_student, self.representation_teacher)

        # 2. The classification loss
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.labels[:labeled_batch_size], logits=logits[:labeled_batch_size]))

        # 3. The regularization loss
        l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])

        # Sum up all three losses
        total_loss = (self.loss + l2_loss_coeff * l2_loss +
                      consistency_coeff * const_loss)

        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(self.labels[:labeled_batch_size],
                             tf.cast(tf.greater_equal(logits[:labeled_batch_size], 0.), tf.float32)), tf.float32))

        # #####################################################################
        # Optimizing the network
        lr = tf.maximum(tf.train.exponential_decay(learning_rate, global_step,
                                                   5000, 0.1, staircase=False), learning_rate / 100)

        optimizer = tf.train.AdamOptimizer(lr)
        tvars = tf.trainable_variables()

        grads, _ = tf.clip_by_global_norm(tf.gradients(total_loss, tvars),
                                          1.5)  # clip the gradients to prevent explosion
        self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)

        self.initializer = tf.global_variables_initializer()

        self.saver = tf.train.Saver()

        # ##########################################################
        # Tensorboard Magic
        # Log stuff to Tensorboard
        tf.summary.scalar('Learning_rate', lr)
        tf.summary.scalar('Consistency_coeff', consistency_coeff)

        # Put losses to Tensorboard
        tf.summary.scalar('Consistency_loss', const_loss, family='Losses')
        tf.summary.scalar('Regularization_loss', l2_loss, family='Losses')
        tf.summary.scalar('Classification_loss', self.loss, family='Losses')

        # Put other metrics to Tensorboard
        tf.summary.scalar('Accuracy', self.accuracy, family='Performance')

        self.merged = tf.summary.merge_all()

    @staticmethod
    def consistency_loss(data1, data2):
        """
        Consistency loss is the Mean Squared Distance between the two representations.

        Coloquial interpretation:
        "The consistency loss seems to pull together the representations from teacher and student*
        :param data1:
        :param data2:
        :return:
        """
        return tf.reduce_mean(tf.reduce_sum(tf.square((data1 - data2)), axis=-1))

    @staticmethod
    def noisy(tensor, stddev):
        noise_layer = GaussianNoise(stddev)
        return noise_layer(tensor)
