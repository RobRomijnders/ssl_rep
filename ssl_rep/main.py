import tensorflow as tf
from datetime import datetime

from ssl_rep.utils.util import DataLoader, generate_data
from ssl_rep.model import Model

# Change this line to toggle between using SSL and not
DO_SSL = True


class Trainer:
    """
    Set up an Object for training the model, follow the training and handle the Tensorflow graph and session
    """
    def __init__(self):
        num_samples_labeled = 20  # How many labeled samples should be included
        num_samples_unlabeled = 10000  # How many unlabeled samples should be included

        # Batch size for the Stochastic Gradient descent. When doing SSL, the batch will be concatenated with the
        # unlabeled samples. So with do_ssl=True, the effective batchsize is twice as much
        self.batch_size = 32

        self.dataloader = DataLoader(*generate_data(num_samples_labeled),
                                     generate_data(num_samples_unlabeled)[0], batch_size=self.batch_size)

        # Empty variables to load on entering
        self.sess, self.model = None, None

    def __enter__(self):
        """
        Run the session within a context manager so that it gets closed properly in all cases. Especially with regards
        to the Tensorflow session.

        :return:
        """
        # Set up the session and model
        self.sess = tf.Session()
        self.model = Model(self.batch_size, do_ssl=DO_SSL)

        # Initialize the model
        self.sess.run(self.model.initializer)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sess.close()

    def train(self):
        # Set up a writer for the Tensorboard summaries
        summary_writer = tf.summary.FileWriter(logdir='log/' + datetime.now().strftime("%Y%m%d-%H%M%S") + '/')

        # Start the actual training
        num_batches = 100000  # How many batches to train on
        for num_batch in range(num_batches):
            # Do training step
            input_data, input_labels = self.dataloader.sample('train', do_ssl=True)
            loss_train, acc_train, _ = self.sess.run([self.model.loss,
                                                      self.model.accuracy,
                                                      self.model.train_op],
                                                     feed_dict={self.model.input: input_data,
                                                                self.model.labels: input_labels,
                                                                self.model.dropout_rate: 0.2})

            # Once every so many step, find the validation score and save a summary for Tensorboard
            if num_batch % 500 == 0:
                # Do Validation step
                input_data, input_labels = self.dataloader.sample('val', do_ssl=False)
                loss_val, acc_val, summary_str = self.sess.run([self.model.loss,
                                                                self.model.accuracy,
                                                                self.model.merged],
                                                               feed_dict={self.model.input: input_data,
                                                                          self.model.labels: input_labels,
                                                                          self.model.dropout_rate: 0.0})

                summary_writer.add_summary(summary_str, global_step=num_batch)
                summary_writer.flush()

                print(f'At step {num_batch:8}/{num_batches:8} we have loss {loss_train:8.3f}/{loss_val:8.3f} '
                      f'and accuracy {acc_train:8.3f}/{acc_val:8.3f}')


if __name__ == '__main__':
    with Trainer() as trainer:
        trainer.train()
