import time

import tensorflow as tf

import project_config as config
from model.attention_model import AttentionModel


class TrainModel:
    def __init__(self, container):
        self.container = container
        self.model = None
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        self.ckpt_manager = None
        self.start_epoch = None
        self.loss_plot = []

    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)

    def create_model(self):
        self.model = AttentionModel(embedding_dim=config.EMBEDDING_DIM, units=config.UNITS,
                                    vocab_size=config.VOCAB_SIZE)

    def checkpoint(self):
        ckpt = tf.train.Checkpoint(encoder=self.model.encoder,
                                   decoder=self.model.decoder,
                                   optimizer=self.optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(ckpt, config.CHECKPOINT_PATH, max_to_keep=5)
        self.start_epoch = 0
        if self.ckpt_manager.latest_checkpoint:
            self.start_epoch = int(self.ckpt_manager.latest_checkpoint.split('-')[-1])
            # restoring the latest checkpoint in checkpoint_path
            ckpt.restore(self.ckpt_manager.latest_checkpoint)

    @tf.function
    def train_step(self, img_tensor, target):
        loss = 0
        # initializing the hidden state for each batch
        # because the captions are not related from image to image
        hidden = self.model.decoder.reset_state(batch_size=target.shape[0])

        dec_input = tf.expand_dims([self.container.TOKENIZER.word_index['<start>']] * target.shape[0], 1)
        with tf.GradientTape() as tape:
            features = self.model.encoder(img_tensor)
            for i in range(1, target.shape[1]):
                # passing the features through the decoder
                predictions, hidden, _ = self.model.decoder(dec_input, features, hidden)
                loss += self.loss_function(target[:, i], predictions)
                # using teacher forcing
                dec_input = tf.expand_dims(target[:, i], 1)

        total_loss = (loss / int(target.shape[1]))

        trainable_variables = self.model.encoder.trainable_variables + self.model.decoder.trainable_variables

        gradients = tape.gradient(loss, trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        return loss, total_loss

    def train_model(self):
        for epoch in range(self.start_epoch, config.EPOCHS):
            start = time.time()
            total_loss = 0
            for (batch, (img_tensor, target)) in enumerate(self.container.TRAIN_DATASET):
                batch_loss, t_loss = self.train_step(img_tensor, target)
                total_loss += t_loss

                if batch % 100 == 0:
                    average_batch_loss = batch_loss.numpy() / int(target.shape[1])
                    print(f'Epoch {epoch + 1} Batch {batch} Loss {average_batch_loss:.4f}')
            # storing the epoch end loss value to plot later
            self.loss_plot.append(total_loss / self.container.NUM_STEPS)

            self.ckpt_manager.save()

            print(f'Epoch {epoch + 1} Loss {total_loss / self.container.NUM_STEPS:.6f}')
            print(f'Time taken for 1 epoch {time.time() - start:.2f} sec\n')

    def push_data_to_container(self):
        self.container.ATTENTION_MODEL = self.model

    def execute(self):
        print("Training model...")
        self.create_model()
        self.checkpoint()
        self.train_model()
        self.push_data_to_container()
        print("Done!")
        return self.container
