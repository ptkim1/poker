import numpy as np
import tensorflow as tf

class Actor_Critic(object):
    """
    Contains both actor and critic networks
    """

    def __init__(self,
                optimizer,
                state_dim,
                summary_writer,
                summary_every=100):

        self.optimizer = optimizer
        self.state_dim = state_dim
        self.summary_writer = summary_writer
        self.summary_every = summary_every

    def actor_inference(self, state):
        """
        Builds the graph nodes for the policy
        and returns the tensor representing the 
        decision. 
        """
        with tf.name_scope('actor'):
            # Starting with a two layer network
            hidden1 = tf.layers.dense(
                inputs=state,
                units=HIDDEN_1_POLICY_UNITS,
                activation = tf.tanh,
                trainable=True,
                name='hidden1'
            )

            hidden2 = tf.layers.dense(
                inputs=hidden1,
                units=HIDDEN_2_POLICY_UNITS,
                activation=tf.tanh,
                trainable=True,
                name='hidden2'
            )

            return hidden2

    def critic_inference(self, state):
        """
        Builds the graph nodes for the critic
        and returns the tensor representing the
        critic's estimate of future reward 
        from the state
        """
        with tf.name_scope('critic'):
            # Starting with a two layer network
            hidden1 = tf.layers.dense(
                inputs=state,
                units=HIDDEN_1_POLICY_UNITS,
                activation = tf.tanh,
                trainable=True,
                name='hidden1'
            )

            hidden2 = tf.layers.dense(
                inputs=hidden1,
                units=HIDDEN_2_POLICY_UNITS,
                activation=tf.tanh,
                trainable=True,
                name='hidden2'
            )

            return hidden2

    def actor_gradient(logits, advantages):
        # Get variables for actor, for gradient calculation 
        actor_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')

        # Compute gradients wrt log(logits)
        actor_gradients = self.optimizer.compute_gradients(tf.log(logits), actor_variables)

        # Apply advantages
        for i, (grad, var) in enumerate(actor_gradients):
            if grad is not None:
                actor_gradients[i] = (grad * advantages, var)

        return actor_gradients

    def critic_loss(estimate, actual):
        critic_loss = tf.reduce_mean(tf.square(estimate - actual))
        return critic_loss

    def actor_training(actor_grad):
        train_op = self.optimizer.apply_gradients(actor_grad)
        return train_op

    def critic_training(critic_loss, learning_rate=self.learning_rate):
        train_op = self.optimizer.minimize(critic_loss)
        return train_op

    def advantages(reward, discount, v_next, v_curr, critic):
        # Need to open up a session here
        advantage = reward + (discount * v_next) - (v_curr)
    
        return advantage