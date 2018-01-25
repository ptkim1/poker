import numpy as np
import tensorflow as tf

class Actor_Critic(object):
    """
    Contains both actor and critic networks
    """

    def __init__(self, PARAMS):
        ## Do shit

    def actor_inference(self, state):
        """
        Builds the graph nodes for the policy
        and returns the tensor representing the 
        decision. 
        """
        with tf.name_scope('policy'):
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
        # Get actor variables using collection
        # approxJ = tf.log(logits) * advantages
        # tf.gradient(approxJ)

    def critic_loss(estimate, actual):
        critic_loss = #1/2 * sum_squared_error(estimate - actual)
        return critic_loss

    def actor_training(actor_grad, ):
        train_op = self.optimizer.apply_gradients(actor_grad)
        return train_op

    def critic_training(critic_loss, learning_rate=self.learning_rate):
        train_op = self.optimizer.minimize(critic_loss)
        return train_op

    def advantages(reward, discount, decision, critic):
        advantage = reward + (discount * critic.eval())
    
    