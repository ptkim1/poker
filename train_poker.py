import tensorflow as tf
import Actor_Critic from actor_critic

OPTIMIZER = # Some optimizer
STATE_DIM = # Some state dim
SUMMARY_WRITER = # Figure this shit out later

# Build Graph
model = Actor_Critic(OPTIMIZER, STATE_DIM, SUMMARY_WRITER)

decision = model.actor_inference(PLACEHOLDER_FOR_STATE)
critic = model.critic_inference(PLACEHOLDER_FOR_STATE)
advantage = 
