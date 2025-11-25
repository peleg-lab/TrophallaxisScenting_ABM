"""# Global Variables"""

tro_counter = 0     # counts # of trophallaxis events
all_counter = 0
n1_counter = 0      # counts every neighboring event if the neighbors have different food values
n2_counter = 0      # counts every neighboring event if the neighbors are not occupied
n3_counter = 0      # counts all the neighboring event
u_counter = 0       # counts all unique encounters
target = None          # the id of the bee who is doing the food exchange
delta_food = 0      # the difference between the amount of food in the full bee's stomach and her hungry neighbor
epsilon = 0         # decides when the model stops
hungry_counter = 0  # counts the number of hungry bees
var = None             # the previous value of the variance of the food of all bees
newvar = None          # the new value of the variance of the food of all bees
deltavar = 0        # the difference between the variance of food (used to set our stopping condition)
hungry_left = 0     # number of agents who still have zero food
food_transfer_rate = 0      # the constant rate of food exchange
max_transfer_t = 0          # the constant to show the maximum time (in ticks) that two bees need to stop for exchanging food
rand_delta_food = 0     # set if you want to implement random values of food exchange/durations

rad = None
clusters = None
run_seed = None
donor_list = []
target_list = []
foods_list = []
xcor_list = []
ycor_list = []
# donor = None
attr = None
attr_rad = 2.5
step_size = 1.0           # step size
neighbor_count = None
block_count = None
L = None

run_counter = 0

## Scenting Values:
# queens:
queen_emission_frequency = 80
queen_initial_concentration = 0.0575
# workers:
worker_initial_concentration = [0.0575]
worker_wait_period = 80

# # Clustering:
# cluster_labels = None
#
# # range 0 - 2000
# # number_of_bees = 110
#
# # range 0 - 100
# # fraction_of_fed_bees = 10
#
# # Theta range 0 - 180
# theta = 180
#
# # range 0 - 10
# # attraction_radius = 2.5
