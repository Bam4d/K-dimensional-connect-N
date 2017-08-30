import numpy as np

class Environment:

    # Default is a 7 by 6 grid and 4 in a row to win
    # Second dimension is always the dimension in which the tokens "fall"
    def __init__(self, dimension_configuration=(7,6), N=4):

        # Reshape this so the first dimension is the gravity dimension

        self.K = len(dimension_configuration)

        reorder = np.array([1, 0])
        if self.K > 2:
            reorder = np.append(np.array([1, 0]), np.arange(2, self.K))

        self.dimension_configuration = dimension_configuration
        self.N = N

        self.state = np.zeros(dimension_configuration, dtype=np.uint8)

    def step(self, action, player):
        done = False
        reward = 0.0
        # Update the state here
        assert np.abs(player) == 1, "player can only have value 1 or -1"

        entry_vector_shape = self.K - 1
        assert len(action) == entry_vector_shape, "entry vector must have length %d" % entry_vector_shape

        token_location = np.sum(np.abs(self.state[:, action]))
        self.state[token_location, action] = player

        if self.check_for_win(player, token_location):
            done = True
            reward = 200

        return self.state, reward, done, None


    def sample_action(self):
        entry_dims = self.action_shape()
        return np.array([np.random.choice(dim, 1)[:] for dim in entry_dims])[:,-1]

    def action_shape(self):
        return self.dimension_configuration[:1]

    def state_shape(self):
        return self.dimension_configuration

    """
    This algorithm will look around the last place where a token was placed looking for a set of N of the same token
    """
    def check_for_win(self, player, last_token_location):

        """
        Check calculate the sum of a line of tokens

        O(N^2)

        :param player:
        :param last_token_location:
        :return:
        """
        def check_line(last_token_location, increment_vector, required_sum):

            start_location = last_token_location
            for i in range(0, self.N):
                token = start_location
                sum = 0
                for i in range(0, self.N):
                    sum += self.state[np.expand_dims(token, axis=1)]

                    if required_sum == sum:
                        return True
                    token += increment_vector

                start_location += increment_vector

        required_sum = player*self.N

        ## Check line

        pass