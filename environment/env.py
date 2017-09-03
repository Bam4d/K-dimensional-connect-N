import numpy as np

class KDimConnectN:

    # Default is a 7 by 6 grid and 4 in a row to win
    # Second dimension is always the dimension in which the tokens "fall"
    def __init__(self, dimension_configuration=(7,6), N=4):

        # Reshape this so the first dimension is the gravity dimension

        self.K = len(dimension_configuration)

        self.dimension_configuration = np.array(dimension_configuration, dtype=np.uint32)
        self.N = N

        self.total_steps = 0
        self.max_steps = np.prod(dimension_configuration)

        self.state = np.zeros(dimension_configuration, dtype=np.int8)

        self.check_vectors = self.pre_calculate_check_line_vectors()

    def step(self, action, player):
        done = False
        reward = -1
        # Update the state here
        assert np.abs(player) == 1, "player can only have value 1 or -1"

        entry_vector_shape = self.K - 1
        assert len(action) == entry_vector_shape, "entry vector must have length %d" % entry_vector_shape

        # Had to find out how to do this on stack overflow
        # https://stackoverflow.com/questions/46022514/dynamically-obtain-a-vector-along-1-dimension-in-n-dimensional-numpy-array
        token_fall_coordinate = np.sum(np.abs(self.state[[slice(None)] +  action.tolist()]))

        # If the given action is not possible (the location is full) we set completed to False to let the agent
        # know the move is not possible ... perhaps a negative reward here?
        valid_move = True
        if token_fall_coordinate >= self.dimension_configuration[0]:
            valid_move = False
        else:

            # Reward for making a valid move ?
            # reward = 1

            token_final_location = np.concatenate((np.array([token_fall_coordinate], dtype=np.int64), action))

            self.state[np.expand_dims(token_final_location,1).tolist()] = player

            if self.check_for_win(player, token_final_location):
                done = True

                # Reward for winning
                reward = 20

            self.total_steps += 1
            if self.total_steps >= self.max_steps:
                done = True


        return self.state, reward, done, valid_move


    def sample_action(self):
        entry_dims = self.action_shape()
        return np.array([np.random.choice(dim, 1)[:] for dim in entry_dims])[:,-1]

    def action_shape(self):
        return self.dimension_configuration[1:]

    def state_shape(self):
        return self.dimension_configuration

    def reset(self):
        self.state = np.zeros(self.dimension_configuration, dtype=np.int8)
        self.total_steps = 0
        return self.state


    def pre_calculate_check_line_vectors(self):
        """
        This algorithm pre-calculates vectors to check for contiguous lines of a players tokens
        We want to calculate this up-front (its very complex, at least O(n * 3^n) and re-use the generated list of vectors each time
        """

        def convert_to_vector(value, K):
            """
            Converts the value to base 3 and then swap "2" with "-1"
            :param value:
            :param K:
            :return:
            """
            vector = np.zeros(K)
            for k in range(0, K):
                base3_power = 3 ** ((K - 1) - k)

                a = int(value / base3_power)
                value = value % base3_power

                vector[k] = a if a < 2 else -1

            return np.array(vector, dtype=np.int64)

        vectors = []

        for i in range(0, self.K):

            take = 3 ** i
            for j in range(0, take):
                converted_vector = convert_to_vector(take + j, self.K)
                vectors.append(converted_vector)

        return vectors


    def check_for_win(self, player, last_token_location):
        """
        This algorithm will look around the last place where a token was placed looking for a set of N of the same token
        """

        def is_valid_position_state(position):
            return np.alltrue(position >= 0) \
            and np.alltrue(self.dimension_configuration - (position) > 0)


        def check_line(last_token_location, increment_vector, player):
            """
            Check calculate the sum of a line of tokens
            """

            done = False
            iters = 0
            # Find the start point
            start_point = np.copy(last_token_location)
            while not done and iters != self.N-1:
                next_position = start_point - increment_vector
                if not is_valid_position_state(next_position):
                    done = True
                else:
                    start_point = next_position
                    iters += 1

            iters = 0
            done = False
            # Find the end point
            end_point = np.copy(last_token_location)
            while not done and iters != self.N-1:
                next_position = end_point + increment_vector
                if not is_valid_position_state(next_position):
                    done = True
                else:
                    end_point = next_position
                    iters += 1

            done = False
            # are there N in a row?
            player_token_count = 0
            check_point = np.copy(start_point)
            while not done:

                token = self.state[np.expand_dims(check_point, axis=1).tolist()]
                if token == player:
                    player_token_count += 1
                else:
                    player_token_count = 0

                if player_token_count == self.N:
                    return True

                if np.alltrue(check_point==end_point):
                    return False

                check_point += increment_vector

        ## The vectors to check for lines of N tokens
        for v in self.check_vectors:
            if check_line(last_token_location, v, player):
                return True

        return False



