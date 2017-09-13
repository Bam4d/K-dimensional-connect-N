# K-dimensional-connect-N
Basic 4-in-a-row game, expanded to K dimensions and N in a row. Then some reinforcement learning gets thrown in!

To read more about the *environment* read [this](https://bam4d.github.io/#/post/k-dimensional-connect-n--part-1-environment/2).

To read more about the *agent* read [this](https://bam4d.github.io/#/post/k-dimensional-connect-n--part-2-deep-learning/3).

## Requirements

python3 (have not tried this with 2.x), but it might work just as well.

you will need to have the following python libraries installed:
```
tensorflow
numpy
matplotlib
```

## To run

`python3 main_deep_q.py`

## To change hyperparameters / game configuration

in main_deep_q.py there are a set of commented out example game board and hyperparameter configurations that look like this:

```
game_board = [8, 7, 6, 5, 4, 3, 2]  # 7 dimensional
to_win = 4 # connect 4
episode_state_history_max = 100000
episode_state_history_min = 10000
update_period = 1000
```

you can choose one of them to un-comment and use, or you can try out your own combinations.

# Issues/comments
Please feel free to post on the blog posts linked above, or leave an issue on this github page.

