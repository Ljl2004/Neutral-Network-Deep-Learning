# plot the score and loss
import matplotlib.pyplot as plt

colors_set = {'Kraftime' : ('#E3E37D', '#968A62')}

def plot(runner, axes, set=colors_set['Kraftime']):
    train_color = set[0]
    dev_color = set[1]

    train_x = [i for i in range(len(runner.train_loss))]
    dev_x = [i * (len(runner.train_loss) / max(len(runner.dev_loss), 1)) for i in range(len(runner.dev_loss))]

    axes[0].plot(train_x, runner.train_loss, color=train_color, label="Train loss")
    axes[0].plot(dev_x, runner.dev_loss, color=dev_color, linestyle="--", label="Dev loss")
    axes[0].set_ylabel("loss")
    axes[0].set_xlabel("iteration")
    axes[0].set_title("")
    axes[0].legend(loc='upper right')

    train_x_scores = [i for i in range(len(runner.train_scores))]
    dev_x_scores = [i * (len(runner.train_scores) / max(len(runner.dev_scores), 1)) for i in range(len(runner.dev_scores))]

    axes[1].plot(train_x_scores, runner.train_scores, color=train_color, label="Train accuracy")
    axes[1].plot(dev_x_scores, runner.dev_scores, color=dev_color, linestyle="--", label="Dev accuracy")
    axes[1].set_ylabel("score")
    axes[1].set_xlabel("iteration")
    axes[1].legend(loc='lower right')