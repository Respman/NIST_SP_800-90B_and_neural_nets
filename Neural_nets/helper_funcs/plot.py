import matplotlib.pylab as plt


def Show_loss_plot(training_results):
    plt.figure(figsize=(8,4))
    plt.plot(training_results['training_loss'], label='neuro')
    plt.ylabel('loss')
    plt.xlabel('Samples of Training')
    plt.title('training loss iterations')
    plt.legend()
    plt.show()


def Show_validation_accuracy_plot(training_results):
    plt.figure(figsize=(8,4))
    plt.grid()
    plt.plot(training_results['validation_accuracy'], label = 'neuro')
    plt.ylabel('validation accuracy')
    plt.title('Accuracy for training set')
    plt.xlabel('Iteration') 
    plt.legend()
    plt.show()