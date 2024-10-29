from training import train_model, evaluate_model
from plot import plot_results

if __name__ == "__main__":
    net, testloader = train_model()
    evaluate_model(net, testloader)
    plot_results()