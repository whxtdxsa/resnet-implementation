import numpy as np
import matplotlib.pyplot as plt
def plotting(train_loss_list, train_acc_list, test_acc_list, file_name="result", path="."):
    print("Saving result ...")
    x1 = np.arange(len(train_loss_list))
    x2 = np.arange(len(train_acc_list))

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)  
    plt.plot(x1, train_loss_list, label='train loss')
    plt.xlabel("iters")
    plt.ylabel("loss")
    plt.legend()
    plt.title("Loss Curve")

    plt.subplot(1, 2, 2) 
    plt.plot(x2, train_acc_list, label='train acc')
    plt.plot(x2, test_acc_list, label='test acc')
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.title("Accuracy Curve")

    plt.tight_layout()  
    plt.savefig(path + f'/{file_name}.png')
    plt.clf()

    print("Complete!")
