"""""
 *  \brief     generate_plots.py
 *  \author    Jonathan Reymond
 *  \version   1.0
 *  \date      2023-02-14
 *  \pre       None
 *  \copyright (c) 2022 CSEM
 *
 *   CSEM S.A.
 *   Jaquet-Droz 1
 *   CH-2000 Neuchâtel
 *   http://www.csem.ch
 *
 *
 *   THIS PROGRAM IS CONFIDENTIAL AND CANNOT BE DISTRIBUTED
 *   WITHOUT THE CSEM PRIOR WRITTEN AGREEMENT.
 *
 *   CSEM is the owner of this source code and is authorised to use, to modify
 *   and to keep confidential all new modifications of this code.
 *
 """

import matplotlib.pyplot as plt
import pickle
import seaborn as sns


choices = ['ventilator', '8K', 'wind', 'wind_minute']
choice_idx = 1
test = False


network_names = ['M3', 'M5', 'M18', 'M34', 'ACDNet']
suffixes = [ '_0.0001']
# network_names = ['M3']



def plot_train_test(train, test, xlabel, ylabel, network_name, store_path, test_str, suff):
    fig, ax = plt.subplots(1, 1, figsize=(12,4))
    x = range(1, len(train) + 1)
    plt.title(network_name)
    ax.plot(x, train, label = "train " + ylabel)
    ax.plot(x, test, label = "test " + ylabel)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.legend()
    plt.savefig(store_path + "figures/" + network_name + "_" + ylabel+ test_str + suff +".png", dpi=200)



if __name__ == '__main__':
    store_path = 'results/' +choices[choice_idx] + '/'

    test_str = ""
    if test :
        test_str = "_test"
    for suff in suffixes:
        for network_name in network_names:
            with open(store_path+ network_name + '_results' + test_str + suff + '.pkl', 'rb') as handle:
                results = pickle.load(handle)

            if choices[choice_idx] == 'wind':
                plot_train_test(results[0], results[1], 'epoch', 'loss', network_name, store_path, test_str, suff)

            else : 
                loss_train, loss_test, acc_train, acc_test = results
                plot_train_test(loss_train, loss_test, 'epoch', 'loss', network_name, store_path, test_str, suff)
                plot_train_test(acc_train, acc_test, 'epoch', 'accuracy', network_name, store_path, test_str, suff)


