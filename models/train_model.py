import torch
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.pyplot as plt

from sd_PNHP import *
from hawkes_loss import HawkesLoss



window_len = 50

def main():
    df_message = pd.read_csv('data/clean/AMZN_message_data.csv',index_col=0)
    df_orders = pd.read_csv('data/clean/AMZN_orderbook_data.csv')

    time_raw = df_message['time'].to_numpy()
    events_raw = df_message['event type thesis'].to_numpy()
    states_raw = df_orders['state indicator'].to_numpy()

    times = split_rolling_window(time_raw, window_len)
    events = split_rolling_window(events_raw, window_len)
    states = split_rolling_window(states_raw, window_len)
    trans_mtrx = np.load('data/transition_matrices/AMZN_matrix.npy')

    test_input = torch.randn(len(times), window_len, 2)

    model = LSTMLayerStacked(LSTMCell, 2, 16, times[0])
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.002)
    loss_fn = HawkesLoss(trans_mtrx=trans_mtrx)

    training_loop(n_epochs=10, model=model, optimiser=optimizer, loss_fn=loss_fn,
                  train_inputs= test_input, time=times, event_seq=events, state_seq=states) #error happened after moving time from initialization to forward.   



def training_loop(n_epochs, model, optimiser, loss_fn, train_inputs, time, event_seq, state_seq):
    torch.autograd.set_detect_anomaly(True)
    for i in range(n_epochs):
        for j, (train_input, time, event_seq, state_seq) in enumerate(zip(train_inputs, time, event_seq, state_seq)):
            def closure():
                optimiser.zero_grad()
                intensity_funcs, t_seq = model(train_input, time)
                loss = loss_fn(intensity_funcs, t_seq, event_seq, state_seq)
                loss.backward()
                return loss
            optimiser.step(closure)
        # print the loss
        out = model(train_input)
        loss_print = loss_fn(out)
        print("Step: {}, Loss: {}".format(i, loss_print))

def split_rolling_window(arr, window_len):
    return sliding_window_view(arr, window_shape = window_len)

if __name__ == "__main__":
    main()