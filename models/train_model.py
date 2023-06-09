import torch
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.pyplot as plt
import time

from sd_PNHP import *
from hawkes_loss import *



window_len = 50

def main():
    df_message = pd.read_csv('data/clean/AMZN_message_data.csv',index_col=0)
    df_orders = pd.read_csv('data/clean/AMZN_orderbook_data.csv')

    time_raw = df_message['time'].to_numpy()
    events_raw = df_message['event type thesis'].to_numpy().reshape(-1,1)
    states_raw = df_orders['state indicator'].to_numpy().reshape(-1,1)
    event_state_concated = np.hstack([events_raw, states_raw])

    event_state_windows = torch.from_numpy(sliding_window_view(event_state_concated, window_shape = (window_len, 2))).squeeze().float()
    times = torch.from_numpy(sliding_window_view(time_raw, window_shape = window_len)).squeeze()
    trans_mtrx = np.load('data/transition_matrices/AMZN_matrix.npy')

    model = LSTMLayerStacked(LSTMCell, 2, 16)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.002)
    loss_fn = HawkesLoss(trans_mtrx=trans_mtrx)
    # loss_fn = Test_loss()

    training_loop(n_epochs=200, model=model, optimiser=optimizer, loss_fn=loss_fn,
                  all_times=times, all_inputs=event_state_windows) #error happened after moving time from initialization to forward.   

def training_loop(n_epochs, model, optimiser, loss_fn, all_times, all_inputs):
    
    for i in range(n_epochs):
        begin_time = time.time()

        for j, (time_window, input_window) in enumerate(zip(all_times, all_inputs)):
            if j % 100 == 0:
                print(f'TRAINING RUN {j}')
            def closure():
                optimiser.zero_grad()
                intensity_funcs, t_seq = model(input_window, time_window)
                loss = loss_fn(intensity_funcs, t_seq, input_window)
                loss.backward()
                return loss
            optimiser.step(closure)
        end_time = time.time()

        torch.save(model.state_dict(), f'trained_models/test_model_epoch{i}.pth')

        # print the loss
        intensity_funcs, t_seq = model(input_window, time_window)
        loss_print = loss_fn(intensity_funcs, t_seq, input_window)
        print(f'step: {i}, loss{loss_print}, epoch time: {end_time - begin_time}')

def split_rolling_window(arr, window_len):
    return sliding_window_view(arr, window_shape = (window_len, 2))

if __name__ == "__main__":
    main()