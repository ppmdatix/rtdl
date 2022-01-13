from load_data import load_data
from create_model import create_model
from learn_that import learn_that, evaluate, device
from plot_losses import plot_losses, create_path
import zero
import sys
import pandas as pd

dataset    = sys.argv[1]
task_type  = sys.argv[2]
epochs     = int(sys.argv[3])
batch_size = int(sys.argv[4])


X, y, old_x, X_all, y_std = load_data("data/", dataset  +".csv", task_type=task_type)
model, optimizer, loss_fn = create_model(X_all, task_type=task_type)

for relational_batch in [True, False]:
    save_path = create_path("results/" + dataset, epochs, batch_size, relational_batch)
    losses = learn_that(
                model, 
                optimizer, 
                loss_fn, 
                X, 
                y, 
                y_std,
                epochs, 
                batch_size, 
                relational_batch, 
                old_x, 
                print_mode=False, 
                _task_type=task_type)
    plot_losses(losses, path=save_path)
    
    df = pd.DataFrame(losses)
    
    # saving the dataframe 
    df.to_csv(save_path+'.csv', index=False) 