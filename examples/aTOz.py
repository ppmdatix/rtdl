from load_data import load_data
from create_model import create_model
from learn_that import learn_that
from plot_losses import plot_losses, create_path
import sys
import pandas as pd
from box_plot import box_plot

dataset = sys.argv[1]


if dataset.lower() == "kdd":
    dataDir = "examples/data/KDD99/"
    path = dataDir + "training_processed.csv"# "fetch_kddcup99.csv"
    resDir = "examples/results/KDD99/"
    target = "labels"
elif dataset.lower() == "forest_cover":
    dataDir = "examples/data/Forest_Cover/"
    path = dataDir + "training_processed.csv"# "forest_cover.csv"
    resDir = "examples/results/Forest_Cover/"
    target = "Cover_Type"

else:
    raise Exception('no such dataset')

task_type  = sys.argv[2]
epochs     = int(sys.argv[3])
batch_size = int(sys.argv[4])
k          = int(sys.argv[5])

target_name = "target"  
if len(sys.argv) > 6:
    target_name = sys.argv[6]

X, y, old_x, X_all, y_std, target_values = \
    load_data(path, task_type=task_type, target_name=target_name)

if task_type == "multiclass":
    n_classes = len(target_values)
else:
    n_classes = None

results = {"rb": [], "norb": []}

for relational_batch in [True, False]:
    for _k in range(k):
        model, optimizer, loss_fn = create_model(X_all, n_classes=n_classes, task_type=task_type)
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
        if relational_batch:
            results["rb"].append(losses["test"][-1])
        else:
            results["norb"].append(losses["test"][-1])
        title = dataset + "-relationalBatch:" + str(relational_batch)
        if _k == 1:
            plot_path = create_path(resDir, epochs, batch_size, relational_batch)
            plot_losses(losses, title=title, path=plot_path)

            df = pd.DataFrame(losses)

            df.to_csv(plot_path + '.csv', index=False)
if k > 1:
    save_path = create_path(resDir.replace("data", "results"), epochs, batch_size, k)
    box_plot(results, path=save_path)
