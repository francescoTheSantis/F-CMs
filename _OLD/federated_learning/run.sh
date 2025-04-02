#!/bin/bash

k_folds=$(python -c "from public.config import k_folds; print(k_folds)")
model_name=$(python -c "from public.config import model_name; print(model_name)")
dataset_name=$(python -c "from public.config import dataset_name; print(dataset_name)")
strategy=$(python -c "from public.config import strategy; print(strategy)")
n_clients=$(python -c "from public.config import n_clients; print(n_clients)")
n_clusters=$(python -c "from public.config import n_clusters; print(n_clusters)")

echo -e "\n\033[1;36mExperiment settings:\033[0m\n\033[1;36m \
    MODEL: $model_name\033[0m\n\033[1;36m \
    Dataset: $dataset_name\033[0m\n\033[1;36m \
    Strategy: $strategy\033[0m\n\033[1;36m \
    Number of clients: $n_clients\033[0m\n\033[1;36m \
    Number of clusters: $n_clusters\033[0m\n\033[1;36m \
    \033[1;36mK-Folds: $k_folds\033[0m\n"

# K-Fold evaluation, if k_folds > 1
for fold in $(seq 0 $(($k_folds - 1))); do        
    echo -e "\n\033[1;36mStarting fold $((fold + 1))\033[0m\n"

    # Clean and create datasets
    rm -rf data/cur_datasets/* 
    cd ..
    python federated_learning/public/generate_datasets.py --fold "$fold" 
    cd federated_learning

    # Start the server and clients
    cd "$strategy"
    python server.py --fold "$fold" &
    sleep 3  

    for i in $(seq 0 $(($n_clients - 1))); do
        echo "Starting client ID $i"
        python client.py --id "$i" --fold "$fold" &
    done

    # This will allow you to use CTRL+C to stop all background processes
    trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
    # Wait for all background processes to complete
    wait

    # Clean up
    echo "Fold completed correctly"
    trap - SIGTERM 

    pkill -f client.py -9
    pkill -f server.py -9

    # Change back to the root directory
    cd ..
    sleep 3

done

# K-Fold evaluation, if k_folds > 1
if [ "$k_folds" -gt 1 ]; then

    echo -e "\n\033[1;36mAveraging the results of all folds\033[0m\n"
    # Averaging the results of all folds
    python public/average_results.py
    # Plot confidence interval plots
    python public/plots_across_folds.py --dataset "$dataset_name"
fi


echo -e "\n\033[1;36mExperiment completed successfully\033[0m\n"
# kill
trap - SIGTERM && kill -- -$$
