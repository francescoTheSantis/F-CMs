#!/bin/bash

# Change directory to the script's directory
cd ../../../../../src

# Read the number of clients from the configuration file
n_clients=$(sed -n 's/^n_clients: //p' ../conf/learning/commons.yaml)

# Initiate server
python server.py &
sleep 3  # Sleep for 3s to give the server enough time to start

for i in $(seq 1 $n_clients); do
    echo "Starting client ID $i"
    # python client.py client_id="$i" &
    python client.py --client_id "$i" &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait

# Clean up
trap - SIGTERM 

# Kill all background processes
pkill -f client.py -9
pkill -f server.py -9
