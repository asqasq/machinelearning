#!/bin/bash
set -x


timeout 15m $SPARK_HOME/bin/spark-submit -v --num-executors 0 --executor-cores 1 --executor-memory 512M --driver-memory 2G \
			     --deploy-mode client \
			     --master spark://localhost:7077 \
			     --conf spark.app.name="asq_neural_network_mnist" \
			     neuornalnetwork04.py -p / -t $1 &

wait
echo "Done."
exit 0;

