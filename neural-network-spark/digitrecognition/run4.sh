#!/bin/bash
set -x


timeout 5m $SPARK_HOME/bin/spark-submit -v --num-executors 0 --executor-cores 1 --executor-memory 512M --driver-memory 2G \
			     --deploy-mode client \
			     --master spark://localhost:7077 \
			     --conf spark.app.name="asq_neural_network_mnist" \
			     neuornalnetwork04.py &

wait
echo "Done."
exit 0;

