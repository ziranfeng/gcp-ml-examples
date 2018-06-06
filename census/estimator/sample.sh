TRAIN_FILE=gs://cloud-samples-data/ml-engine/census/data/adult.data.csv
EVAL_FILE=gs://cloud-samples-data/ml-engine/census/data/adult.test.csv

DATE=`date '+%Y%m%d_%H%M%S'`
export JOB_NAME=census_$DATE
export GCS_JOB_DIR=gs://dlab_competation/path/to/myjobs/$JOB_NAME
echo $GCS_JOB_DIR
export TRAIN_STEPS=5000

gcloud ml-engine jobs submit training $JOB_NAME \
                                    --stream-logs \
                                    --runtime-version 1.4 \
                                    --job-dir $GCS_JOB_DIR \
                                    --module-name trainer.task \
                                    --package-path trainer/ \
                                    --region us-central1 \
                                    -- \
                                    --train-files $TRAIN_FILE \
                                    --eval-files $EVAL_FILE \
                                    --train-steps $TRAIN_STEPS \
                                    --eval-steps 100