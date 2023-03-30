echo $$
echo $BASHPID
export TRAINTIME=$(date +"%Y%d%d_%H%M%S")
echo $TRAINTIME 
# python train.py |& tee -a ../data/log_$TRAINTIME.txt
fil-profile run train.py |& tee -a ../data/log_$TRAINTIME.txt