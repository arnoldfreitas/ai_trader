echo $$
echo $BASHPID
export TRAINTIME=$(date +"%Y%d%d_%H%M%S")
echo $TRAINTIME 
python -m trace --trace train.py |& tee -a ../data/log_$TRAINTIME.txt
# fil-profile train.py |& tee -a ../data/log_$TRAINTIME.txt