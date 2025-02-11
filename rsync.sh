CLUSTER=$1

rsync --info=progress2 -urltv --filter=':- .gitignore' -e ssh ./ "cc-$CLUSTER":~/pip/wandb-offline-sync-hook