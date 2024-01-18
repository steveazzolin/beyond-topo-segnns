SEALION=azzolin@10.196.36.66
MARZOLA=steve.azzolin@marzola.disi.unitn.it:/home/steve.azzolin

SEED=$1
DATA=GOODTwitter_length_covariate

mkdir -p ./storage/checkpoints/round${SEED}/${DATA}/



scp -r \
    ${MARZOLA}/sedignn/LECI_fork/storage/checkpoints/round${SEED}/${DATA}/repr_GSATvGIN_3l*avgedgeattnmean \
    ./storage/checkpoints/round${SEED}/${DATA}/
