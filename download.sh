SEALION=azzolin@10.196.36.66
MARZOLA=steve.azzolin@marzola.disi.unitn.it:/home/steve.azzolin

SEED=$1
DATA=GOODTwitter_length_covariate

mkdir -p ./storage/checkpoints/round${SEED}/${DATA}/

/home/steve.azzolin/sedignn/LECI_fork/storage/checkpoints/round1/GOODTwitter_length_covariate/repr_LECIvGIN_3l_meanpool_0.5dp_mitig_backboneNone_mitig_samplingfeatavgedgeattnmean

scp -r \
    ${MARZOLA}/sedignn/LECI_fork/storage/checkpoints/round${SEED}/${DATA}/repr_LECIvGIN_3l*avgedgeattnmean \
    ./storage/checkpoints/round${SEED}/${DATA}/
