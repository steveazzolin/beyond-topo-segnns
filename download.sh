SEALION=azzolin@10.196.36.66
MARZOLA=steve.azzolin@marzola.disi.unitn.it:/home/steve.azzolin

SEED=$1
DATA=GOODCMNIST_color_covariate

mkdir -p ./storage/checkpoints/round${SEED}/${DATA}/

# scp -r \
#     ${MARZOLA}/sedignn/LECI_fork/storage/checkpoints/round${SEED}/${DATA}/repr_GIN_3l*avgedgeattndefault \
#     ./storage/checkpoints/round${SEED}/${DATA}/

scp -r \
    ${MARZOLA}/sedignn/LECI_fork/storage/checkpoints/round${SEED}/${DATA}/repr_LECIvGIN_5l_meanpool_0.5dp_mitig_backboneNone_mitig_samplingfeatmitig_readoutweightedmitig_virtualweightedavgedgeattnmean \
    ./storage/checkpoints/round${SEED}/${DATA}/


# scp -r \
#     ${MARZOLA}/sedignn/LECI_fork/storage/checkpoints/round${SEED}/${DATA}/repr_GSATGIN_3l*avgedgeattndefault \
#     ./storage/checkpoints/round${SEED}/${DATA}/

# scp -r \
#     ${MARZOLA}/sedignn/LECI_fork/storage/checkpoints/round${SEED}/${DATA}/repr_CIGAGIN_3l*avgedgeattndefault \
#     ./storage/checkpoints/round${SEED}/${DATA}/
