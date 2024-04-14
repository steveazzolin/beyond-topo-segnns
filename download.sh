SEALION=azzolin@10.196.36.66
MARZOLA=steve.azzolin@marzola.disi.unitn.it:/home/steve.azzolin

SEED=$1
DATA=GOODMotif2_basis_covariate

mkdir -p ./storage/checkpoints/round${SEED}/${DATA}/

# scp -r \
#     ${MARZOLA}/sedignn/LECI_fork/storage/checkpoints/round${SEED}/${DATA}/repr_GIN_3l*avgedgeattndefault \
#     ./storage/checkpoints/round${SEED}/${DATA}/

# scp -r \
#     ${MARZOLA}/sedignn/LECI_fork/storage/checkpoints/round${SEED}/${DATA}/repr_LECIvGIN_5l_meanpool_0.5dp_mitig_backboneNone_mitig_samplingfeatmitig_readoutweightedmitig_virtualweightedavgedgeattnmean \
#     ./storage/checkpoints/round${SEED}/${DATA}/


# scp -r \
#     ${MARZOLA}/sedignn/LECI_fork/storage/checkpoints/round${SEED}/${DATA}/repr_GSATGIN_3l*avgedgeattndefault \
#     ./storage/checkpoints/round${SEED}/${DATA}/

scp -r \
    "${MARZOLA}/sedignn/LECI_fork/storage/checkpoints/round${SEED}/${DATA}/repr_LECIGIN_3l_meanpool_0.5dp_mitig_backboneNone_mitig_samplingfeatmitig_readoutweightedmitig_virtualweightedmitig_explscorestopK0.8avgedgeattnmean" \
    ./storage/checkpoints/round${SEED}/${DATA}/

# scp -r \
#     ${MARZOLA}/sedignn/LECI_fork/storage/checkpoints/round${SEED}/${DATA}/repr_LECIGIN_3l_meanpool_0.5dp_mitig_backboneNone_mitig_samplingrawmitig_explscoresannealavgedgeattnmean \
#     ./storage/checkpoints/round${SEED}/${DATA}/
    