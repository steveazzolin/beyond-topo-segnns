SEALION=azzolin@10.196.36.66
MARZOLA=steve.azzolin@marzola.disi.unitn.it:/home/steve.azzolin

SEED=$1
DATA=GOODCMNIST_color_covariate
# GOODMotif/basis GOODMotif2/basis GOODMotif/size GOODSST2/length GOODTwitter/length GOODHIV/scaffold GOODCMNIST/color LBAPcore/assay


mkdir -p ./storage/checkpoints/round${SEED}/${DATA}/

# scp -r \
#     ${MARZOLA}/sedignn/LECI_fork/storage/checkpoints/round${SEED}/${DATA}/repr_GIN_3l*avgedgeattndefault \
#     ./storage/checkpoints/round${SEED}/${DATA}/

scp -r \
    ${MARZOLA}/sedignn/LECI_fork/storage/checkpoints/round${SEED}/${DATA}/repr_LECIvGIN_5l_meanpool_0.5dp_mitig_backboneNone_mitig_samplingfeatmitig_explscorestopk0.8avgedgeattnmean \
    ./storage/checkpoints/round${SEED}/${DATA}/

# ${MARZOLA}/sedignn/LECI_fork/storage/checkpoints/round${SEED}/${DATA}/repr_LECIvGIN_3l_meanpool_0.5dp_mitig_backboneNone_mitig_samplingfeatmitig_readoutweightedmitig_virtualweightedmitig_explscoreshardavgedgeattnmean \
# ${MARZOLA}/sedignn/LECI_fork/storage/checkpoints/round${SEED}/${DATA}/repr_LECIvGIN_3l_meanpool_0.5dp_mitig_backboneNone_mitig_samplingfeatmitig_readoutweightedmitig_virtualweightedmitig_explscoresannealavgedgeattnmean \

# Itdgcspap?
# scp -r \
#     ${MARZOLA}/sedignn/LECI_fork/storage/checkpoints/round${SEED}/${DATA}/repr_LECIGIN_3l_meanpool_0.5dp_mitig_backboneNone_mitig_samplingfeatmitig_readoutweightedmitig_virtualweightedmitig_explscoreshardavgedgeattnmean \
#     ./storage/checkpoints/round${SEED}/${DATA}/
# scp -r \
#     ${MARZOLA}/sedignn/LECI_fork/storage/checkpoints/round${SEED}/${DATA}/repr_LECIGIN_3l_meanpool_0.5dp_mitig_backboneNone_mitig_samplingfeatmitig_readoutweightedmitig_virtualweightedmitig_explscoresannealavgedgeattnmean \
#     ./storage/checkpoints/round${SEED}/${DATA}/
    