SEALION=azzolin@10.196.36.66
MARZOLA=steve.azzolin@marzola.disi.unitn.it:/home/steve.azzolin

SEED=$1

mkdir -p ./storage/checkpoints/round${SEED}/GOODMotif_basis_covariate/

scp -r \
    ${MARZOLA}/sedignn/LECI_fork/storage/checkpoints/round${SEED}/GOODMotif_basis_covariate/repr_LECIGIN_3l*avgedgeattnmean \
    ./storage/checkpoints/round${SEED}/GOODMotif_basis_covariate/
