import numpy as np

acc = [0.88810194, 0.915864,   0.9223796,  0.9223796 ]
std = [0.00780967, 0.00624516, 0.00590159, 0.00590159]
acc = np.round(np.array(acc), decimals=3)
std = np.round(np.array(std), decimals=3)
print(",".join([f"{acc[i]} +- {std[i]}" for i in range(len(acc))]))

acc = [0.82124996, 0.87299997, 0.8865,     0.88625   ]
std = [0.01739251, 0.0026926,  0.00310242, 0.00418329]
acc = np.round(np.array(acc), decimals=3)
std = np.round(np.array(std), decimals=3)
print(",".join([f"{acc[i]} +- {std[i]}" for i in range(len(acc))]))


acc = [0.7875,     0.822,      0.84250003, 0.842     ]
std = [0.00737395, 0.00407738, 0.01063603, 0.01044629]
acc = np.round(np.array(acc), decimals=3)
std = np.round(np.array(std), decimals=3)
print(",".join([f"{acc[i]} +- {std[i]}" for i in range(len(acc))]))


acc = [0.77849996, 0.8172501,  0.84075004, 0.842     ]
std = [0.00663327, 0.00398435, 0.01056527, 0.01044629]
acc = np.round(np.array(acc), decimals=3)
std = np.round(np.array(std), decimals=3)
print(",".join([f"{acc[i]} +- {std[i]}" for i in range(len(acc))]))






print("\n\n\n\n")
print("-"*50)
print("\n\n\n")



suff = [0.94623595, 0.984072,   0.98929197]
suff_std = [0.00419049, 0.0056346,  0.00237171]
suff = np.round(np.array(suff), decimals=3)
suff_std = np.round(np.array(suff_std), decimals=3)
print(",".join([f"{suff[i]} +- {suff_std[i]}" for i in range(len(suff_std))]))

nec = [0.01445,  0.011238, 0.008512, 0.007368]
nec_std = [0.0059129,  0.00568832, 0.00509005, 0.00403246]
nec = np.round(np.array(nec), decimals=3)
nec_std = np.round(np.array(nec_std), decimals=3)
print(",".join([f"{nec[i]} +- {nec_std[i]}" for i in range(len(nec))]))

necpp = [0.018628, 0.019878, 0.015842, 0.014326]
necpp_std = [0.00732172, 0.00882229, 0.00777751, 0.00670157]
necpp = np.round(np.array(necpp), decimals=3)
necpp_std = np.round(np.array(necpp_std), decimals=3)
print(",".join([f"{necpp[i]} +- {necpp_std[i]}" for i in range(len(necpp_std))]))

fidp = [0.029412, 0.025676, 0.020586, 0.01997 ]
fidp_std = [0.00988219, 0.00660429, 0.0034871,  0.00374218]
fidp = np.round(np.array(fidp), decimals=3)
fidp_std = np.round(np.array(fidp_std), decimals=3)
print(",".join([f"{fidp[i]} +- {fidp_std[i]}" for i in range(len(fidp_std))]))

fidm = [0.017366, 0.019704, 0.018926, 0.      ]
fidm_std =  [0.00564331, 0.00484141, 0.00339183, 0.        ]
fidm = np.round(np.array(fidm), decimals=3)
fidm_std = np.round(np.array(fidm_std), decimals=3)
print(",".join([f"{fidm[i]} +- {fidm_std[i]}" for i in range(len(fidm))]))

