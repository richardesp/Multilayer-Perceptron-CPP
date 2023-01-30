import os

"""
### FIRST EXPERIMENT USING OFFLINE MODE WITH ONE TYPE ERROR AND OUTPUT FUNCTION ###

#################################################################################################### XOR PROBLEM

error_function = 1
# Softmax as output function
# Online mode
iterations = 1000 # For XOR and ILDPB problem, 500 for nominist problem

os.system(
    f"./la2 -t ../../datasetsLA2IMC/dat/train_xor.dat -T ../../datasetsLA2IMC/dat/test_xor.dat -f {error_function} -s -l 2 -h 100 -e 0.7 -m 1 -i {iterations} -V 0.2 -v xor_problem &")

#################################################################################################### ILDPB PROBLEM

error_function = 1
# Softmax as output function
iterations = 1000 # For XOR and ILDPB problem, 500 for nominist problem

os.system(
    f"./la2 -t ../../datasetsLA2IMC/dat/train_ildp.dat -T ../../datasetsLA2IMC/dat/test_ildp.dat -f {error_function} -s -l 1 -h 4 -e 0.7 -m 1 -i {iterations} -V 0.2 -v ildpb_problem &")

os.system(
    f"./la2 -t ../../datasetsLA2IMC/dat/train_ildp.dat -T ../../datasetsLA2IMC/dat/test_ildp.dat -f {error_function} -s -l 1 -h 8 -e 0.7 -m 1 -i {iterations} -V 0.2 -v ildpb_problem &")

os.system(
    f"./la2 -t ../../datasetsLA2IMC/dat/train_ildp.dat -T ../../datasetsLA2IMC/dat/test_ildp.dat -f {error_function} -s -l 1 -h 16 -e 0.7 -m 1 -i {iterations} -V 0.2 -v ildpb_problem &")

os.system(
    f"./la2 -t ../../datasetsLA2IMC/dat/train_ildp.dat -T ../../datasetsLA2IMC/dat/test_ildp.dat -f {error_function} -s -l 1 -h 64 -e 0.7 -m 1 -i {iterations} -V 0.2 -v ildpb_problem &")

os.system(
    f"./la2 -t ../../datasetsLA2IMC/dat/train_ildp.dat -T ../../datasetsLA2IMC/dat/test_ildp.dat -f {error_function} -s -l 2 -h 4 -e 0.7 -m 1 -i {iterations} -V 0.2 -v ildpb_problem &")

os.system(
    f"./la2 -t ../../datasetsLA2IMC/dat/train_ildp.dat -T ../../datasetsLA2IMC/dat/test_ildp.dat -f {error_function} -s -l 2 -h 8 -e 0.7 -m 1 -i {iterations} -V 0.2 -v ildpb_problem &")

os.system(
    f"./la2 -t ../../datasetsLA2IMC/dat/train_ildp.dat -T ../../datasetsLA2IMC/dat/test_ildp.dat -f {error_function} -s -l 2 -h 16 -e 0.7 -m 1 -i {iterations} -V 0.2 -v ildpb_problem &")

os.system(
    f"./la2 -t ../../datasetsLA2IMC/dat/train_ildp.dat -T ../../datasetsLA2IMC/dat/test_ildp.dat -f {error_function} -s -l 2 -h 64 -e 0.7 -m 1 -i {iterations} -V 0.2 -v ildpb_problem &")

#################################################################################################### noMINIST PROBLEM

error_function = 1
# Softmax as output function
iterations = 500 # For XOR and ILDPB problem, 500 for nominist problem

os.system(
    f"./la2 -t ../../datasetsLA2IMC/dat/train_nomnist.dat -T ../../datasetsLA2IMC/dat/test_nomnist.dat -f {error_function} -s -l 1 -h 4 -e 0.7 -m 1 -i {iterations} -V 0.2 -v nominist_problem &")

os.system(
    f"./la2 -t ../../datasetsLA2IMC/dat/train_nomnist.dat -T ../../datasetsLA2IMC/dat/test_nomnist.dat -f {error_function} -s -l 1 -h 8 -e 0.7 -m 1 -i {iterations} -V 0.2 -v nominist_problem &")

os.system(
    f"./la2 -t ../../datasetsLA2IMC/dat/train_nomnist.dat -T ../../datasetsLA2IMC/dat/test_nomnist.dat -f {error_function} -s -l 1 -h 16 -e 0.7 -m 1 -i {iterations} -V 0.2 -v nominist_problem &")

os.system(
    f"./la2 -t ../../datasetsLA2IMC/dat/train_nomnist.dat -T ../../datasetsLA2IMC/dat/test_nomnist.dat -f {error_function} -s -l 1 -h 64 -e 0.7 -m 1 -i {iterations} -V 0.2 -v nominist_problem &")

os.system(
    f"./la2 -t ../../datasetsLA2IMC/dat/train_nomnist.dat -T ../../datasetsLA2IMC/dat/test_nomnist.dat -f {error_function} -s -l 2 -h 4 -e 0.7 -m 1 -i {iterations} -V 0.2 -v nominist_problem &")

os.system(
    f"./la2 -t ../../datasetsLA2IMC/dat/train_nomnist.dat -T ../../datasetsLA2IMC/dat/test_nomnist.dat -f {error_function} -s -l 2 -h 8 -e 0.7 -m 1 -i {iterations} -V 0.2 -v nominist_problem &")

os.system(
    f"./la2 -t ../../datasetsLA2IMC/dat/train_nomnist.dat -T ../../datasetsLA2IMC/dat/test_nomnist.dat -f {error_function} -s -l 2 -h 16 -e 0.7 -m 1 -i {iterations} -V 0.2 -v nominist_problem &")

os.system(
    f"./la2 -t ../../datasetsLA2IMC/dat/train_nomnist.dat -T ../../datasetsLA2IMC/dat/test_nomnist.dat -f {error_function} -s -l 2 -h 64 -e 0.7 -m 1 -i {iterations} -V 0.2 -v nominist_problem &")

### SECOND EXPERIMENT USING OFFLINE MODE ###

#################################################################################################### XOR PROBLEM

iterations = 1000 # For XOR and ILDPB problem, 500 for nominist problem
# Offline mode

os.system(
    f"./la2 -t ../../datasetsLA2IMC/dat/train_xor.dat -T ../../datasetsLA2IMC/dat/test_xor.dat -f 0 -l 2 -h 100 -e 0.7 -m 1 -i {iterations} -V 0.2 -v xor_problem &")

os.system(
    f"./la2 -t ../../datasetsLA2IMC/dat/train_xor.dat -T ../../datasetsLA2IMC/dat/test_xor.dat -f 0 -s -l 2 -h 100 -e 0.7 -m 1 -i {iterations} -V 0.2 -v xor_problem &")

os.system(
    f"./la2 -t ../../datasetsLA2IMC/dat/train_xor.dat -T ../../datasetsLA2IMC/dat/test_xor.dat -f 1 -s -l 2 -h 100 -e 0.7 -m 1 -i {iterations} -V 0.2 -v xor_problem &")

#################################################################################################### ILDPB PROBLEM

iterations = 1000 # For XOR and ILDPB problem, 500 for nominist problem
# Offline mode

os.system(
    f"./la2 -t ../../datasetsLA2IMC/dat/train_ildp.dat -T ../../datasetsLA2IMC/dat/test_ildp.dat -f 0 -l 2 -h 4 -e 0.7 -m 1 -i {iterations} -V 0.2 -v ildp_problem &")

os.system(
    f"./la2 -t ../../datasetsLA2IMC/dat/train_ildp.dat -T ../../datasetsLA2IMC/dat/test_ildp.dat -f 0 -s -l 2 -h 4 -e 0.7 -m 1 -i {iterations} -V 0.2 -v ildp_problem &")

os.system(
    f"./la2 -t ../../datasetsLA2IMC/dat/train_ildp.dat -T ../../datasetsLA2IMC/dat/test_ildp.dat -f 1 -s -l 2 -h 4 -e 0.7 -m 1 -i {iterations} -V 0.2 -v ildp_problem &")

#################################################################################################### NOMNIST PROBLEM

iterations = 500 # For XOR and ILDPB problem, 500 for nominist problem
# Offline mode

os.system(
    f"./la2 -t ../../datasetsLA2IMC/dat/train_nomnist.dat -T ../../datasetsLA2IMC/dat/test_nomnist.dat -f 0 -l 2 -h 64 -e 0.7 -m 1 -i {iterations} -V 0.2 -v nomnist_problem &")

os.system(
    f"./la2 -t ../../datasetsLA2IMC/dat/train_nomnist.dat -T ../../datasetsLA2IMC/dat/test_nomnist.dat -f 0 -s -l 2 -h 64 -e 0.7 -m 1 -i {iterations} -V 0.2 -v nomnist_problem &")

os.system(
    f"./la2 -t ../../datasetsLA2IMC/dat/train_nomnist.dat -T ../../datasetsLA2IMC/dat/test_nomnist.dat -f 1 -s -l 2 -h 64 -e 0.7 -m 1 -i {iterations} -V 0.2 -v nomnist_problem &")
"""


### THIRD EXPERIMENT USING ONLINE MODE ###

#################################################################################################### XOR PROBLEM

iterations = 1000 # For XOR and ILDPB problem, 500 for nominist problem
# Offline mode

os.system(
    f"./la2 -t ../../datasetsLA2IMC/dat/train_xor.dat -T ../../datasetsLA2IMC/dat/test_xor.dat -o -f 0 -l 2 -h 100 -e 0.7 -m 1 -i {iterations} -V 0.2 -v xor_problem &")

os.system(
    f"./la2 -t ../../datasetsLA2IMC/dat/train_xor.dat -T ../../datasetsLA2IMC/dat/test_xor.dat -o -f 0 -s -l 2 -h 100 -e 0.7 -m 1 -i {iterations} -V 0.2 -v xor_problem &")

os.system(
    f"./la2 -t ../../datasetsLA2IMC/dat/train_xor.dat -T ../../datasetsLA2IMC/dat/test_xor.dat -o -f 1 -s -l 2 -h 100 -e 0.7 -m 1 -i {iterations} -V 0.2 -v xor_problem &")

#################################################################################################### ILDPB PROBLEM

iterations = 1000 # For XOR and ILDPB problem, 500 for nominist problem
# Offline mode

os.system(
    f"./la2 -t ../../datasetsLA2IMC/dat/train_ildp.dat -T ../../datasetsLA2IMC/dat/test_ildp.dat -o -f 0 -l 2 -h 4 -e 0.7 -m 1 -i {iterations} -V 0.2 -v ildp_problem &")

os.system(
    f"./la2 -t ../../datasetsLA2IMC/dat/train_ildp.dat -T ../../datasetsLA2IMC/dat/test_ildp.dat -o -f 0 -s -l 2 -h 4 -e 0.7 -m 1 -i {iterations} -V 0.2 -v ildp_problem &")

os.system(
    f"./la2 -t ../../datasetsLA2IMC/dat/train_ildp.dat -T ../../datasetsLA2IMC/dat/test_ildp.dat -o -f 1 -s -l 2 -h 4 -e 0.7 -m 1 -i {iterations} -V 0.2 -v ildp_problem &")

#################################################################################################### NOMNIST PROBLEM

iterations = 500 # For XOR and ILDPB problem, 500 for nominist problem
# Offline mode

os.system(
    f"./la2 -t ../../datasetsLA2IMC/dat/train_nomnist.dat -T ../../datasetsLA2IMC/dat/test_nomnist.dat -o -f 0 -l 2 -h 64 -e 0.7 -m 1 -i {iterations} -V 0.2 -v nomnist_problem &")

os.system(
    f"./la2 -t ../../datasetsLA2IMC/dat/train_nomnist.dat -T ../../datasetsLA2IMC/dat/test_nomnist.dat -o -f 0 -s -l 2 -h 64 -e 0.7 -m 1 -i {iterations} -V 0.2 -v nomnist_problem &")

os.system(
    f"./la2 -t ../../datasetsLA2IMC/dat/train_nomnist.dat -T ../../datasetsLA2IMC/dat/test_nomnist.dat -o -f 1 -s -l 2 -h 64 -e 0.7 -m 1 -i {iterations} -V 0.2 -v nomnist_problem &")