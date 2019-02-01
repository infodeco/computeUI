from __future__ import print_function
import dit
import time
from admUI import computeQUI

# Examples with probability distributions generated using dit
examples = [
    # XOR:
    dit.Distribution(['000', '011', '101', '110'], [1. / 4] * 4),
    # perturbed XOR:
    dit.Distribution(['000', '011', '101', '110'], [0.1, 0.2, 0.3, 0.4]),
    # AND:
    dit.Distribution(['000', '001', '010', '111'], [1. / 4] * 4),
    dit.Distribution(['012', '020', '101', '102', '120'],
                     [0.421568114489, 0.0786651096858, 0.253963103031,
                      0.160027829754, 0.08577584304]),
    dit.Distribution(['012', '020', '101', '102', '120'],
                     [0.4, 0.1, 0.25, 0.16, 0.09]),
    dit.Distribution(
        ['000', '001', '010', '011', '100', '101', '110', '111'],
        [0.033333348023, 0.0666666740115, 0.0666666740115, 0.133333303954,
         0.171428518128, 0.128571448728, 0.228571401696, 0.171428631448]),
    dit.Distribution(
        ['000', '001', '010', '011', '101', '110', '200', '201', '210'],
        [0.00464987943077, 0.0159455072032, 0.00161909206555, 0.00134153842824,
         0.452357030098, 0.186331210642,
         0.00651647516163, 0.170868608161, 0.16037065881]),
    dit.Distribution(['010', '100', '101', '110', '111', '121'],
                     [0.250766178235, 0.101750798968, 0.0920533030959,
                      0.0609473464827, 0.466445827817, 0.0280365454012]),
    # Warning: Maximum number of iterations reached in outer loop.
    dit.Distribution(['002', '011', '012', '100', '102', '111', '112', '122'],
                     [0.167748246356, 0.00391032994228, 0.351521384625,
                      0.0353228914823, 0.00898089799626, 0.404681231646,
                      0.0186113432744, 0.00922367467759]),
    # Warning: Maximum number of iterations reached in inner loop:
    dit.Distribution(['001', '010', '021', '110', '121', '201'],
                     [0.167967184186, 0.0205698172054, 0.110877895765,
                      0.137539239956, 0.185423915657, 0.37762194723]),
    dit.Distribution(['000', '010', '020', '102', '110', '122'],
                     [0.0869196091623, 0.0218631235533, 0.133963681059,
                      0.164924698739, 0.429533105427, 0.16279578206])]

# count some statistics
max_delta = 0.
total_time_admUI = 0.
total_time_dit = 0.

for d in examples:
    d.set_rv_names('SXY')
    print(d.to_dict())
# admUI
    start_time = time.time()
    Q = computeQUI(distSXY=d, DEBUG=True)
    admUI_time = time.time() - start_time
    total_time_admUI += admUI_time
    UIX = (dit.shannon.conditional_entropy(Q, 'S', 'Y')
           + dit.shannon.conditional_entropy(Q, 'X', 'Y')
           - dit.shannon.conditional_entropy(Q, 'SX', 'Y'))
    # UIX2 = (dit.shannon.entropy(Q, 'SY') + dit.shannon.entropy(Q, 'XY')
    #         - dit.shannon.entropy(Q, 'SXY') - dit.shannon.entropy(Q, 'Y'))
    # print(abs(UIX - UIX2) < 1e-10)
    UIY = (dit.shannon.conditional_entropy(Q, 'S', 'X')
           + dit.shannon.conditional_entropy(Q, 'Y', 'X')
           - dit.shannon.conditional_entropy(Q, 'SY', 'X'))
    SI = dit.shannon.mutual_information(Q, 'S', 'X') - UIX
    # SI2 = (dit.shannon.entropy(Q, 'S') + dit.shannon.entropy(Q, 'X')
    #        - dit.shannon.entropy(Q, 'SX') - UIX)
    # SI3 = (dit.shannon.entropy(Q, 'S') + dit.shannon.entropy(Q, 'X')
    #        + dit.shannon.entropy(Q, 'Y') - dit.shannon.entropy(Q, 'SX')
    #        - dit.shannon.entropy(Q, 'SY') - dit.shannon.entropy(Q, 'XY')
    #        + dit.shannon.entropy(Q, 'SXY'))
    CI = dit.shannon.mutual_information(d, 'S', 'XY') - UIX - UIY - SI
    # ##### CIQ should be close to zero:
    # CIQ = dit.shannon.mutual_information(Q, 'S', 'XY') - UIX - UIY - SI
    print(Q.to_dict())
# pid from dit
    start_time = time.time()
    dit_pid = dit.pid.PID_BROJA(d, ['X', 'Y'], 'S')
    dit_time = time.time() - start_time
    total_time_dit += dit_time

    dit_R = dit_pid.get_partial((('X', ), ('Y', )))
    dit_UX = dit_pid.get_partial((('X', ), ))
    dit_UY = dit_pid.get_partial((('Y', ), ))
    dit_CI = dit_pid.get_partial((('X', 'Y'), ))
    print("admUI: PID(R=", SI, ", UX=", UIX, ", UY=", UIY, ", S=", CI, ")",
          sep='')
    print("dit  : PID(R=", dit_R, ", UX=", dit_UX, ", UY=", dit_UY,
          ", S=", dit_CI, ")", sep='')
    print("--- admUI:", round(admUI_time, 3), "seconds --- dit:",
          round(dit_time, 3), "seconds ---")
    delta = [abs(dit_R - SI), abs(dit_UX - UIX), abs(dit_UY - UIY),
             abs(dit_CI - CI)]
    max_delta = max(max_delta, max(delta))
    print()

'''
# Example d12 without using dit
print("\nRunning admUI without dit on example d12...")
ns=2
nx=3
ny=2
P = np.array([0.0869196091623, 0, 0.0218631235533, 0, 0.133963681059,
              0, 0, 0.164924698739, 0.429533105427, 0, 0, 0.16279578206])
Psxy = np.reshape(P,[ns,nx,ny])
Psx = np.sum(Psxy,axis=2)
Psy = np.sum(Psxy,axis=1)
PS = np.sum(Psx,axis=1)
PXgS = np.divide(np.transpose(Psx), PS)
PYgS = np.divide(np.transpose(Psy), PS)
PS = PS.reshape((-1, 1))
Q = computeQUI_numpy(PXgS, PYgS, PS)
print("p=\n", Q)
'''

print("Maximal deviation between admUI and dit:", max_delta)
print("Average running time admUI:", total_time_admUI / len(examples),
      "seconds")
print("Average running time dit  :", total_time_dit / len(examples), "seconds")
