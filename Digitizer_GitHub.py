import numpy as np
from scipy.optimize import differential_evolution, NonlinearConstraint

'''Adding the temperature range with desired step size and their respective resistance values of the thermistor'''
'''T (degree centigrade) refer to the temperature in the range of 0 to 121 (exclusive) degree centigarde with a step size of 5 degree centigrade'''
'''RT (ohms) refer to the thermistor resistance values for the above temperatures'''
T = np.array(list(range(0, 121, 5)))
RT = np.array([36130, 28610, 22800, 18300, 14770, 12000, 9804, 8054, 6652, 5522, 4607, 3862, 3252, 
               2751, 2337, 1993, 1707, 1467, 1266, 1096, 952.4, 830.2, 726, 636.9, 560.4])

'''Integration Phase T2, when both the switches VSW1 and VSW2 are set to logic-high (H)'''
'''T2 (in µs) refers to the fixed pulse duration i.e., T2 = TP'''
T2 = 1e4

'''
defines the function that computes the ratio (say, V (= Dθ = (Tθ / T2))) 
'''
def Calculate_V(RT, T1, T2, Tm, Ra, Rb):
    vector1 = np.vectorize(np.int_)
    Tm1 = np.array((T1) * ((RT)/(RT+Ra))) + ((T2) * ((RT)/(RT+Rb)))
    Tm = vector1(Tm1)
    V = Tm/T2
    return(V)

'''
Computation of the best fit line (BFL) for the output equation of the triple-slope thermistor digitizer (say, V (= Dθ = (Tθ / T2)))
'''
def BFL(T, RT, T1, T2, Tm, Ra, Rb):

    vector1 = np.vectorize(np.int_)
    Tm1 = np.array((T1) * ((RT)/(RT+Ra))) + ((T2) * ((RT)/(RT+Rb)))
    Tm = vector1(Tm1)

    V = Tm/T2
    
    X = np.ones(shape = (len(T), 2))
    X[...,1] = np.array(T).transpose()
    b1 = np.linalg.inv(np.matmul(X.transpose(), X))
    b2 = np.matmul(b1, X.transpose())
    b  = np.matmul(b2, np.array(V).transpose())
    return (b.tolist())

'''
defines the function that computes the de-integration phase (say, Tm (= Tθ))
'''
def Calculate_Tm(RT, Ra, Rb, T1, T2):
    vector1 = np.vectorize(np.int_)
    Tm1 = np.array((T1) * ((RT)/(RT+Ra))) + ((T2) * ((RT)/(RT+Rb)))
    Tm = vector1(Tm1)
    return(Tm)

'''
defines the function that computes the conversion time, Tc
'''
def ConversionTime(RT, T1, T2, Ra, Rb):
    vector1 = np.vectorize(np.int_)
    Tm1 = np.array((T1) * ((RT)/(RT+Ra))) + ((T2) * ((RT)/(RT+Rb)))
    Tm = vector1(Tm1)

    Tc = T1 + T2 + Tm
    return(Tc)

'''
defines the function that computes the difference between Ra and Rb
'''
def Difference(RT, T1, T2, Ra, Rb):
    Diff = Ra - Rb
    return(Diff)

'''
defines the optimization problem i.e, objective function that needs to be minimized (%NL) and computes the decision variable vector.
R is a decision variable vector, R = [Ra, Rb, T1], at which minimum %NL can be achieved.
'''
def to_minimize(R):

    Ra = R[0]
    Rb = R[1]
    T1 = R[2]

    vector1 = np.vectorize(np.int_)
    Tm1 = np.array((T1) * ((RT)/(RT+Ra))) + ((T2) * ((RT)/(RT+Rb)))
    Tm = vector1(Tm1)
    
    V = Tm/T2

    b = BFL(T, RT, T1, T2, Tm, Ra, Rb)
    c, m = b

    y = m*T+c
    
    NLn = max(abs(V-y))
    NLd = abs(max(V)-min(V))
    NL = 100 * (NLn/NLd) 

    return (NL)

'''
defines the constraint function on the conversion time Tc
'''
def constr_f1(x):
    Ra, Rb, T1= x
    return np.array(ConversionTime(RT, T1, T2, Ra, Rb))

'''
defines the constraint function, Diff (= Ra - Rb) to be positive i.e., Ra > Rb
'''
def constr_f2(x):
    Ra, Rb, T1= x
    return np.array(Difference(RT, T1, T2, Ra, Rb))
  
'''upper bound on the conversion time (Tc) as 200000 µs'''  
nlc1 = NonlinearConstraint(constr_f1, 0, 200000)

'''upper bound on the D (= Ra - Rb) as 1000000'''
nlc2 = NonlinearConstraint(constr_f2, 0, 1e6)

'''nlc ensures that both the aforementioned non-linear constraints (nlc1, and nlc2) satisfies'''
nlc = (nlc1, nlc2)

'''
lower and upper limit/bound of search space for each optimal parameters
search space for Ra is [500, 1000000] Ω
search space for Rb is [500, 1000000] Ω
search space for T1 [1, 2000000] µs 
'''
bounds = [(0.5,1e3), (0.5,1e3), (1,200000)]

'''
differential evolution algorithm minimizes the function named "to_minimize", with the defined bounds and constraint
'''
'''
maxiter refer to the maximum iterations for the framed constrained optimization problem'''
'''
polish can be True or False. for the optimization problems with many constraints, polishing can take a long time 
due to the Jacobian computations. 
'''
'''
integrality - For each decision variable, a boolean value (0 or 1) indicating whether the decision variable is constrained to integer values.
if the integrality is 1 for the decision variable, only integer values lying between the lower and upper bounds are used for solving 
the optimization problem. 
'''
'''
integrality = [1, 1, 1]
This ensures that:
# - Ra is selected in integer steps (e.g., 1Ω resolution),
# - Rb is selected in integer steps (e.g., 1Ω resolution),
# - T1 is selected in integer steps (e.g., 1µs resolution). 
'''
result = differential_evolution(to_minimize, bounds, constraints=(nlc), maxiter = 1000, polish = False, integrality=[1,1,1])

'''prints the result'''
Ra, Rb, T1= result.x

vector1 = np.vectorize(np.int_)
Tm1 = np.array((T1) * ((RT)/(RT+Ra))) + ((T2) * ((RT)/(RT+Rb)))
Tm = vector1(Tm1)

Tc = T1 + T2 + Tm
V = Tm/T2

'''prints the below function returnable values'''
print(Calculate_V(RT, T1, T2, Tm, Ra, Rb))
print(BFL(T, RT, T1, T2, Tm, Ra, Rb))
print(result)