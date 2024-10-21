
NEURON {
    POINT_PROCESS calcium_based_synapse 
    RANGE W_0, tau, theta_p, gamma_p, theta_d, gamma_d, beta, alpha, Volumee, length_sp, radi, I_syn ,delta_t 
    USEION my_ca WRITE my_cad
    NONSPECIFIC_CURRENT I
}


STATE {
    W
    I_syn
    my_ca
}

PARAMETER {

    W_0 = 0.5
    tau = 1:8000 :ms
    tau_syn = 1 
    gamma_p = 90 
    gamma_d = .01 

    theta_p = .00011
    theta_d = .00005
    gamma_rate= .11
    Faraday = 96485.309 : 
    I0 = 4.0 : 1pA
    length_sp = 1 (um) 
    radi = 1 (um) 
    delta_t = -1 (us)
    area
    c = 1


}

ASSIGNED {
    one_over_tau
    beta
    alpha 
    Volumee
    area0
}

INITIAL {
    beta = 0 
    my_ca = 0
    W = W_0
    one_over_tau = 1/tau
    Volumee= radi*3.14*radi*length_sp
    area0 = area
    I_syn = 0
}

BREAKPOINT {
    SOLVE state METHOD stochastic  
    alpha = (gamma_rate/(2*Faraday))*area/(1000*Volumee)  
    I = -I_syn
    beta = I_syn*alpha*delta_t*W   
    my_cad = my_cad + beta 


}


DERIVATIVE state {
    LOCAL hsp
    LOCAL hsd
    hsp = step_right(my_cad - theta_p)
    hsd = step_right(my_cad - theta_d)
    
    W' = ( gamma_p*(1-W)*hsp  - gamma_d*W*hsd ):*one_over_tau 
    I_syn' = -I_syn/tau_syn
}



NET_RECEIVE(weight) {  
    : only at spike time
    I_syn = I_syn + I0
    
}


