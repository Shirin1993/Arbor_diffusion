NEURON {
    SUFFIX calcium_decay
    USEION my_ca WRITE my_cad
    RANGE tau
}


PARAMETER { tau = 100 } 


BREAKPOINT {
   SOLVE dmy_ca METHOD sparse

}

DERIVATIVE dmy_ca {

   my_cad' = -my_cad/tau

}
