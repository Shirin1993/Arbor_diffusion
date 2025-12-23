import arbor
import matplotlib.pylab as plt
import matplotlib
import numpy as np
from arbor import units as U

delta_t = .01*U.ms
tau_ca = 100*U.ms

class recipe(arbor.recipe):
    def __init__(self, cell, probes):
        arbor.recipe.__init__(self)
        self.the_cell = cell
        self.the_probes = probes
        self.the_props = arbor.neuron_cable_properties()
        self.the_props.catalogue = arbor.default_catalogue()
        the_cat = arbor.load_catalogue("./inj/custom_inj-catalogue.so")  # load the catalogue of custom mechanisms
        
        the_cat_decay = arbor.load_catalogue("./dec/custom_dec-catalogue.so")  # load the catalogue of custom mechanisms
        self.the_props.set_ion("my_ca",valence=2, int_con=0*U.mM, ext_con=0*U.mM, rev_pot =0*U.mV )
        mch = the_cat['calcium_based_synapse']
        self.theta_p = (mch.parameters['theta_p'].default)
        self.theta_d = (mch.parameters['theta_d'].default)

        defcat = arbor.default_catalogue()
        defcat.extend(the_cat, '')
        defcat.extend(the_cat_decay, '')

        self.the_props.catalogue = defcat
        

    def num_cells(self):  # is necessary
        return 1

    def cell_kind(self, gid):
        return arbor.cell_kind.cable

    def cell_description(self, gid):
        return self.the_cell

    def probes(self, gid):
        return self.the_probes

    def global_properties(self, kind):
        return self.the_props
    def event_generators(self, gid):
        return [arbor.event_generator("syn0", 1., arbor.explicit_schedule(np.linspace(1,10,2)*U.ms)),
                arbor.event_generator("syn1", 0, arbor.explicit_schedule(np.array([])*U.ms) ),
                 arbor.event_generator("syn2", 1., arbor.explicit_schedule(np.linspace(1,10,2)*U.ms)),
                 arbor.event_generator("syn3", 0, arbor.explicit_schedule(np.array([])*U.ms))]


tree = arbor.segment_tree()
# Length are in um based on this : https://docs.arbor-sim.org/en/stable/fileformat/nmodl.html#units
dendrite_radius = 1  # um
dendrite_length = 80  # um

neck_radius = 1  # um
neck_length =  1  # um

spine_radius = 1.0  # um
spine_length = 1.0  # um

# custom tags are not allowed at the moment (cf. https://github.com/arbor-sim/arbor/pull/1996)
''' Dendrite : *1-----*2|-----*3    in x direction 
  *1 : -dendrite_length/2
  *2 :  0    
  *3 : +dendrite_length/2 '''

spine_x_pos = [-1.0,0.00, 1.0,3.0]
NumSpines = len(spine_x_pos)
j = 0
labels = arbor.label_dict({})

# ---------------- Defining location of dendritic segments , SPine and neck----------------
for i in range ( NumSpines+1):
    
    if i ==0:
        locals()["dendrite" + str(i)] = tree.append(arbor.mnpos,
                    arbor.mpoint(-dendrite_length/2, 0, 0, dendrite_radius),
                    arbor.mpoint(spine_x_pos[i], 0, 0, dendrite_radius), tag=i)
        labels['dendrite'+str(i)] = "(tag "+str(i)+")"
    if i == NumSpines:
        locals()["dendrite" + str(i)] = tree.append(locals()["dendrite" + str(i-1)],
                    arbor.mpoint(spine_x_pos[i-1], 0, 0, dendrite_radius),
                    arbor.mpoint(dendrite_length/2, 0, 0, dendrite_radius), tag=NumSpines)
        labels['dendrite'+str(i)] = "(tag "+str(NumSpines)+")"
    if i!=0 and i!= NumSpines:
        
        locals()["dendrite" + str(i)] = tree.append(locals()["dendrite" + str(i-1)],
                    arbor.mpoint(spine_x_pos[i-1], 0, 0, dendrite_radius),
                    arbor.mpoint(spine_x_pos[i], 0, 0, dendrite_radius), tag=i)
        labels['dendrite'+str(i)] = "(tag "+str(i)+")"
        
    if i!= NumSpines:
        j = j+1
        locals()["spine" + str(i)]  = tree.append(locals()["dendrite" + str(i)] ,
                           arbor.mpoint(spine_x_pos[i], 0, dendrite_radius, spine_radius),
                           arbor.mpoint(spine_x_pos[i], 0, dendrite_radius+spine_length, spine_radius), tag=i+j+NumSpines)
        labels['spine'+str(i)] = "(tag "+str(i+j+NumSpines)+")"



# ----------------morphology ----------------
morph = arbor.morphology(tree);

#print("morph.num_branches =", morph.num_branches)

# ---------------- Decor ----------------
decor = arbor.decor()
# ---------------- calcium diffusion properties ----------------
ca_diff = 220*1e-5*U.m2/U.s # diffusivity
decor.set_ion("my_ca",int_con=0.0*U.mM,  diff=ca_diff)
length_per_cv = 1 #*U.um
#decor.discretization(arbor.cv_policy(f'(max-extent {length_per_cv})'))
cv_policy = arbor.cv_policy(f'(max-extent {length_per_cv})')
##Custom decay mech
mech_decay=arbor.mechanism("calcium_decay/my_ca",{'tau' : tau_ca.value_as(U.ms)})
decor.paint('(all)', arbor.density(mech_decay))


# ---------------------- Injection of calcium ----------------------
for i in range(NumSpines):
    mech_inject = arbor.mechanism("calcium_based_synapse", {'delta_t':delta_t.value_as(U.us)})
    decor.place('(on-components 1 (region "spine' + str(i) + '"))',
                arbor.synapse(mech_inject), "syn" + str(i))

# Set up ion diffusion
# -------------------- --create probes ----------------------

probes = [arbor.cable_probe_ion_diff_concentration_cell("my_ca","tag_my_ca")]
probes_I =[arbor.cable_probe_point_state_cell("calcium_based_synapse", "I_syn", "tag_I_syn")]
#probe_area = [arbor.cable_probe_point_state(0,"calcium_based_synapse", "area0", "tag_area")]
probes_W =[arbor.cable_probe_point_state_cell("calcium_based_synapse", "W", "tag_W")]
probes_all = []
probes_all.extend(probes)
probes_all.extend(probes_I)
#probes_all.extend(probe_area)
probes_all.extend(probes_W)
# --------------- Defining cell and recipe and simulation  --------------

#cel = arbor.cable_cell(morph, decor, labels)
cel = arbor.cable_cell(morph, decor, labels, discretization=cv_policy)

rec = recipe(cel, probes_all)
sim = arbor.simulation(rec)
arbor.write_component(cel, 'morpho_2spines' + 'config1' + ".acc")
# ------------------Setting Handles-----------------------

offset = 0
T_regular_schedule = delta_t  
ca_prob_handle = sim.sample(0,"tag_my_ca", arbor.regular_schedule(T_regular_schedule))
I_prob_handle = sim.sample(0, "tag_I_syn", arbor.regular_schedule(T_regular_schedule))
A_prob_handle = sim.sample(0, "tag_area", arbor.regular_schedule(T_regular_schedule))
W_prob_handle = sim.sample(0, "tag_W", arbor.regular_schedule(T_regular_schedule))



## ----------------Run Simularion ----------------
Sim_time = 70*U.ms # ms. 
sim.run(tfinal=Sim_time, dt=delta_t)
mt = sim.probe_metadata(0,"tag_my_ca")[0]  #for Cal
mt2 = sim.probe_metadata(0, "tag_I_syn")[0] # for I_syn
mt3 = sim.probe_metadata(0, "tag_area")[0] # for area
mt4 = sim.probe_metadata(0, "tag_W")[0] # for weight

##___________________Ca_______________________
fig, axes = plt.subplots(4, sharex=True)
vmax = 0.8
line_color = '#000080'
data, meta = sim.samples(ca_prob_handle)[0]
theta_p = 1e3*rec.theta_p 
theta_d = 1e3*rec.theta_d 
xlim_0 = -10;xlim_1 = 7000
#xlim_ = xlim_1 - xlim_0
ylim_ = max(1e3*data.T[39+1])
for i,sp in enumerate([39,41,43,46]): #len(mt)
    if sp==39:
        label_com = 'spine1'
        np.savetxt(f'ACaS0_{delta_t.value_as(U.ms)}.txt',data.T[sp+1])
    if sp==41:
        label_com = 'spine2'
        np.savetxt(f'ACaS1_{delta_t.value_as(U.ms)}.txt',data.T[sp+1])
    if sp==43:
        label_com = 'spine3'
        np.savetxt(f'ACaS2_{delta_t.value_as(U.ms)}.txt',data.T[sp+1])
    if sp==46:
        label_com = 'spine4'  
        np.savetxt(f'ACaS3_{delta_t.value_as(U.ms)}.txt',data.T[sp+1])
    axes[i].plot([theta_d]*(len(data[:,sp+1])))
    axes[i].plot([theta_p]*(len(data[:,sp+1])))
    axes[i].plot(1e3*data[:,sp+1], label = label_com )
    axes[i].legend(loc ='upper right')
    axes[i].set_xlim(xlim_0,xlim_1)
    axes[i].set_ylim(0,ylim_)
plt.savefig('plot_D = '+str(ca_diff.value_as(U.m2/U.s))+'.svg')

# _____________Dendritic segments _________________
fig, axes = plt.subplots(5, sharex=True)
vmax = 0.8
line_color = '#000080'
data, meta = sim.samples(ca_prob_handle)[0]

for i,dn in enumerate([40,42,45]): #len(mt)
    if dn==40:
        label_com = 'dend1'
        np.savetxt(f'ACaD0_{delta_t.value_as(U.ms)}.txt',data.T[dn+1])
    if dn==42:
        label_com = 'dend2'
        np.savetxt(f'ACaD1_{delta_t.value_as(U.ms)}.txt',data.T[dn+1])
    if dn==45:
        label_com = 'dend3'
        np.savetxt(f'ACaD2_{delta_t.value_as(U.ms)}.txt',data.T[dn+1])


    axes[i].plot(1e3*data[:,dn+1], label = label_com )  
    axes[i].legend(loc ='upper right')
    axes[i].set_xlim(xlim_0,xlim_1)
    axes[i].set_ylim(0,ylim_)
plt.savefig('Dend_plot_D = '+str(ca_diff.value_as(U.m2/U.s))+'.svg')

##___________________Current___________________

fig, axes = plt.subplots(NumSpines, sharex=True)
vmax = 0.8
line_color = '#000080'
data, meta = sim.samples(I_prob_handle)[0]

ylim_ = 1
for i in range(0,len(mt2)): #NSpines
    label_com= '?'
    if i==0:
        label_com = 'Ispin_1'  
        np.savetxt(f'AIS0_{delta_t.value_as(U.ms)}.txt',data.T[i+1])
    if i==1:
        label_com = 'Ispin_2'
        np.savetxt(f'AIS1_{delta_t.value_as(U.ms)}.txt',data.T[i+1])
    if i==2:
        label_com = 'Ispin_3'  
        np.savetxt(f'AIS2_{delta_t.value_as(U.ms)}.txt',data.T[i+1])
    if i==3:
        label_com = 'Ispin_4'
        np.savetxt(f'AIS3_{delta_t.value_as(U.ms)}.txt',data.T[i+1])
    axes[i].set_xlim(xlim_0,xlim_1)
    axes[i].set_ylim(0,ylim_)
    axes[i].plot(data[:,i+1], label = label_com)
    axes[i].legend(loc ='upper right')
plt.savefig('I = '+str(ca_diff.value_as(U.m2/U.s))+'.svg')
##___________________Weight___________________

fig, axes = plt.subplots(NumSpines, sharex=True)
vmax = 0.8
line_color = '#000080'
data, meta = sim.samples(W_prob_handle)[0]

ylim_ =2
for i in range(0,len(mt4)): #NSpines
    label_com= '?'
    if i==0:
        label_com = 'Wspin_1'  
        np.savetxt(f'AWS0_{delta_t.value_as(U.ms)}.txt',data.T[i+1])
    if i==1:
        label_com = 'Wspin_2'
        np.savetxt(f'AWS1_{delta_t.value_as(U.ms)}.txt',data.T[i+1])
    if i==2:
        label_com = 'Wspin_3'  
        np.savetxt(f'AWS2_{delta_t.value_as(U.ms)}.txt',data.T[i+1])
    if i==3:
        label_com = 'Wspin_4'
        np.savetxt(f'AWS3_{delta_t.value_as(U.ms)}.txt',data.T[i+1])
#    axes[i].set_xlim(xlim_0,xlim_1)
#    axes[i].set_ylim(0,ylim_)
    axes[i].plot(data[:,i+1], label = label_com)
    axes[i].legend(loc ='upper right')

plt.savefig('W = '+str(ca_diff.value_as(U.m2/U.s))+'.svg')
