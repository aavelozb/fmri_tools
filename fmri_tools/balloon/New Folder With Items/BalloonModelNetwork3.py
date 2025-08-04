# adapted from https://github.com/rmillin/balloon-model/ for simulating netwroks
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import ode, solve_ivp
import random as rn
from sklearn.preprocessing import StandardScaler


def plot_signals(timeline, S, title=''):
    Nt, Nnodes = S.shape
    fig, ax = plt.subplots(Nnodes, 1, figsize=(12,4))
    for inode in range(Nnodes):
        ax[inode].plot(timeline, S[:, inode])
        ax[inode].set_ylabel(r'$n_{'+f'{inode+1}'+r'}$')
        if len(title)>0: ax[0].set_title( title )
        ax[-1].set_xlabel('time')


def generateConnectivityMatrix(Nnds=5, Nconn=3):
    C = np.eye(Nnds)
    node_pairs = []
    for i in range( Nnds ):
        for j in range( Nnds ):
            if i > j:
                node_pairs.append( (i, j) ) 
    nodes_connected = []
    while len(nodes_connected) < Nconn:
        n = rn.choice( node_pairs )
        if n not in nodes_connected:
            nodes_connected.append( n )
    for i, j in nodes_connected:
        C[i, j] = rn.choice(np.arange(0.5,0.9,0.1))
    return C


def generateNodeEvents(timeline, nblocks=5, duration_range=(1,3)):
    N_pts = len(timeline)
    time_offset = 10
    possible_onsets = np.arange(time_offset, N_pts-time_offset)
    sep_bwn_ons = 10
    durations = []
    onsets = []    
    for i in range(nblocks):
        current_block_duration = rn.randint(duration_range[0], duration_range[1])
        current_onset = rn.choice( possible_onsets )
        onsets.append( current_onset )
        durations.append( current_block_duration )
        possible_onsets = possible_onsets[np.logical_or( possible_onsets < current_onset - sep_bwn_ons,
                                                possible_onsets > current_onset + sep_bwn_ons)]
    u = np.zeros(N_pts)
    for i,d in zip(onsets, durations):
        u[i:i+d] = 1
    return u, timeline, onsets, durations


def generateNetworkEvents(timeline, Nnds, nblocks=5, duration_range=(1,3)):
    N_pts = len(timeline)
    U = np.zeros(shape=(N_pts,Nnds))
    for n in range( Nnds ):
        U[:,n], _, _, _ = generateNodeEvents(timeline, nblocks, duration_range)
    return U

# Eq14 Buxton 
def neureq(t, I, params): 
    kappa1, tau, N0, interpolators, A, C = params
    ut = np.array([f(t) for f in interpolators])

    N = ut - C @ I
    # N = ut - I
    if np.any(N > -N0):
        dIdt = (kappa1/tau) * N - A@I
    else:
        dIdt = (-kappa1/tau) * N0 - A@I # neural response cannot go below 0
    return dIdt


# make the function that converts neural response to flow in the vasculature
def floweq(t, y, params):
    x1, x2 = y
    kappa2, gamma, neuronalact, timing = params
    neuronal = np.interp(t, timing, neuronalact)
    derivs = [x2, neuronal-x2*kappa2-(x1-1)*gamma]
    return derivs


# system of differential equations for balloon model
def balloonsystemeq(t, y, params): 
    x1, x2 = y
    tau1, tau2, alpha, E0, flowin, timing = params
    f = interp1d(timing, flowin)
    fin = f(t)
    E = 1 - (1 - E0)**(1/fin) # equation for oxygen extraction
    return [(1/tau1) * (fin*E/E0 - x1/x2*(x2**(1/alpha) + tau2/(tau1+tau2)*(fin-x2**(1/alpha)))),
             1/(tau1+tau2)*(fin - (x2**(1/alpha)))]


class BalloonModel():
    
    def __init__(self, timing, u, C=None):

        if u.ndim == 1: u = u[:, np.newaxis]

        self.Nt, self.Nnds = u.shape
        # self.u = u # events

        self.timing = timing # timeline
        self.newdeltat = 1
        self.newtiming = np.arange(min(self.timing), max(self.timing), self.newdeltat)

        if C is None:
            print('No C provided, assigning the identity')
            self.C = np.eye(self.Nnds)
        else:
            self.C = C

        self.u_intrinsic = u.copy()
        self.u = u.copy() #np.transpose(self.C @ self.u.T)

        ######### Balloon Model Parameters #########

        # Stimulus to Neural
        self.const = 1 
        self.kappa1 = 2 # (0-3)
        self.tau = 2 # (1-3)
        self.N0 = 0

        # Neural to Flow in
        self.kappa2 = .65; # prior from the paper: 0.65
        self.gamma = .41; # prior from the paper: 0.41
        # initial conditions for ODE solver
        self.f0 = 1. # flow in to vasculature
        self.s0 = 0. # signal to the vasulature

        # Balloon
        self.TE = 0.03
        self.alpha = .4 # 0.32 in Friston
        self.E0 = 0.4
        self.V0 = 0.03 # 0.03 in Buxton, Uludag, et al.
        self.F0 = 0.01
        self.tau2 = 30 # typical value based on fits from Mildner, Norris, Schwarzbauer, and Wiggins (2001)
        self.B0 = 3 # 3T scanner
        self.tau1 = self.V0/self.F0

        if self.B0 == 3: self.r0 = 108
        elif self.B0 == 1.5: self.r0 = 15
        else: self.r0 = 25 * (self.B0/1.5)^2 # Pinglei
        # assuming dominance of macrovascular component
        if self.B0 == 3 or self.B0 == 1.5: self.epsilon = 0.13 
        v = 40.3 * (self.B0 / 1.5)
        self.k1 = 4.3 * v * self.E0 * self.TE
        self.k2 = self.epsilon * self.r0 * self.E0 * self.TE
        self.k3 = 1 - self.epsilon
        # initial conditions for ODE solver
        self.q0 = 1
        self.v0 = 1

        return
    

    # make the neural model function    
    def StimulusToNeural(self):

        interpolators = [interp1d(self.timing, self.u[:, j], kind='nearest') for j in range(self.Nnds)]

        A = (1/self.tau) * np.eye(self.Nnds)
        I0 = -self.N0

        params = [self.kappa1, self.tau, self.N0, interpolators, A, self.C] # Bundle parameters for ODE solver
        sol = solve_ivp(neureq, [min(self.timing), max(self.timing)], self.Nnds*[I0], t_eval=self.timing, args=(params,)) # ODE solver

        I = sol.y.T
        newstim = np.concatenate([f(sol.t)[:, np.newaxis] for f in interpolators], axis=1)
        neur = newstim - I
        neur[neur<0] = 0 # imposes that N0+N>=0
        self.t_neural, self.neural = sol.t, neur

        return 

    # # make the neural model function    
    # def StimulusToNeural(self):

    #     newu_list = list()
    #     for j in range(self.Nnds):
    #         f_interp = interp1d(self.timing, self.u[:, j], kind='nearest')
    #         newu = f_interp(self.newtiming)
    #         newu_list.append(newu[:, np.newaxis] * self.const)
    #     self.newu = np.concatenate( newu_list, axis=1 )

    #     interpolators = [interp1d(self.newtiming, self.newu[:, j], kind='nearest') for j in range(self.Nnds)]

    #     A = (1/self.tau) * np.eye(self.Nnds)
    #     I0 = -self.N0

    #     t = self.newtiming # time array for solution
    #     params = [self.kappa1, self.tau, self.N0, interpolators, A, self.C] # Bundle parameters for ODE solver
    #     sol = solve_ivp(neureq, [min(t), max(t)], self.Nnds*[I0], args=(params,)) # ODE solver

    #     I = sol.y.T
    #     newstim = np.concatenate([f(sol.t)[:, np.newaxis] for f in interpolators], axis=1)
    #     neur = newstim - I
    #     neur[neur<0] = 0 # imposes that N0+N>=0
    #     self.t_neural, self.neural = sol.t, neur

    #     return 



    def NeuraltoFlowin(self):

        if self.Nnds>1:
            self.flowin = list()
            self.vascular = list()
            for inode in range(self.Nnds):
                neural_signal = self.neural[:, inode]
                params = [self.kappa2, self.gamma, neural_signal, self.t_neural] # Bundle parameters for ODE solver
                y0 = [self.f0, self.s0] # Bundle initial conditions for ODE solver
                t, flowin, vascular = self.NeuraltoFlowinOneNode(self.t_neural, params, y0)
                self.flowin.append(flowin)
                self.vascular.append(vascular)
                # print(f'node {inode+1}')
            self.flowin = np.column_stack(self.flowin)
            self.vascular = np.column_stack(self.vascular)
            self.t_flowin_vascular = t
        
        return


    def NeuraltoFlowinOneNode(self, timing, params, y0):
        
        t = timing # time array for solution
        solver = ode(floweq).set_integrator('dopri5') # ODE solver
        solver.set_initial_value(y0).set_f_params(params)
        k = 0
        soln = [y0]
        while solver.successful() and solver.t < t[-1]:
            k += 1
            solver.integrate(t[k])
            soln.append(solver.y)
        psoln = np.array(soln) 
        flowin = psoln[:,0]
        vascular = psoln[:,1]

        return t, flowin, vascular
    

    def FlowinToBold(self):

        if self.Nnds>1:
            self.bold = list()
            self.q_signal = list()
            self.v_signal = list()
            for inode in range(self.Nnds):
                flowin_signal = self.flowin[:, inode]                
                params = [self.tau1, self.tau2, self.alpha, self.E0, flowin_signal, self.t_flowin_vascular] # Bundle parameters for ODE solver
                y0 = [self.q0, self.v0] # Bundle initial conditions for ODE solver
                t, bold_signal, q_signal, v_signal = self.FlowinToBoldOneNode(self.t_flowin_vascular, params, y0)
                self.bold.append(bold_signal)
                self.q_signal.append(q_signal)
                self.v_signal.append(v_signal)
                # print(f'node {inode+1}')
            self.bold = np.column_stack(self.bold)
            self.q_signal = np.column_stack(self.q_signal)
            self.v_signal = np.column_stack(self.v_signal)
            self.t_bold = t
        
        return


    def FlowinToBoldOneNode(self, timing, params, y0):

        tInc = 0.1
        t = np.arange(timing[0], timing[-1], tInc) # time array for solution
        solver = ode(balloonsystemeq).set_integrator("dop853") # ODE solver
        solver.set_initial_value(y0).set_f_params(params)
        k = 0
        soln = [y0]
        while solver.successful() and solver.t < t[-1]:
            k += 1
            solver.integrate(t[k])
            soln.append(solver.y)
        psoln = np.array(soln) # list to a numpy array
        q_signal = psoln[:,0]
        v_signal = psoln[:,1]
        bold_signal = self.V0 * (self.k1 * (1 - q_signal) + self.k2 * (1 - q_signal/v_signal) + self.k3 * (1 - v_signal))
        t_bold = t
        return t_bold, bold_signal, q_signal, v_signal


    def add_noise(self):
        self.bold_noisy = self.bold.copy()
        self.bold_noisy = self.bold_noisy + 0.01 * np.random.randn(*self.bold_noisy.shape)
        self.bold_noisy = StandardScaler().fit_transform(self.bold_noisy)
        # print(self.bold_noisy.mean(axis=0))
        # print(self.bold_noisy.std(axis=0))


    def plot_signals(self, which_one='BOLD', add_events=True):
        if which_one == 'neural': timeline, S = self.t_neural, self.neural
        elif which_one == 'flow in': timeline, S = self.t_flowin_vascular, self.flowin
        elif which_one == 'vascular': timeline, S = self.t_flowin_vascular, self.vascular
        elif which_one == 'q(t)': timeline, S = self.t_bold, self.q_signal
        elif which_one == 'v(t)': timeline, S = self.t_bold, self.v_signal
        elif which_one == 'BOLD': timeline, S = self.t_bold, self.bold
        elif which_one == 'BOLD noisy': timeline, S = self.t_bold, self.bold_noisy

        fig, ax = plt.subplots(self.Nnds, 1, figsize=(12,4))
        for inode in range(self.Nnds):
            ax[inode].plot(timeline, S[:, inode])
            ax[inode].set_ylabel(r'$n_{'+f'{inode+1}'+r'}$')
            ax[0].set_title( which_one )
            ax[-1].set_xlabel('time')
            if add_events:
                [ax[inode].axvline(x=i, color='red', alpha=0.5) for i in range(self.Nt) if self.u_intrinsic[i, inode]!= 0] # linestyle='--'