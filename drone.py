import casadi
import casadi.tools as ctools
import numpy as np
import matplotlib.pyplot as plt


# Define model and get simulator.
Delta = .10
Nt = 25
Nx = 12
Nu = 4
 
def ode(x, w): #, extForcesRot = None, extForcesWind = None):
    '''
    x[0] -> x_ned
    x[1] -> y_ned
    x[2] -> h_ned ( = -z_ned)
    x[3] -> u
    x[4] -> v
    x[5] -> w
    x[6] -> phi (roll)
    x[7] -> theta (pitch)
    x[8] -> psi (yaw)
    x[9] -> p
    x[10] -> q
    x[11] -> r
    '''
    m = 2. #masa del cuadricoptero
    d = .24 #distancia desde el centro de masa al centro de un motor, tomada en el plano xy de Mf
    ro = 1.2 #densidad del aire
    b = 54.2e-6 #coeficiente de empuje de los rotores del quad
    Ix = 8.1e-3 #momento de inercia respecto a x en Mf
    Iy = 8.1e-3 #momento de inercia respecto a y en Mf
    Iz = 14.2e-3 #momento de inercia respecto a z en Mf
    Jr = 104e-6 #incercia rotacional de una helice
    eps = 1.1e-6 #factor de arrastre para yaw
    g = 9.80665 #gravedad
    
    mu=1e-5 #coeficiente de arrastre de los motores
    C=3e-4 #constante de friccion con el aire
    A=.2 #area total del cuadricoptero
    Ax=.05 #proyeccion del area del cuadricoptero sobre el plano yz de Mf
    Ay=.05 #proyeccion del area del cuadricoptero sobre el plano xz de Mf
    Az=.1 #proyeccion del area del cuadricoptero sobre el plano xy de Mf
    k_amort=1e-5 #constante de amortiguacion rotacional
    
    return [x[3]*np.cos(x[7])*np.cos(x[8]) + x[4] * (np.sin(x[6])*np.sin(x[7])*np.cos(x[8]) - np.cos(x[6])*np.sin(x[8])) + x[5] * (np.cos(x[6])*np.sin(x[7])*np.cos(x[8]) + np.sin(x[6])*np.sin(x[8])),
        x[3]*np.cos(x[7])*np.sin(x[8]) + x[4] * (np.sin(x[6])*np.sin(x[7])*np.sin(x[8]) + np.cos(x[6])*np.cos(x[8])) + x[5] * (np.cos(x[6])*np.sin(x[7])*np.sin(x[8]) - np.sin(x[6])*np.cos(x[8])),
        x[3]*np.sin(x[7]) - x[4]*np.sin(x[6])*np.cos(x[7]) - x[5]*np.cos(x[6])*np.cos(x[7]),
        x[11]*x[4] - x[10]*x[5] - g*np.sin(x[7]) - (mu/m)*x[3] - C*Ax*ro/(2*m) * x[3]*np.fabs(x[3]),
        x[9]*x[5] - x[11]*x[3] + g*np.sin(x[6])*np.cos(x[7]) - (mu/m)*x[4] - C*Ay*ro/(2*m) * x[4]*np.fabs(x[4]),
        x[10]*x[3] - x[9]*x[4] + g*np.cos(x[6])*np.cos(x[7]) - b/m * (w[0]**2 + w[1]**2 + w[2]**2 + w[3]**2) - C*Az*ro/(2*m) * x[5]*np.fabs(x[5]),
        x[9] + x[10]*np.sin(x[6])*np.tan(x[7]) + x[11]*np.cos(x[6])*np.tan(x[7]),
        x[10]*np.cos(x[6]) - x[11]*np.sin(x[6]),
        x[10]*np.sin(x[6])/np.cos(x[7]) + x[11]*np.cos(x[6])/np.cos(x[7]),
        (Iy-Iz)/Ix *x[10]*x[11] + d*b/(np.sqrt(2)*Ix) * (-w[0]**2 - w[1]**2 + w[2]**2 + w[3]**2) - k_amort*ro*A/Ix * x[9] + Jr/Ix * x[10]*(w[0]-w[1]+w[2]-w[3]),
        (Iz-Ix)/Iy *x[9]*x[11] + d*b/(np.sqrt(2)*Iy) * (w[0]**2 - w[1]**2 - w[2]**2 + w[3]**2) - k_amort*ro*A/Iy * x[10] - Jr/Iy * x[9]*(w[0]-w[1]+w[2]-w[3]),
        (Ix-Iy)/Iz *x[9]*x[10] + eps/Iz * (w[0]**2 - w[1]**2 + w[2]**2 - w[3]**2) - k_amort*ro*A/Iz * x[11]]
    


# Define symbolic variables.
x = casadi.SX.sym("x",Nx)
u = casadi.SX.sym("u",Nu)
xsp = casadi.SX.sym("xsp",Nx)

# Make integrator object.
ode_integrator = casadi.Function("f_int",
    casadi.daeIn(x=x,p=u),
    casadi.daeOut(ode=casadi.vertcat(ode(x,u))))
ode_cvodes_casadi = casadi.Integrator("F", "cvodes", ode_integrator, {"abstol":1e-8, "reltol":1e-8, "tf":Delta})


# Then get nonlinear casadi functions
# and rk4 discretization.
ode_casadi = casadi.Function("f",
    [x,u],[casadi.vertcat(ode(x,u))])
# ode_casadi.init()

[k1] = ode_casadi([x,u])
[k2] = ode_casadi([x + Delta/2*k1,u])
[k3] = ode_casadi([x + Delta/2*k2,u])
[k4] = ode_casadi([x + Delta*k3,u])
xrk4 = x + Delta/6*(k1 + 2*k2 + 2*k3 + k4)
ode_rk4_casadi = casadi.Function("f_rk4",[x,u],[xrk4])

Q = np.diag([100,100,100,0,0,0,0,0,0,0,0,0])
R = np.diag([10.0,10.0,10.0,10.0])
Q0 = Q.copy()*10

# Define stage cost and terminal weight.

# lfunc = (casadi.mul([x.T,Q,x])
#     + casadi.mul([u.T,R,u]))
# l = casadi.SXFunction("l",[x,u],[lfunc])
lfunc = (casadi.mul([(x-xsp).T,Q,(x-xsp)])
   + casadi.mul([u.T,R,u]))
l = casadi.Function("l",[x,u,xsp],[lfunc])

# Pffunc = casadi.mul([x.T,Q0,x])
# Pf = casadi.SXFunction("Pf",[x],[Pffunc])
Pffunc = casadi.mul([(x-xsp).T,Q0,(x-xsp)])
Pf = casadi.Function("Pf",[x,xsp],[Pffunc])

# Bounds on u.
uub = 2000.0
ulb = -2000.0

# Make optimizers.
x0 = np.array([0,0,3.0,0,0,0,0,0,0,0,0,0])

# Create variables struct.
var = ctools.struct_symSX([(
    ctools.entry("x",shape=(Nx,),repeat=Nt+1),
    ctools.entry("u",shape=(Nu,),repeat=Nt)
)])

# Create parameters struct
par = ctools.struct_symSX([(
    ctools.entry("xsp",shape=(Nx,),repeat=Nt+1)
)])

# Set bounds on variables
varlb = var(-casadi.inf)
varub = var(casadi.inf)
varguess = var(0)

# Adjust the relevant constraints.
for t in range(Nt):
    varlb["u",t,:] = ulb
    varub["u",t,:] = uub

# Now build up constraints and objective.
obj = casadi.SX(0)
con = []
for t in range(Nt):
    con.append(ode_rk4_casadi([var["x",t],
        var["u",t]])[0] - var["x",t+1])
#     obj += l([var["x",t],var["u",t]])[0]
# obj += Pf([var["x",Nt],])[0]
    obj += l([var["x",t],var["u",t],par["xsp",t]])[0]
obj += Pf([var["x",Nt],par["xsp",t]])[0]

# Build solver object.
con = casadi.vertcat(con)
conlb = np.zeros((Nx*Nt,))
conub = np.zeros((Nx*Nt,))

nlp = casadi.Function("NLP_Obj", casadi.nlpIn(x=var,p=par),
    casadi.nlpOut(f=obj,g=con))
solver = casadi.NlpSolver("NLP", "ipopt",nlp, {"print_level":2, "print_time":False, "linear_solver":'ma27'})

solver.setInput(conlb,"lbg")
solver.setInput(conub,"ubg")

# Now simulate.
Nsim = 150
times = Delta*Nsim*np.linspace(0,1,Nsim+1)
x = np.zeros((Nsim+1,Nx))
x[0,:] = x0

u = np.tile(np.array([212,212,212,212]),(Nsim,1))

x_sp = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
curr_par = par(0)
curr_par["xsp"] = x_sp

for t in range(Nsim):
    # Fix initial state.
    varlb["x",0,:] = x[t,:]
    varub["x",0,:] = x[t,:]
    varguess["x",0,:] = x[t,:]
    varguess["u",0,:] = u[t,:]
    
    solver.setInput(curr_par,"p")
    solver.setInput(varguess,"x0")
    solver.setInput(varlb,"lbx")
    solver.setInput(varub,"ubx")

    # Solve nlp.
    solver.evaluate()
    status = solver.getStat("return_status")
    optvar = var(solver.getOutput("x"))

    # Print stats.
    print("%d: %s" % (t,status))

    u[t,:] = np.array(optvar["u",0,:]).flatten()
    
    # Simulate.
    ode_cvodes_casadi.setInput(x[t,:],"x0")
    # ode_cvodes_casadi.setInput([215,215,215,215],"p")
    ode_cvodes_casadi.setInput(u[t,:],"p")
    ode_cvodes_casadi.evaluate()
    x[t+1,:] = np.array(
        ode_cvodes_casadi.getOutput("xf")).flatten()
    ode_cvodes_casadi.reset()


# Plots.
plt.close()
plt.ioff()
fig = plt.figure(0)
fig2 = plt.figure(1)
Nx = 3
numrows = max(Nx,Nu)
numcols = 2

# u plots. Need to repeat last element
# for stairstep plot.
u = np.concatenate((u,u[-1:,:]))
for i in range(Nu):
    ax = fig.add_subplot(numrows,
        numcols,numcols*(i+1))
    ax.step(times,u[:,i],"-k")
    ax.set_xlabel("Time")
    ax.set_ylabel("Control %d" % (i + 1))

# x plots.
for i in range(Nx):
    ax = fig.add_subplot(numrows,
        numcols,numcols*(i+1) - 1)
    ax.plot(times,x[:,i],"-k",label="System")
    ax.set_xlabel("Time")
    ax.set_ylabel("State %d" % (i + 1))

fig.tight_layout(pad=.5)


plt.show()