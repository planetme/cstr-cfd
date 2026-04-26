import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
from scipy.interpolate import griddata
from datetime import datetime

# Power law fitting of LDPE melt shear thinning properties at 220C to real data:
# https://doi.org/10.1002/pen.25333

data = [ # 1/s , Pa*s
    (6561, 15),
    (2187, 31),
    (729, 62),
    (243, 115),
    (81, 205),
    (27, 362),
    (9, 624),
]

ln_mu = np.log([mu for _, mu in data])
ln_gamma_dot = np.log([gamma_dot for gamma_dot, _ in data])

# Fit a line to the log-log data
coefficients = np.polyfit(ln_gamma_dot, ln_mu, 1)
n = coefficients[0] + 1  # Power law index
K = np.exp(coefficients[1])  # Consistency index

print(f"Power law index (n): {n}")
print(f"Consistency index (K): {K}")

# Plotting the data and the fitted curve
ln_gamma_dot_fit = np.linspace(min(ln_gamma_dot), max(ln_gamma_dot), 100)
ln_mu_fit = coefficients[0] * ln_gamma_dot_fit + coefficients[1]

plt.figure(1)
plt.scatter(ln_gamma_dot, ln_mu, label='Data Points', color='red')
plt.plot(ln_gamma_dot_fit, ln_mu_fit, label=f'Fitted Line, n={n:.2f}, K={K:.2f}', color='blue')
plt.xlabel('ln(dv/dt)')
plt.ylabel('ln(μ)')
plt.title('Power Law Fit of LDPE Melt Shear Thinning Properties')
plt.legend()

# Construction of finite difference model
def log(message):
    time = datetime.now().strftime("%H:%M:%S")
    print(f'[{time}] {message}')

log("Constructing finite difference model for CSTR flow...")

# 1. Geometry of CSTR

r_min, r_max = 0.1, 1 # m, radial domain of interest (from shaft center to vessel wall)
dr = 0.01 # m, radial resolution
r_blade_tip = 0.75 # m, blade tip radius
theta_min, theta_max = 0, np.pi # radians, angular domain of interest (half-circle around shaft, we can use symmetry here)
dtheta = np.pi / 180 # 1 degree resolution
theta_blade_tip = np.pi / 12 # radians, blade tip angle (5 degrees)

# 2. Construct spatial meshes

N_r = int((r_max - r_min) / dr) + 1 # number of nodes
r_arr = np.linspace(r_min, r_max, N_r) # instantiate node positions

N_theta = int((theta_max - theta_min) / dtheta) + 1 # number of nodes
theta_arr = np.linspace(theta_min, theta_max, N_theta) # instantiate node positions

r, theta = np.meshgrid(r_arr, theta_arr, indexing='ij') # create 2D mesh

# 3. Mask solid and BC regions

is_solid = (r <= r_blade_tip) & ((theta < theta_blade_tip) | (theta > np.pi - theta_blade_tip)) # mask out blade regions
is_wall = (r == r_max) # mask out vessel wall region
is_shaft = (r == r_min) # mask out shaft region
is_symmetry = (theta == theta_min) | (theta == theta_max) # mask out symmetry planes
adj_r_plus = np.roll(is_solid, +1, axis=0) & (~is_solid) # r values at adjacent nodes in +r direction, useful for tip of blade BC
adj_r_minus = np.roll(is_solid, -1, axis=0) & (~is_solid) # r values at adjacent nodes in -r direction
adj_theta_plus = np.roll(is_solid, +1, axis=1) & (~is_solid) # theta values at adjacent nodes in +theta direction, useful for blade edge BC
adj_theta_minus = np.roll(is_solid, -1, axis=1) & (~is_solid) # theta values at adjacent nodes in -theta direction, useful for blade edge BC
is_blade_adj = (adj_r_plus | adj_r_minus | adj_theta_plus | adj_theta_minus) # mask out nodes adjacent to blade, which will have BCs due to no penetration/no slip.
is_interior = ~(is_solid | is_wall | is_shaft | is_symmetry) # mask out interior fluid region

# Priority map of masks

node_type = np.zeros((N_r, N_theta), dtype=int)
node_type[is_interior] = 0
node_type[is_solid] = 1
node_type[is_wall] = 2
node_type[is_shaft] = 3
node_type[is_blade_adj] = 4
node_type[is_symmetry] = 5  # overwrites corners

# Plot node type map for verification

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.pcolormesh(theta_arr, r_arr, node_type)
ax.set_thetamin(0)
ax.set_thetamax(180)
ax.set_title('Node type map')

# 4. Initialize velocity field, "first guess" based on BCs for the later Picard iteration.
log("Initializing velocity field based on BCs for Picard iteration...")

omega = np.pi / 3 # rad/s, angular velocity of wall for stationary agitator reference frame (10 RPM)

v_r = np.zeros((N_r, N_theta)) # m/s, radial velocity component
v_theta = np.zeros((N_r, N_theta)) # m/s, angular velocity component

v_r[is_wall] = 0.0 # no penetration at wall
v_theta[is_wall] = -omega * r_max # no slip at wall
v_r[is_shaft] = 0.0 # no penetration at shaft
v_theta[is_shaft] = 0.0 # no slip at shaft
v_r[is_blade_adj] = 0.0 # no penetration at blade
v_theta[is_blade_adj] = 0.0 # no slip at blade
v_r[is_solid] = 0.0 # no flow in solid
v_theta[is_solid] = 0.0 # no flow in solid

# Use Couette stokes flow solution as initial guess for interior nodes, which is a reasonable approximation for this flow and should help speed up convergence of the Picard iteration.
# Analytical Couette flow initial condition
C1 = -omega * r_max**2 / (r_max**2 - r_min**2)
C2 = -C1 * r_min**2

v_theta = C1*r + C2/r   # smooth profile satisfying both BCs
v_r = np.zeros((N_r, N_theta))

# Re-enforce BCs
v_theta[is_wall]  = -omega * r_max
v_theta[is_shaft] = 0.0
v_theta[is_solid] = 0.0
v_theta[is_blade_adj] = 0.0

# 5. Conduct Picard iteration until convergence
# Since velocity is dependent on the viscosity field which itself is dependent on the shear rate/velocity field, we will need to iteratively solve this by lagging the viscosity field one step behind the velocity field.

# Picard iteration parameters
max_iterations = 500
tolerance = 1e-6

for iteration in range(1, max_iterations + 1):

    # 6. Computing shear rate and viscosity field from velocity field
    # Begin with computing rate of strain tensor/derivatives components from velocity field using finite difference approximations. We will use central differences for interior nodes.

    # First derivative arrays for interior spatial nodes (shape N_r-2 x N_theta-2)
    dvr_dr = (v_r[2:, 1:-1] - v_r[:-2, 1:-1]) / (2*dr) # radial normal component of rate of strain tensor, outputs arrays with [(v_r_r2 - v_r_r0) / (2*dr), (v_r_r3 - v_r_r1) / (2*dr), ...] for all theta values at interior nodes.
    dvr_dth = (v_r[1:-1, 2:] - v_r[1:-1, :-2]) / (2*dtheta) # angular normal component
    dvth_dr = (v_theta[2:, 1:-1] - v_theta[:-2, 1:-1]) / (2*dr) # radial tangential component
    dvth_dth = (v_theta[1:-1, 2:] - v_theta[1:-1, :-2]) / (2*dtheta) # angular tangential component

    # Compute shear rate array from rate of strain tensor components for interior nodes.
    r_int = r[1:-1, 1:-1] # m, matches r matrix dimensions to the derivative matrices
    v_r_int = v_r[1:-1, 1:-1] # m/s
    v_theta_int = v_theta[1:-1, 1:-1] # m/s

    Gamma_rr = dvr_dr # radial normal stress
    Gamma_thth = (dvth_dth + v_r_int) / r_int # angular normal stress
    Gamma_rth = 0.5 * (dvth_dr - (v_theta_int - dvr_dth) / r_int)

    shear_rate = np.sqrt(2 * (Gamma_rr**2 + Gamma_thth**2 + 2 * Gamma_rth**2)) # s^-1, magnitude of shear rate from second invariant of rate of strain tensor. 2x Gamma_rth^2 since it's off-diagonal and the ROS tensor is symmetric.
    gamma_dot = np.maximum(shear_rate, 1) # clip to avoid viscosity blowup to zero shear due to power law. 1 s^-1 is a reasonable minimum shear rate for this flow given the geometry and velocities.
    gamma_dot_min = 1.0  # s^-1, physically reasonable minimum
    mu = K * gamma_dot**(n - 1) # Pa*s, power law viscosity field

    # Fill boundary nodes for mu field (shape N_r x N_theta)
    mu_full = np.zeros((N_r, N_theta))
    mu_full[1:-1, 1:-1] = mu # fill interior nodes with computed viscosity values
    mu_full[0,:] = mu_full[1, :] # fill shaft nodes with adjacent interior node viscosity values
    mu_full[-1,:] = mu_full[-2, :] # fill wall nodes with adjacent interior node viscosity values
    mu_full[:, 0] = mu_full[:, 1] # fill symmetry nodes with adjacent interior node viscosity values
    mu_full[:, -1] = mu_full[:, -2] # fill symmetry nodes with adjacent interior node viscosity values 
    mu_full[is_solid] = 0

    # Calculate viscosity derivatives for interior nodes for solving the Cauchy momentum equation.
    dmu_dr = (mu_full[2:, 1:-1] - mu_full[:-2, 1:-1]) / (2*dr) # radial derivative of viscosity
    dmu_dth = (mu_full[1:-1, 2:] - mu_full[1:-1, :-2]) / (2*dtheta) # angular derivative of viscosity

    # Density for momentum balance
    rho = 920 # kg/m^3, density of LDPE melt at 220C

    # 7. Solve the coupled momentum equations for the velocity field with the viscosity field calculated from the previous iteration.
    # The coupled system can be described with the coefficient matrix A hold contributions from stress divergence, Coriolis, the modified pressure gradient, and the continuity equation.
    # We will solve for three fields, v_r, v_theta, and our centrifugually modified pressure P*. This system is homogeneous (no source terms) since we are solving for the velocity field that satisfies the momentum equations given the viscosity field, and we will enforce the BCs separately.

    # We will use a sparse matrix solver since the coefficient matrix K will contain mostly zeros.
    N = N_r * N_theta # total number of nodes
    A = sp.lil_matrix((3*N, 3*N)) # initialize sparse coefficient matrix in list of lists (LIL) format for efficient construction. Each list entry corresponds to a row in the matrix with the coefficients for the unknowns at each nodes consistent with the discretized momentum equations.
    b = np.zeros(3*N) # initialize right hand side vector, which is zero since we are solving for the velocity field that satisfies the homogeneous momentum equations given the viscosity field.

    
    # Neighbor flattened indices getter functions
    def vr(di, dj):
            ii, jj = i+di, j+dj
            assert 0 <= ii < N_r and 0 <= jj < N_theta, f"Index out of bounds: ({ii},{jj})" # for debug purposes lol
            return 0*N + ii*N_theta + jj
    
    def vth(di, dj):
            ii, jj = i+di, j+dj
            assert 0 <= ii < N_r and 0 <= jj < N_theta, f"Index out of bounds: ({ii},{jj})"
            return 1*N + ii*N_theta + jj
    
    def p(di, dj):
            ii, jj = i+di, j+dj
            assert 0 <= ii < N_r and 0 <= jj < N_theta, f"Index out of bounds: ({ii},{jj})"
            return 2*N + ii*N_theta + jj

    for i in range(N_r):
        for j in range(N_theta):
            # We will use the k indices to flatten the 2D node indices into a 1D index, similar to Voigt notation for stress/strain tensors. This will enable us to solve the linear system.
            k_r = 0*N + i*N_theta + j # index for v_r unknown at node (i,j)
            k_th = 1*N + i*N_theta + j # index for v_theta unknown
            k_p = 2*N + i*N_theta + j # index for P* unknown
            
            # The ultimate structure of a row in the coefficient matrix is thus:
            # [v_r_0, v_r_1, ..., v_r_N | v_th_0, v_th_1, ..., v_th_N | P_0, P_1, ..., P_N]
            # ←— block 0 — idx w/ k_r —→ ←—— block 1 — idx w/ k_th ——→ ← block 2 idx w/ k_p →

            # Dirichlet BCs for velocity at wall: v_r = 0, v_theta = -omega*r, P* = 0 (gauge pressure). This enforces no slip and penetration.
            if is_wall[i,j]:
                A[k_r,  k_r]  = 1.0; b[k_r]  = 0.0
                A[k_th, k_th] = 1.0; b[k_th] = -omega*r_arr[i]
                A[k_p,  k_p]  = 1.0; b[k_p]  = 0.0
                continue

            # Similar Dirichlet BCs for non-moving solid boundaries.
            if is_shaft[i,j] or is_solid[i,j] or is_blade_adj[i,j]: 
                A[k_r,  k_r]  = 1.0; b[k_r]  = 0.0
                A[k_th, k_th] = 1.0; b[k_th] = 0.0
                A[k_p,  k_p]  = 1.0; b[k_p]  = 0.0
                continue
            
            # Symmetry BCs: no penetration across symmetry plane, and no shear across symmetry plane. This enforces that the flow is symmetric across the plane.
            if is_symmetry[i,j]: # v_r = 0, dv_theta/dtheta = 0, dP/dtheta = 0
                A[k_r, k_r] = 1.0 # Dirichlet BC
                b[k_r] = 0.0
                A[k_th, k_th] = 1.0 # Neumann BC
                A[k_th, 1*N + i*N_theta + (j+1 if j==0 else j-1)] = -1.0
                b[k_th] = 0.0
                A[k_p, k_p] = 1.0 # Neumann BC
                A[k_p, 2*N + i*N_theta + (j+1 if j==0 else j-1)] = -1.0
                b[k_p] = 0.0
                continue

            # Handling of interior fluid nodes
            ri = r_arr[i]
            mu_ij = mu_full[i,j]
            dmu_dr_ij = dmu_dr[i-1, j-1]
            dmu_dth_ij = dmu_dth[i-1, j-1]

            # r-momentum balance: Coriolis + viscous stress - dP*/dr = 0

            # Coriolis: +2*rho*omega*v_theta
            A[k_r, vth(0,0)] += 2*rho*omega

            # Viscous stress terms: 1/r * d/dr(r*tau_rr) + 1/r * d/dth(tau_rth) - tau_thth/r
            # 1/r d/dr(r*tau_rr) expands to:

            # 2*mu*d2vr/dr2 central difference
            A[k_r, vr(+1,0)] += 2*mu_ij/dr**2
            A[k_r, vr( 0,0)] -= 4*mu_ij/dr**2
            A[k_r, vr(-1,0)] += 2*mu_ij/dr**2

            # 2*mu/r*dvr/dr central difference
            A[k_r, vr(+1,0)] += mu_ij/ri/dr
            A[k_r, vr(-1,0)] -= mu_ij/ri/dr

            # 2*dmu/dr*dvr/dr
            A[k_r, vr(+1,0)] += dmu_dr_ij/dr
            A[k_r, vr(-1,0)] -= dmu_dr_ij/dr

            # 1/r d/dth(tau_rth) expands to:

            # 1/r dmu/dth dvth/dr central difference
            A[k_r, vth(+1,0)] += dmu_dth_ij/(2*ri*dr)
            A[k_r, vth(-1,0)] -= dmu_dth_ij/(2*ri*dr)

            # -1/r^2 dmu/dth vth 
            A[k_r, vth(0,0)] -= dmu_dth_ij/ri**2

            # 1/r^2 dmu/th dvr/dth central difference
            A[k_r, vr(0,+1)] += dmu_dth_ij/(2*ri**2*dtheta)
            A[k_r, vr(0,-1)] -= dmu_dth_ij/(2*ri**2*dtheta)

            # -mu/r^2 d2vth/drdth central difference
            A[k_r, vth(+1,+1)] += mu_ij/(4*ri*dr*dtheta)
            A[k_r, vth(+1,-1)] -= mu_ij/(4*ri*dr*dtheta)
            A[k_r, vth(-1,+1)] -= mu_ij/(4*ri*dr*dtheta)
            A[k_r, vth(-1,-1)] += mu_ij/(4*ri*dr*dtheta)

            # -mu/r^2 dvth/dth central difference
            A[k_r, vth(0,+1)] -= mu_ij/(2*ri**2*dtheta)
            A[k_r, vth(0,-1)] += mu_ij/(2*ri**2*dtheta)

            # mu/r^2 d2vr/dth2
            A[k_r, vr(0,+1)] += mu_ij/(ri**2*dtheta**2)
            A[k_r, vr(0, 0)] -= 2*mu_ij/(ri**2*dtheta**2)
            A[k_r, vr(0,-1)] += mu_ij/(ri**2*dtheta**2)

            # Pressure gradient term - dP*/dr central difference
            A[k_r, p(+1,0)] -= 1/(2*dr)
            A[k_r, p(-1,0)] += 1/(2*dr)

            # tau_thth/r expands to:

            # -2mu/r^2 vr 
            A[k_r, vr(0,0)] -= 2*mu_ij/ri**2

            # -2mu/r^2 dvth/dth central difference
            A[k_r, vth(0,+1)] -= mu_ij/(ri**2*dtheta)
            A[k_r, vth(0,-1)] += mu_ij/(ri**2*dtheta)

            # theta-momentum balance: - Coriolis + viscous stress - dP*/dr = 0

            # Coriolis: +2*rho*omega*v_theta
            A[k_th, vr(0,0)] -= 2*rho*omega

            # Viscous stress terms: 1/r^2 d/dr(r*tau_rth) + 1/r d/dth(tau_thth)
            # 1/r^2 d/dr(r*tau_rth) expands to:

            # 2mu/r dvth/dr central difference
            A[k_th, vth(+1,0)] += 2*mu_ij/(ri*2*dr)
            A[k_th, vth(-1,0)] -= 2*mu_ij/(ri*2*dr)

            # -2mu/r^2 vth
            A[k_th, vth(0,0)] -= 2*mu_ij/ri**2

            # 2mu/r^2 dvr/dth central difference
            A[k_th, vr(0,+1)] += mu_ij/(ri**2*dtheta)
            A[k_th, vr(0,-1)] -= mu_ij/(ri**2*dtheta)

            # dmu/dr dvth/dr central difference
            A[k_th, vth(+1,0)] += dmu_dr_ij/(2*dr)
            A[k_th, vth(-1,0)] -= dmu_dr_ij/(2*dr)

            # -dmu/dr vth/r
            A[k_th, vth(0,0)] -= dmu_dr_ij/ri

            # 1/r dmu/dr dvr/dth central difference
            A[k_th, vr(0,+1)] += dmu_dr_ij/(ri*2*dtheta)
            A[k_th, vr(0,-1)] -= dmu_dr_ij/(ri*2*dtheta)

            # mu d2vth/dr2 central difference
            A[k_th, vth(+1,0)] += mu_ij/dr**2
            A[k_th, vth(0,0)] -= 2*mu_ij/dr**2
            A[k_th, vth(-1,0)] += mu_ij/dr**2

            # -mu/r dvth/dr central difference
            A[k_th, vth(+1,0)] -= mu_ij/(2*ri*dr)
            A[k_th, vth(-1,0)] += mu_ij/(2*ri*dr)

            # mu/r^2 vth
            A[k_th, vth(0,0)] += mu_ij/ri**2

            # -mu/r^2 dvr/dth central difference
            A[k_th, vr(0,+1)] -= mu_ij/(2*ri**2*dtheta)
            A[k_th, vr(0,-1)] += mu_ij/(2*ri**2*dtheta)

            # mu/r d2vr/drdth central difference
            A[k_th, vr(+1,+1)] -= mu_ij/(4*ri*dr*dtheta)
            A[k_th, vr(+1,-1)] += mu_ij/(4*ri*dr*dtheta)
            A[k_th, vr(-1,+1)] += mu_ij/(4*ri*dr*dtheta)
            A[k_th, vr(-1,-1)] -= mu_ij/(4*ri*dr*dtheta)

            # 1/r d/dth(tau_thth) expands to:

            # 2/r^2 * dmu/dth * dvth/dth
            A[k_th, vth(0,+1)] += dmu_dth_ij/(ri**2*dtheta)
            A[k_th, vth(0,-1)] -= dmu_dth_ij/(ri**2*dtheta)

            # 2/r^2 * dmu/dth * vr/r
            A[k_th, vr(0,0)] += 2*dmu_dth_ij/ri**2  # note: adds to Coriolis node

            # 2*mu/r^2 * d2vth/dth2
            A[k_th, vth(0,+1)] += 2*mu_ij/(ri**2*dtheta**2)
            A[k_th, vth(0, 0)] -= 4*mu_ij/(ri**2*dtheta**2)
            A[k_th, vth(0,-1)] += 2*mu_ij/(ri**2*dtheta**2)

            # 2*mu/r^2 * dvr/dth
            A[k_th, vr(0,+1)] += mu_ij/(ri**2*dtheta)
            A[k_th, vr(0,-1)] -= mu_ij/(ri**2*dtheta)

            # Pressure gradient term - 1/r dP*/dtheta central difference
            A[k_th, p(0,+1)] -= 1/(2*ri*dtheta)
            A[k_th, p(0,-1)] += 1/(2*ri*dtheta)

            # Continuity equation: 1/r d/dr(r*vr) + 1/r d/dth(vth) = 0
            # 1/r d/dr(r*vr) expands to:

            # vr/r
            A[k_p, vr(0,0)] += 1/ri

            # dvr/dr central difference
            A[k_p, vr(+1,0)] += 1/(2*dr)
            A[k_p, vr(-1,0)] -= 1/(2*dr)

            # and 1/r dvth/dth
            A[k_p, vth(0,+1)] += 1/(2*ri*dtheta)
            A[k_p, vth(0,-1)] -= 1/(2*ri*dtheta)

    # Set reference pressure to zero at one node to ensure a unique solution (since pressure is defined up to a constant). This ensures the coefficient matrix has a nonzero eigenvalue and is invertible.
    # Pin pressure at a known interior fluid node, e.g. middle of domain (r = 0.55 m, theta = pi/2 rad).
    k_pin = 2*N + (N_r//2)*N_theta + (N_theta//2)
    A[k_pin, :] = 0.0
    A[k_pin, k_pin] = 1.0
    b[k_pin] = 0.0

    # Solve and unflatten matrices
    x = spsolve(csr_matrix(A), b)
    v_r_new = x[0*N:1*N].reshape((N_r, N_theta))
    v_theta_new = x[1*N:2*N].reshape((N_r, N_theta))
    P_star = x[2*N:3*N].reshape((N_r, N_theta))

    # 8. Check for Picard convergence
    max_vr_change = np.max(np.abs(v_r_new - v_r))
    max_vth_change = np.max(np.abs(v_theta_new - v_theta))

    log(f"Iteration {iteration}: change_vr={max_vr_change:.3e}, change_vth={max_vth_change:.3e}")

    # Set velocity field for next Picard iteration
    v_r = v_r_new
    v_theta = v_theta_new

    if max_vr_change < tolerance and max_vth_change < tolerance:
        log(f"Picard iteration converged after {iteration} iterations. Max vr change: {max_vr_change}, Max vtheta change: {max_vth_change}")
        break
 
if iteration == max_iterations:
    log(f"Picard iteration did not converge after {max_iterations} iterations. Max vr change: {max_vr_change}, Max vtheta change: {max_vth_change}")
    sys.exit(1)

# 8. Post-processing

# Inspect shear rate and viscosity at blade-tip-adjacent fluid nodes (one cell radially
# outside the blade tip, sampled at three angular positions: tip-center, +theta corner, −theta corner).
i_tip = int(round((r_blade_tip - r_min) / dr)) + 1 # one fluid node above r=r_blade_tip
j_center_b1 = 0 # theta=0, center of blade 1's tip
j_plus_b1 = int(round(theta_blade_tip / dtheta)) + 1 # just past +theta corner of blade 1
j_minus_b1 = N_theta - j_plus_b1 # just past −theta corner (wrap)
for j_probe, label in [(j_center_b1, 'tip-center'), (j_plus_b1, '+θ corner'), (j_minus_b1, '−θ corner')]:
    gd = shear_rate[i_tip - 1, j_probe]    # shear_rate has shape (N_r-2, N_theta)
    mv = mu_full[i_tip, j_probe]
    print(f"Blade-tip probe ({label}, r={r_arr[i_tip]:.3f} m, θ={np.rad2deg(theta_arr[j_probe]):.1f}°): "
          f"γ̇ = {gd:.3e} 1/s, μ = {mv:.3e} Pa·s")

# Expand theta domain to 2pi (skip the duplicated pi boundary node so this matches the mirrored field arrays below)
theta_full = np.concatenate([theta_arr, theta_arr[1:] + np.pi])  # 0 to 2pi
N_theta_full = 2 * N_theta - 1

# Mirror velocity and viscosity fields 
v_r_full = np.concatenate([v_r, v_r[:, 1:]], axis=1)
v_theta_full = np.concatenate([v_theta, v_theta[:, 1:]], axis=1)
mu_full_360 = np.concatenate([mu_full, mu_full[:, 1:]], axis=1)

# Full 360° polar mesh matching the mirrored fields
r_full_mesh, theta_full_mesh = np.meshgrid(r_arr, theta_full, indexing='ij')

# Handle pressure calculations
P_star_full = np.concatenate([P_star, P_star[:, 1:]], axis=1)
P_full = P_star_full + 0.5 * rho * (omega * r_full_mesh)**2

# Convert to inertial reference frame
v_theta_inertial = v_theta_full + omega * r_full_mesh

# Plotting in the non-inertial reference

# Velocity magnitude field
v_mag = np.sqrt(v_r_full**2 + v_theta_full**2)
v_mag_inertial = np.sqrt(v_r_full**2 + v_theta_inertial**2)
fig, axes = plt.subplots(1, 2, subplot_kw={'projection': 'polar'})
plt.suptitle(f'Velocity magnitudes for Ω={omega/(2*np.pi)*60:.2f} RPM')
for ax, field, label in zip(axes, [v_mag, v_mag_inertial], ['|v| (rotating frame)', '|v| (inertial frame)']):
    c = ax.pcolormesh(theta_full, r_arr, field, cmap='viridis')
    ax.set_thetamin(0); ax.set_thetamax(360)
    plt.colorbar(c, ax=ax, label='|v| (m/s)')
    ax.set_title(label)

# Velocity component fields
fig, axes = plt.subplots(1, 3, subplot_kw={'projection': 'polar'})
plt.suptitle(f'Velocity fields for Ω={omega/(2*np.pi)*60:.2f} RPM')
for ax, field, label in zip(axes, [v_r_full, v_theta_full, v_theta_inertial], ['v_r (m/s)', 'v_θ (m/s, rotating frame)', 'v_θ (m/s, inertial frame)']):
    c = ax.pcolormesh(theta_full, r_arr, field, cmap='RdBu_r')
    ax.set_thetamin(0); ax.set_thetamax(360)
    plt.colorbar(c, ax=ax, label=label, orientation='horizontal')
    ax.set_title(label)

# Viscosity field
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
c = ax.pcolormesh(theta_full, r_arr, mu_full_360, cmap='plasma')
ax.set_thetamin(0); ax.set_thetamax(360)
plt.colorbar(c, label='μ (Pa·s)')
ax.set_title(f'Apparent viscosity for Ω={omega/(2*np.pi)*60:.2f} RPM')

# Flow-driven pressure field. The collocated-grid scheme admits a high-frequency
# checkerboard pressure null mode that contaminates the visualization but does
# not affect the velocity solution. Smooth it out for plotting only.
from scipy.ndimage import uniform_filter
P_full_plot = uniform_filter(P_full, size=3, mode='nearest')
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
c = ax.pcolormesh(theta_full, r_arr, P_full_plot, cmap='inferno')
ax.set_thetamin(0); ax.set_thetamax(360)
plt.colorbar(c, label='P (Pa)')
ax.set_title(f'Flow-driven pressure field for Ω={omega/(2*np.pi)*60:.2f} RPM')

# Convert polar grid to Cartesian coordinates (full 360°)
X_pol = r_full_mesh * np.cos(theta_full_mesh)
Y_pol = r_full_mesh * np.sin(theta_full_mesh)

# Cartesian velocity components for both frames
vx_rot = v_r_full * np.cos(theta_full_mesh) - v_theta_full * np.sin(theta_full_mesh)
vy_rot = v_r_full * np.sin(theta_full_mesh) + v_theta_full * np.cos(theta_full_mesh)
vx_inert = v_r_full * np.cos(theta_full_mesh) - v_theta_inertial * np.sin(theta_full_mesh)
vy_inert = v_r_full * np.sin(theta_full_mesh) + v_theta_inertial * np.cos(theta_full_mesh)

# Uniform Cartesian grid covering the full disk
x_lin = np.linspace(-r_max, r_max, 400)
y_lin = np.linspace(-r_max, r_max, 400)
X_cart, Y_cart = np.meshgrid(x_lin, y_lin)
R_cart = np.sqrt(X_cart**2 + Y_cart**2)
mask_outside = (R_cart > r_max) | (R_cart < r_min)

points = np.column_stack([X_pol.ravel(), Y_pol.ravel()])

def interp_to_cart(field):
    out = griddata(points, field.ravel(), (X_cart, Y_cart), method='linear')
    out[mask_outside] = np.nan
    return out

vx_rot_c, vy_rot_c = interp_to_cart(vx_rot), interp_to_cart(vy_rot)
vx_inert_c, vy_inert_c = interp_to_cart(vx_inert), interp_to_cart(vy_inert)

phi = np.linspace(0, 2*np.pi, 200)
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
plt.suptitle(f'Flow streamlines for Ω={omega/(2*np.pi)*60:.2f} RPM')
for ax, vxc, vyc, title in zip(
    axes,
    [vx_rot_c, vx_inert_c],
    [vy_rot_c, vy_inert_c],
    ['Streamlines (rotating frame)', 'Streamlines (inertial frame)'],
):
    speed = np.sqrt(vxc**2 + vyc**2)
    ax.streamplot(x_lin, y_lin, vxc, vyc, density=2, color=speed, cmap='viridis', linewidth=0.8)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')

    # Blade outlines (both sides)
    for theta_edge in (0.0, np.pi):
        theta_blade = np.linspace(theta_edge - theta_blade_tip, theta_edge + theta_blade_tip, 100)
        r_blade = np.full_like(theta_blade, r_blade_tip)
        x_blade = np.concatenate([[0], r_blade*np.cos(theta_blade), [0]])
        y_blade = np.concatenate([[0], r_blade*np.sin(theta_blade), [0]])
        ax.fill(x_blade, y_blade, alpha=0.3, color='gray')

    # Domain bounding circles
    ax.plot(r_max*np.cos(phi), r_max*np.sin(phi), color='black', lw=1)
    ax.plot(r_min*np.cos(phi), r_min*np.sin(phi), color='black', lw=1)
    ax.set_xlim(-r_max*1.05, r_max*1.05)
    ax.set_ylim(-r_max*1.05, r_max*1.05)

Re_field = np.full_like(mu_full, np.nan)
fluid = ~is_solid
Re_field[fluid] = rho * np.sqrt(v_r[fluid]**2 + v_theta[fluid]**2) * r[fluid] / mu_full[fluid]
print(f"Max Re = {np.nanmax(Re_field):.3f}")

# Reynolds number field, mirrored to full 360°
Re_field_full = np.concatenate([Re_field, Re_field[:, 1:]], axis=1)
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
c = ax.pcolormesh(theta_full, r_arr, Re_field_full, cmap='viridis')
ax.set_thetamin(0); ax.set_thetamax(360)
plt.colorbar(c, label='Re')
ax.set_title(f'Local Reynolds number for Ω={omega/(2*np.pi)*60:.2f} RPM, (max = {np.nanmax(Re_field_full):.2f})')

log("Done!")
plt.show()