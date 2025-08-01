import torch, torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ────────────────────────────── Hyper-params ──────────────────────────────
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
DIM_IN         = 2          # x-dimension of the PDE domain
HIDDEN         = 50         # neurons per hidden layer
DEPTH          = 4          # hidden layers
N_DOMAIN_PTS   = 3_000      # collocation points inside Ω
N_BOUNDARY_PTS = 1_000      # points on ∂Ω
EPOCHS         = 5_000
LR             = 1e-3
# ───────────────────────────────── Domain Ω  ───────────────────────────────
# Here: unit square; change to your own sampler.
def sample_domain(n):
    return torch.rand(n, DIM_IN, device=DEVICE)          # (n, 2) in (0,1)^2
def sample_boundary(n):
    # Four edges of the unit square
    edges = []
    m = n // 4
    edges.append(torch.stack([torch.rand(m), torch.zeros(m)], dim=1))
    edges.append(torch.stack([torch.rand(m), torch.ones(m)],  dim=1))
    edges.append(torch.stack([torch.zeros(m), torch.rand(m)], dim=1))
    edges.append(torch.stack([torch.ones(m),  torch.rand(m)], dim=1))
    return torch.cat(edges, 0).to(DEVICE)
# ───────────────────────────── Neural network uθ ───────────────────────────
def make_mlp(d_in, hidden, depth, d_out=1):
    layers = [nn.Linear(d_in, hidden), nn.Tanh()]
    for _ in range(depth-1):
        layers += [nn.Linear(hidden, hidden), nn.Tanh()]
    layers += [nn.Linear(hidden, d_out)]
    return nn.Sequential(*layers)

net = make_mlp(DIM_IN, HIDDEN, DEPTH).to(DEVICE)

# ──────────────────────────────── PDE helpers ──────────────────────────────
def gradient(y, x):
    '''∇y wrt x (y: (N,1), x: (N,D)) → (N,D)'''
    return torch.autograd.grad(
        y, x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

def laplacian(y, x):
    '''∇²y for scalar y'''
    grads = gradient(y, x)          # (N,D)
    lap = 0.
    for i in range(DIM_IN):
        lap += gradient(grads[:, i:i+1], x)[:, i:i+1]
    return lap

# ───────────────────────── Example: 2-D Poisson  ∇²u = f(x,y) ─────────────
def f_source(xy):
    x, y = xy[:, 0:1], xy[:, 1:2]
    return torch.exp(-x) * (x - 2 + y**3 + 6*y)

def pde_residual(x, u, grads, grads2):
    """Return r(x)=0 if u satisfies PDE.  For Poisson: r = ∇²u - f."""
    return grads2 - f_source(x)

def boundary_condition(x):
    """Dirichlet value g(x) on ∂Ω for the toy Poisson problem."""
    x0, x1 = x[:, 0:1], x[:, 1:2]
    return torch.exp(-x0) * (x0 + x1**3)

# ───────────────────────────── Loss function ───────────────────────────────
def loss_pinn():
    # Interior points Ω
    x_int = sample_domain(N_DOMAIN_PTS).requires_grad_(True)
    u_int = net(x_int)
    lap_u = laplacian(u_int, x_int)
    resid  = pde_residual(x_int, u_int, None, lap_u)   # (N,1)

    # Boundary points ∂Ω
    x_bnd = sample_boundary(N_BOUNDARY_PTS)
    u_bnd = net(x_bnd)
    bc    = u_bnd - boundary_condition(x_bnd)

    return (resid**2).mean() + (bc**2).mean()

# ───────────────────────────── Training loop ───────────────────────────────
opt = torch.optim.Adam(net.parameters(), lr=LR)

for epoch in range(1, EPOCHS+1):
    opt.zero_grad()
    l = loss_pinn()
    l.backward()
    opt.step()

    if epoch % 500 == 0:
        print(f'Epoch {epoch:5d} | Loss {l.item():.3e}')

# ─────────────────────── Evaluate & plot on grid ───────────────────────────
with torch.no_grad():
    g = 100
    xv, yv = torch.meshgrid(torch.linspace(0,1,g), torch.linspace(0,1,g), indexing='ij')
    XY     = torch.stack([xv.flatten(), yv.flatten()], dim=1).to(DEVICE)
    U_pred = net(XY).cpu().view(g, g).numpy()
    U_true = np.exp(-xv) * (xv + yv**3)  # analytic for demo

fig = plt.figure(figsize=(10,4))
ax1 = fig.add_subplot(1,2,1, projection='3d')
ax1.plot_surface(xv, yv, U_pred, cmap='viridis'); ax1.set_title('PINN')
ax2 = fig.add_subplot(1,2,2, projection='3d')
ax2.plot_surface(xv, yv, U_true, cmap='plasma');  ax2.set_title('Analytic')
plt.show()



# Physics‑Informed Neural‑Network (PINN) loss snippets you can paste directly in a Python file

# ─────────────────────────────────────────────────────────────────────────────

# Helper: automatic derivatives ------------------------------------------------

# def diff(u, x, order=1):
# """Nth‑order derivative of scalar tensor u with respect to x (autograd)."""
# for _ in range(order):
# u = torch.autograd.grad(
# u, x,
# grad_outputs=torch.ones_like(u),
# create_graph=True,
# retain_graph=True,
# only_inputs=True,
# )[0]
# return u

# -----------------------------------------------------------------------------

# 1) Poisson  ─────────────────────────────────────────────────────────────────--

# equation :  laplace(u) = f

# loss     :  MSE( laplace(u) - f(x) )

# u      = net(x)      # x ≡ (x,y)
# lap_u  = diff(u, x, 2).sum(-1, keepdim=True)  # ∂²u/∂x² + ∂²u/∂y²
# pde_poisson = (lap_u - f(x))**2

# -----------------------------------------------------------------------------

# 2) Heat  ─────────────────────────────────────────────────────────────────────-

# equation :  u_t = α * laplace(u)

# xt     = torch.cat([x, t], 1)   # x=(x,y), t=(t,)
# u      = net(xt)
# utt    = diff(u, t, 1)          # ∂u/∂t
# lap_u  = diff(u, x, 2).sum(-1, keepdim=True)
# alpha  = 0.01
# pde_heat = (utt - alpha*lap_u)**2

# -----------------------------------------------------------------------------

# 3) Wave  ─────────────────────────────────────────────────────────────────────-

# equation :  u_tt = c² * laplace(u)

# utt    = diff(u, t, 2)          # ∂²u/∂t²
# pde_wave = (utt - c**2 * lap_u)**2

# -----------------------------------------------------------------------------

# 4) 2‑D Incompressible Navier‑Stokes  ----------------------------------------

# Unknowns : u(x,y,t), v(x,y,t) velocity; p(x,y,t) pressure

# Params   : ρ (density), ν (kinematic viscosity)

# rho, nu = 1.0, 0.01
# xyt     = torch.cat([x, t], 1)   # domain coords (x,y,t)

# u = net_u(xyt)   # horizontal velocity
# v = net_v(xyt)   # vertical   velocity
# p = net_p(xyt)   # pressure

# First‑order partials ---------------------------------------------------------

# u_t = diff(u, t, 1);             v_t = diff(v, t, 1)
# u_x = diff(u, x[:,0:1], 1);       u_y = diff(u, x[:,1:2], 1)
# v_x = diff(v, x[:,0:1], 1);       v_y = diff(v, x[:,1:2], 1)
# p_x = diff(p, x[:,0:1], 1);       p_y = diff(p, x[:,1:2], 1)

# Second‑order for viscosity term ---------------------------------------------

# lap_u = diff(u, x, 2).sum(-1, keepdim=True)
# lap_v = diff(v, x, 2).sum(-1, keepdim=True)

# PDE Residuals ---------------------------------------------------------------

# res_u = u_t + uu_x + vu_y + (1.0/rho)p_x - nulap_u
# res_v = v_t + uv_x + vv_y + (1.0/rho)p_y - nulap_v
# res_c = u_x + v_y                       # continuity (mass conservation)

# pde_navier = res_u2 + res_v2 + res_c**2

# -----------------------------------------------------------------------------

# Total physics loss (mean‑squared):

# physics_loss = (pde_poisson.mean()        # or pde_heat.mean(), etc.
# + pde_navier.mean())      # add other terms as needed

