import numpy as np
import autograd
import autograd.numpy as anp
from autograd.scipy.signal import convolve as ag_convolve
from autograd.tracer import getval

import tidy3d as td
import tidy3d.web as web

td.config.logging.level = "ERROR"

# =====================================================================
# 1. CONSTANTS & PARAMETERS
# =====================================================================
C_C0 = td.C_0
C_FS = 1e-15
C_THZ = 1e12

WVL_0 = 2.9013670   
FREQ_0 = C_C0 / WVL_0

FWIDTH = 7 * C_THZ
FREQ_RANGE = (FREQ_0 - FWIDTH/2, FREQ_0 + FWIDTH/2)        
WVL_RANGE = (C_C0 / FREQ_RANGE[1], C_C0 / FREQ_RANGE[0])

N_TANTALA = 2.036
EPS_TANTALA = N_TANTALA**2

THICKNESS = 0.8       
WG_LENGTH = 2 * WVL_0     
WG_WIDTH = 1.7          

REGION_LENGTH = 6.5     
REGION_WIDTH = 6.5      
REGION_GRID_SIZE = 0.4  
nx = int(REGION_LENGTH / REGION_GRID_SIZE)
ny = int(REGION_WIDTH / REGION_GRID_SIZE)

FRAME_WIDTH = 0.5       # Width of the robust supporting frame
TAPER_LENGTH = 2.0      # Length of the taper from frame to waveguide
ROUNDING_RADIUS = 0.6   # Radius for rounded corners

BEAM_ANGLE_DEG = 0
BEAM_WAIST = 2.62
BEAM_WAIST_DISTANCE = 2

# =====================================================================
# 2. FRAME & WAVEGUIDE GEOMETRY GENERATOR
# =====================================================================
def get_rounded_frame_polygon():
    """
    Generates a single solid polygon for the Frame + Tapers + Waveguide,
    applying corner rounding (fillets) to all sharp edges.
    """
    x_left = -REGION_LENGTH/2 - FRAME_WIDTH
    x_right = REGION_LENGTH/2 + FRAME_WIDTH
    y_top = REGION_WIDTH/2 + FRAME_WIDTH
    y_bot = -REGION_WIDTH/2 - FRAME_WIDTH
    x_wg_end = REGION_LENGTH/2 + FRAME_WIDTH + TAPER_LENGTH + WG_LENGTH
    
    # Vertices tracing the outer boundary counter-clockwise
    vertices = [
        [x_wg_end, WG_WIDTH/2],                 # 0: WG top (Do not round - hits PML)
        [x_right + TAPER_LENGTH, WG_WIDTH/2],   # 1: Taper-WG joint top 
        [x_right, y_top],                       # 2: Frame Top-Right 
        [x_left, y_top],                        # 3: Frame Top-Left 
        [x_left, y_bot],                        # 4: Frame Bottom-Left 
        [x_right, y_bot],                       # 5: Frame Bottom-Right 
        [x_right + TAPER_LENGTH, -WG_WIDTH/2],  # 6: Taper-WG joint bot 
        [x_wg_end, -WG_WIDTH/2]                 # 7: WG bot (Do not round)
    ]
    
    rounded_poly = []
    n = len(vertices)
    for i in range(n):
        if i in [0, 7]: # Keep waveguide ends perfectly straight for the PML
            rounded_poly.append(vertices[i])
            continue
            
        p_prev = np.array(vertices[i-1])
        p_curr = np.array(vertices[i])
        p_next = np.array(vertices[(i+1)%n])
        
        v1 = p_prev - p_curr
        v2 = p_next - p_curr
        l1, l2 = np.linalg.norm(v1), np.linalg.norm(v2)
        v1, v2 = v1 / l1, v2 / l2
        
        angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
        d = ROUNDING_RADIUS / np.tan(angle / 2)
        d = min(d, l1 / 2.1, l2 / 2.1) # Prevent over-filleting short edges
        eff_radius = d * np.tan(angle / 2)
        
        p_start = p_curr + v1 * d
        p_end = p_curr + v2 * d
        
        # Calculate normal vector pointing inwards
        n1 = np.array([-v1[1], v1[0]])
        cross_prod = v1[0]*v2[1] - v1[1]*v2[0]
        if cross_prod < 0: n1 = -n1
            
        center = p_start + n1 * eff_radius
        angle_start = np.arctan2(p_start[1] - center[1], p_start[0] - center[0])
        angle_end = np.arctan2(p_end[1] - center[1], p_end[0] - center[0])
        
        if cross_prod < 0:
            if angle_start < angle_end: angle_start += 2*np.pi
        else:
            if angle_end < angle_start: angle_end += 2*np.pi
            
        arc_angles = np.linspace(angle_start, angle_end, 15)
        for a in arc_angles:
            rounded_poly.append([center[0] + eff_radius * np.cos(a), center[1] + eff_radius * np.sin(a)])
            
    return rounded_poly

# =====================================================================
# 3. FILTERING & BINARIZATION 
# =====================================================================
MIN_FEATURE_SIZE = 0.1 
FILTER_RADIUS = max(MIN_FEATURE_SIZE, REGION_GRID_SIZE * 1.5)

def get_conic_filter(radius, dx, dy):
    rx, ry = int(np.ceil(radius / dx)), int(np.ceil(radius / dy))
    x, y = np.arange(-rx, rx + 1) * dx, np.arange(-ry, ry + 1) * dy
    X, Y = np.meshgrid(x, y)
    kernel = np.maximum(0, radius - np.sqrt(X**2 + Y**2))
    return kernel / np.sum(kernel)

CONIC_KERNEL = get_conic_filter(FILTER_RADIUS, REGION_GRID_SIZE, REGION_GRID_SIZE)

def apply_topology_constraints(params, beta):
    filtered = ag_convolve(params, CONIC_KERNEL, mode='same')
    density = 0.5 * (anp.tanh(beta * filtered) + 1.0)
    return density

# =====================================================================
# 4. OMNIDIRECTIONAL CONNECTIVITY
# =====================================================================
# Mask identifying the 4 edges of the inverse design region
BOUNDARY_MASK = np.zeros((nx, ny), dtype=bool)
BOUNDARY_MASK[0, :] = True
BOUNDARY_MASK[-1, :] = True
BOUNDARY_MASK[:, 0] = True
BOUNDARY_MASK[:, -1] = True

def connectivity_penalty(density_matrix, iterations=100):
    kernel = anp.array([[0.0, 0.25, 0.0], 
                        [0.25, 0.0, 0.25], 
                        [0.0, 0.25, 0.0]])
    
    heat = anp.zeros_like(density_matrix)
    
    for _ in range(iterations):
        # Inject heat from the surrounding structural frame
        heat = anp.where(BOUNDARY_MASK, 1.0, heat)
        h_diff = ag_convolve(heat, kernel, mode='same')
        heat = h_diff * density_matrix
        
    heat = anp.where(BOUNDARY_MASK, 1.0, heat)
    penalty = anp.sum(density_matrix * (1.0 - heat)**2)
    return penalty

# =====================================================================
# 5. SIMULATION DEFINITION
# =====================================================================
def make_simulation(params, beta):
    density = apply_topology_constraints(params, beta)
    permittivity = 1.0 + (EPS_TANTALA - 1.0) * density
    
    # 1. Base Frame Geometry
    frame_vertices = get_rounded_frame_polygon()
    frame_slab = td.PolySlab(vertices=frame_vertices, axis=2, slab_bounds=(-THICKNESS/2, THICKNESS/2))
    frame_structure = td.Structure(geometry=frame_slab, medium=td.Medium(permittivity=EPS_TANTALA))
    
    # 2. Inverse Design Medium
    xs = anp.linspace(-REGION_LENGTH/2, REGION_LENGTH/2, nx)
    ys = anp.linspace(-REGION_WIDTH/2, REGION_WIDTH/2, ny)
    zs = anp.array([-THICKNESS/2, THICKNESS/2])
    
    eps_3d = permittivity.reshape((nx, ny, 1))
    eps_3d = anp.repeat(eps_3d, 2, axis=2)
    
    eps_dataset = td.SpatialDataArray(data=eps_3d, coords={"x": xs, "y": ys, "z": zs})
    design_medium = td.CustomMedium(eps_dataset=eps_dataset)
    
    design_box = td.Box(center=(0, 0, 0), size=(REGION_LENGTH, REGION_WIDTH, THICKNESS))
    design_structure = td.Structure(geometry=design_box, medium=design_medium)
    
    # 3. Simulation Setup
    gaussian_source = td.GaussianBeam(
        center=(0, 0, BEAM_WAIST_DISTANCE),
        size=(3*BEAM_WAIST, 3*BEAM_WAIST, 0),
        angle_theta=BEAM_ANGLE_DEG*np.pi/180,
        pol_angle=np.pi/2,   
        direction="-", 
        waist_radius=BEAM_WAIST,
        waist_distance=-BEAM_WAIST_DISTANCE,
        source_time=td.GaussianPulse(freq0=FREQ_0, fwidth=FWIDTH),
    )
    
    mode_monitor = td.ModeMonitor(
        center=(REGION_LENGTH/2 + FRAME_WIDTH + TAPER_LENGTH + WG_LENGTH/2, 0, 0),
        size=(0, WG_WIDTH + 3*WVL_0, THICKNESS + 3*WVL_0),
        freqs=[FREQ_0],
        mode_spec=td.ModeSpec(num_modes=1, target_neff=N_TANTALA),
        name="mode_coupling"
    )
    
    # Calculate simulation domain bounds to ensure waveguide doesn't get chopped off
    x_min_sim = -REGION_LENGTH/2 - FRAME_WIDTH - 1.0
    x_max_sim = REGION_LENGTH/2 + FRAME_WIDTH + TAPER_LENGTH + WG_LENGTH + 0.5
    sim_size_x = x_max_sim - x_min_sim
    sim_center_x = (x_max_sim + x_min_sim) / 2.0
    
    y_min_sim = -REGION_WIDTH/2 - FRAME_WIDTH - 1.0
    y_max_sim = REGION_WIDTH/2 + FRAME_WIDTH + 1.0
    sim_size_y = y_max_sim - y_min_sim

    sim = td.Simulation(
        center=(sim_center_x, 0, 0),
        size=(sim_size_x, sim_size_y, 4.0),
        grid_spec=td.GridSpec.auto(min_steps_per_wvl=15),
        # Boolean Priority: design_structure comes LAST, so its air pixels hollow out the solid frame!
        structures=[frame_structure, design_structure], 
        sources=[gaussian_source],
        monitors=[mode_monitor],
        run_time=2e-12,
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML())
    )
    
    return sim, density

def objective_fn(params, beta):
    sim, density = make_simulation(params, beta)    
    sim_data = web.run(sim, task_name="grating_coupler_opt", verbose=False)
    
    # Extract coupled mode amplitude in the forward (+) direction
    amps = sim_data["mode_coupling"].amps.sel(direction="+").values
    coupling_efficiency = anp.sum(anp.abs(amps)**2)
    
    island_penalty = connectivity_penalty(density, iterations=nx)
    penalty_weight = 0.5 
    
    loss = -coupling_efficiency + (penalty_weight * island_penalty)
    
    print(f"  --> Beta: {beta:.2f} | Eff: {float(getval(coupling_efficiency)):.4f} | Pen: {float(getval(island_penalty)):.4f}")
    
    return loss

# =====================================================================
# 6. MAIN OPTIMIZATION LOOP
# =====================================================================
if __name__ == "__main__":
    np.random.seed(114514)
    params = np.random.uniform(-0.1, 0.1, size=(nx, ny))

    learning_rate = 0.1
    b1, b2, eps = 0.9, 0.999, 1e-8
    m, v = np.zeros_like(params), np.zeros_like(params)

    val_and_grad_fn = autograd.value_and_grad(objective_fn, argnum=0)

    num_epochs = 50
    beta_schedule = np.linspace(1.0, 30.0, num_epochs)

    print("Starting Native Autograd Inverse Design Optimization...")
    for epoch in range(num_epochs):
        beta = beta_schedule[epoch]
        
        loss, gradients = val_and_grad_fn(params, beta)
        gradients = np.array(getval(gradients))
        
        t = epoch + 1
        m = b1 * m + (1.0 - b1) * gradients
        v = b2 * v + (1.0 - b2) * (gradients ** 2)
        m_hat = m / (1.0 - b1 ** t)
        v_hat = v / (1.0 - b2 ** t)
        
        params = params - learning_rate * m_hat / (np.sqrt(v_hat) + eps)
        print(f"Epoch {epoch+1:03d} | Total Loss: {float(getval(loss)):.4f}\n")