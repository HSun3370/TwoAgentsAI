import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt

##############################################################################
# 1) Static AI Economy Model: Definitions & Solver with Revised FOCs
##############################################################################

def ai_equations(vars_, A, K_g, beta, gamma, iota, psi, K_a, alpha, Z, D, theta):
    """
    Computes the residuals of the 4-equation system describing the static equilibrium.
    
    Unknowns (in order):
      L_g  : Human labor in the general goods sector.
      L_a  : Labor in the AI sector.
      w    : Wage.
      p    : Price of AI computation.
      
    Intermediate variables:
      - AI Computation: X = Z * K_a^alpha * L_a^(1-alpha)
      - AI Agent: L_AI = D^theta * X^(1-theta)
      - Composite labor (used in goods production): 
            composite = [iota * L_g^gamma + (1-iota) * L_AI^gamma]^(1/gamma)
      - General goods output: Y = A * K_g^beta * composite^(1-beta)
      
    Equilibrium conditions:
      (1) FOC for L_g:
          Y*(1-beta)*iota*(L_g/composite)^gamma / L_g - w = 0.
      (2) FOC for X:
          Y*(1-beta)*(1-iota)*(1-theta)*(L_AI/composite)^gamma / X - p = 0.
      (3) FOC for L_a in the AI sector:
          (1-alpha)*p*X/L_a - w = 0.
      (4) Labor market clearing:
          1 - L_g - L_a - psi*(Y/w) = 0.
           ( C = Y : In static equilibrium, there is no investment so all production goes to consumptiom.)
    """
    
    L_g, L_a, w, p = vars_
    
    # AI sector production:
    X = Z * (K_a ** alpha) * (L_a ** (1 - alpha))
    
    # Artificial labor (the AI agent):
    L_AI = (D ** theta) * (X ** (1 - theta))
    
    # Composite labor used in general goods production:
    composite = (iota * (L_g ** gamma) + (1 - iota) * (L_AI ** gamma)) ** (1 / gamma)
    
    # Output in the general goods sector:
    Y = A * (K_g ** beta) * (composite ** (1 - beta))
    
    # (1) FOC for human labor L_g:
    eq1 = Y * (1 - beta) * iota * (L_g / composite) ** gamma / L_g - w
    
    # (2) FOC for computation X:
    eq2 = Y * (1 - beta) * (1 - iota) * (1 - theta) * (L_AI / composite) ** gamma / X - p
    
    # (3) FOC for AI sector labor L_a:
    eq3 = (1 - alpha) * p * X / L_a - w
    
    # (4) Labor market clearing (including leisure):
    eq4 = 1 - L_g - L_a - psi * (Y / w)
    
    return np.array([eq1, eq2, eq3, eq4])


def solve_ai_model(
    alpha=0.5,    # Exponent on capital in the AI sector.
    theta=0.5,    # Exponent for data in forming AI-labor.
    gamma=0.1,    # Substitution parameter (elasticity) between human and AI labor.
    iota=0.5,     # Weight on human labor in the composite labor input.
    beta=0.3,     # Exponent on capital in the general goods sector.
    psi=1.0,      # Leisure parameter (N = psi*(Y/w)).
    A=0.1,        # TFP in the general goods sector.
    K_g=4.0,      # Capital in the general goods sector.
    Z=1.0,        # Productivity in the AI sector.
    K_a=1.0,      # Capital in the AI sector.
    D=1.0,        # Data available for AI.
    guess=(0.3, 0.3, 1.0, 1.0),
    method='hybr'
):
    """
    Solve the static model for the unknowns: (L_g, L_a, w, p). Also computes derived variables:
      - X: AI sector production.
      - L_AI: Artificial labor.
      - composite: Composite labor input.
      - Y: Output in the general goods sector.
      - N: Leisure.
      - Profits: Pi_g and Pi_a.
      - Shadow prices: V_Kg, V_Ka, and V_D.
    """
    sol = root(
        ai_equations,
        guess,
        args=(A, K_g, beta, gamma, iota, psi, K_a, alpha, Z, D, theta),
        method=method
    )
    if not sol.success:
        raise RuntimeError(f"Solve failed: {sol.message}")
    
    L_g, L_a, w, p = sol.x
    
    # Recompute intermediate variables:
    X = Z * (K_a ** alpha) * (L_a ** (1 - alpha))
    L_AI = (D ** theta) * (X ** (1 - theta))
    composite = (iota * (L_g ** gamma) + (1 - iota) * (L_AI ** gamma)) ** (1 / gamma)
    Y = A * (K_g ** beta) * (composite ** (1 - beta))
    N = 1 - L_g - L_a  # (Should equal psi*(Y/w) in equilibrium.)
    
    # Profits:
    Pi_g = Y - w * L_g - p * X
    Pi_a = p * X - w * L_a
    
    # Shadow prices:
    V_Kg = beta * Y / K_g
    V_Ka = alpha * p * X / K_a
    V_D  = (1 - beta) * (1 - iota) * theta * (L_AI / composite) ** gamma * Y / D
    
    return {
        'L_g': L_g,
        'L_a': L_a,
        'w': w,
        'p': p,
        'X': X,
        'L_AI': L_AI,
        'composite': composite,
        'Y': Y,
        'N': N,
        'Pi_g': Pi_g,
        'Pi_a': Pi_a,
        'V_Kg': V_Kg,
        'V_Ka': V_Ka,
        'V_D': V_D
    }

##############################################################################
# 2) Parameter Sweep: Storing All Variables and Derived Ratios/Prices
##############################################################################

def run_sweep(param_name, values, fixed_params):
    """
    For each value in 'values' (for the parameter 'param_name'),
    solve the model and store the following variables:
      - L_g, L_a, w, p, X, Y, Pi_g, Pi_a, L_AI, composite, N
      - Shadow prices: V_Kg, V_Ka, V_D
      - Additionally, store the ratio: L_AI/composite.
    """
    keys = ['L_g', 'L_a', 'w', 'p', 'X', 'Y', 'Pi_g', 'Pi_a', 
            'L_AI', 'composite', 'N', 'V_Kg', 'V_Ka', 'V_D', 'ratio_LAI']
    storage = {k: [] for k in keys}
    
    for val in values:
        params = dict(fixed_params)
        params[param_name] = val
        
        try:
            eq = solve_ai_model(**params)
        except Exception:
            for k in keys:
                storage[k].append(np.nan)
            continue
        
        # Ratio of AI labor over composite labor:
        ratio_LAI = eq['L_AI'] / eq['composite'] if eq['composite'] != 0 else np.nan
        
        storage['L_g'].append(eq['L_g'])
        storage['L_a'].append(eq['L_a'])
        storage['w'].append(eq['w'])
        storage['p'].append(eq['p'])
        storage['X'].append(eq['X'])
        storage['Y'].append(eq['Y'])
        storage['Pi_g'].append(eq['Pi_g'])
        storage['Pi_a'].append(eq['Pi_a'])
        storage['L_AI'].append(eq['L_AI'])
        storage['composite'].append(eq['composite'])
        storage['N'].append(eq['N'])
        storage['V_Kg'].append(eq['V_Kg'])
        storage['V_Ka'].append(eq['V_Ka'])
        storage['V_D'].append(eq['V_D'])
        storage['ratio_LAI'].append(ratio_LAI)
        
    # Convert lists to numpy arrays.
    for k in storage:
        storage[k] = np.array(storage[k], dtype=float)
    
    return storage

##############################################################################
# 3) Custom Plotting: Separate Figures for Selected Variables
##############################################################################

def custom_plots(x_vals, results, x_label, title_prefix):
    """
    Creates separate figures for:
      1. Both labors L_g and L_a.
      2. Wage w.
      3. Price of AI computation p.
      4. Profits of the two sectors: Pi_g and Pi_a.
      5. General goods production Y.
      6. Ratio of AI labor over composite: L_AI/composite.
      7. Shadow prices: V_Kg, V_Ka, V_D.
    """
    # Figure 1: Both labors in one figure.
    plt.figure(figsize=(5, 4))
    plt.plot(x_vals, results['L_g'], 'bo-', label=r'$L_g$')
    plt.plot(x_vals, results['L_a'], 'ro-', label=r'$L_a$')
    plt.xlabel(x_label)
    plt.ylabel("Labor")
    plt.title(f"{title_prefix}")
    plt.legend()
    
    # Figure 2: Wage
    plt.figure(figsize=(5, 4))
    plt.plot(x_vals, results['w'], 'go-')
    plt.xlabel(x_label)
    plt.ylabel("Wage (w)")
    plt.title(f"{title_prefix}: Wage")
    
    # Figure 3: Price of AI computation
    plt.figure(figsize=(5, 4))
    plt.plot(x_vals, results['p'], 'mo-')
    plt.xlabel(x_label)
    plt.ylabel("Price (p)")
    plt.title(f"{title_prefix}: Price of AI Computation")
    
    # Figure 4: Profits of the two sectors
    plt.figure(figsize=(5, 4))
    plt.plot(x_vals, results['Pi_g'], 'co-', label=r'$\Pi_g$ (General Goods Sector)')
    plt.plot(x_vals, results['Pi_a'], 'yo-', label=r'$\Pi_a$ (AI Sector)')
    plt.xlabel(x_label)
    plt.ylabel("Profit")
    plt.title(f"{title_prefix}")
    plt.legend()
    
    # Figure 5: General goods production Y
    plt.figure(figsize=(5, 4))
    plt.plot(x_vals, results['Y'], 'ko-')
    plt.xlabel(x_label)
    plt.ylabel("Output (Y)")
    plt.title(f"{title_prefix}: General Goods Production")
    
    # Figure 6: Ratio of AI labor over composite labor: L_AI / composite
    plt.figure(figsize=(5, 4))
    plt.plot(x_vals, results['ratio_LAI'], 'bo-')
    plt.xlabel(x_label)
    plt.ylabel(r"AI Agent/Composite Labor")
    plt.title(f"{title_prefix}")
    
    # Figure 7: Shadow prices: V_Kg, V_Ka, and V_D
    plt.figure(figsize=(5, 4))
    plt.plot(x_vals, results['V_Kg'], 'r^-', label=r'$V_{K_g}$')
    plt.plot(x_vals, results['V_Ka'], 's-', color='green', label=r'$V_{K_a}$')
    plt.plot(x_vals, results['V_D'], 'd-', color='blue', label=r'$V_D$')
    plt.xlabel(x_label)
    plt.ylabel("Shadow Price")
    plt.title(f"{title_prefix}")
    plt.legend()
    
    plt.show()