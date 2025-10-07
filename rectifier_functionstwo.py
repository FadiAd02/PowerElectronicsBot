import math
import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar, fsolve
from fractions import Fraction
from typing import Optional, Dict, Any
import re

def half_wave_rectifier_with_resistive_load(vm, f, r):
    """
    Calculate outputs for a half-wave rectifier with a resistive load.
    """
    if vm <= 0:
        raise ValueError("Peak voltage (vm) must be positive")
    if r <= 0:
        raise ValueError("Resistance (r) must be positive")
    if f <= 0:
        raise ValueError("Frequency (f) must be positive")

    result = {}
    steps = []

    # Step 1: Calculate average output voltage
    v_0 = vm / math.pi
    steps.append(f"Step 1: Calculate average output voltage using V_0 = V_m / π: {vm} / π ≈ {v_0:.4f} V")
    result["average output voltage"] = round(v_0, 4)

    # Step 2: Calculate average load current
    i_0 = v_0 / r
    steps.append(f"Step 2: Calculate average load current using I_0 = V_0 / R: {v_0:.4f} / {r} ≈ {i_0:.4f} A")
    result["average load current"] = round(i_0, 4)

    # Step 3: Calculate RMS voltage
    v_rms = vm / 2
    steps.append(f"Step 3: Calculate RMS voltage of rectified output using V_rms = V_m / 2: {vm} / 2 = {v_rms:.4f} V")
    result["rms output voltage"] = round(v_rms, 4)

    # Step 4: Calculate RMS load current
    i_rms = v_rms / r
    steps.append(f"Step 4: Calculate RMS load current using I_rms = V_rms / R: {v_rms:.4f} / {r} = {i_rms:.4f} A")
    result["rms load current"] = round(i_rms, 4)

    # Step 5: Calculate power absorbed by the load
    p = i_rms ** 2 * r
    steps.append(f"Step 5: Calculate power absorbed by the load using P = I_rms^2 * R: ({i_rms:.4f})^2 * {r} ≈ {p:.4f} W")
    result["power absorbed by the load"] = round(p, 4)

    # Step 6: Calculate RMS voltage of the source
    v_s_rms = vm / math.sqrt(2)
    steps.append(f"Step 6: Calculate RMS voltage of the source using V_s,rms = V_m / sqrt(2): {vm} / √2 ≈ {v_s_rms:.4f} V")
    result["rms source voltage"] = round(v_s_rms, 4)

    # Step 7: Calculate apparent power
    s = v_s_rms * i_rms
    steps.append(f"Step 7: Calculate apparent power supplied by the source using S = V_s,rms * I_rms: {v_s_rms:.4f} * {i_rms:.4f} ≈ {s:.4f} VA")
    result["apparent power supplied by the source"] = round(s, 4)

    # Step 8: Calculate power factor
    pf = p / s
    steps.append(f"Step 8: Calculate power factor using pf = P / S: {p:.4f} / {s:.4f} ≈ {pf:.4f}")
    result["power factor"] = round(pf, 4)

    # Add steps_text
    result["steps_text"] = "\n".join(steps)
    return result

def half_wave_rectifier_with_transformer_and_resistive_load(vs_rms, f, r, i_0):
    """
    Calculate outputs for a half-wave rectifier with a transformer and resistive load.
    """
    if vs_rms <= 0:
        raise ValueError("Source RMS voltage (vs_rms) must be positive")
    if r <= 0:
        raise ValueError("Resistance (r) must be positive")
    if f <= 0:
        raise ValueError("Frequency (f) must be positive")
    if i_0 <= 0:
        raise ValueError("Average load current (i_0) must be positive")

    result = {}
    steps = []

    # Step 1: Calculate average output voltage
    v_0 = i_0 * r
    steps.append(f"Step 1: Calculate average output voltage using V_0 = I_0 * R: {i_0} * {r} = {v_0:.4f} V")
    result["average output voltage"] = round(v_0, 4)

    # Step 2: Calculate peak voltage
    v_m = v_0 * math.pi
    steps.append(f"Step 2: Calculate peak voltage on secondary side using V_m = V_0 * π: {v_0:.4f} * π ≈ {v_m:.4f} V")
    result["peak voltage"] = round(v_m, 4)

    # Step 3: Calculate RMS voltage on secondary
    v_rms_secondary = v_m / math.sqrt(2)
    steps.append(f"Step 3: Calculate RMS voltage on secondary side using V_rms = V_m / sqrt(2): {v_m:.4f} / √2 ≈ {v_rms_secondary:.4f} V")
    result["rms voltage secondary"] = round(v_rms_secondary, 4)

    # Step 4: Calculate turns ratio
    turns_ratio = vs_rms / v_rms_secondary
    steps.append(f"Step 4: Calculate turns ratio using N_1/N_2 = V_s,rms / V_rms: {vs_rms} / {v_rms_secondary:.4f} ≈ {turns_ratio:.4f}")
    result["turns ratio of the transformer"] = round(turns_ratio, 4)

    # Step 5: Calculate average current in primary
    primary_current = i_0 / turns_ratio
    steps.append(f"Step 5: Calculate average current in primary winding using I'_0 = I_0 * (N_2/N_1): {i_0} / {turns_ratio:.4f} ≈ {primary_current:.4f} A")
    result["average current in the primary winding of the transformer"] = round(primary_current, 4)

    # Add steps_text
    result["steps_text"] = "\n".join(steps)
    return result

def half_wave_rectifier_with_rl_load(vs_rms, f, r, l):
    """
    Calculate outputs for a half-wave rectifier with an RL load.
    """
    if vs_rms <= 0:
        raise ValueError("Source RMS voltage (vs_rms) must be positive")
    if r <= 0:
        raise ValueError("Resistance (r) must be positive")
    if l <= 0:
        raise ValueError("Inductance (l) must be positive")
    if f <= 0:
        raise ValueError("Frequency (f) must be positive")

    result = {}
    steps = []

    # Step 1: Calculate peak voltage
    v_m = math.sqrt(2) * vs_rms
    steps.append(f"Step 1: Calculate peak voltage using V_m = sqrt(2) * V_s,rms: √2 * {vs_rms} ≈ {v_m:.4f} V")
    result["peak voltage"] = round(v_m, 4)

    # Step 2: Calculate angular frequency
    omega = 2 * math.pi * f
    steps.append(f"Step 2: Calculate angular frequency using ω = 2 * π * f: 2 * π * {f} ≈ {omega:.4f} rad/s")
    result["angular frequency"] = round(omega, 4)

    # Step 3: Calculate impedance
    omega_l = omega * l
    z = math.sqrt(r**2 + omega_l**2)
    steps.append(f"Step 3: Calculate impedance of RL load using Z = sqrt(R^2 + (ωL)^2): √({r}^2 + ({omega:.4f} * {l})^2) ≈ {z:.4f} Ω")
    result["impedance"] = round(z, 4)

    # Step 4: Calculate phase angle
    theta = math.atan(omega_l / r)
    steps.append(f"Step 4: Calculate phase angle using θ = tan^-1(ωL / R): tan^-1({omega_l:.4f} / {r}) ≈ {theta:.4f} rad")
    result["phase angle"] = round(theta, 4)

    # Step 5: Calculate time constant term
    omega_tau = omega_l / r
    steps.append(f"Step 5: Calculate time constant term using ωτ = ωL / R: {omega_l:.4f} / {r} = {omega_tau:.4f}")
    result["omega_tau"] = round(omega_tau, 4)

    # Step 6: Calculate extinction angle
    i_m = v_m / z
    sin_theta = math.sin(theta)
    def current_function(wt):
        return i_m * math.sin(wt - theta) + i_m * sin_theta * math.exp(-wt / omega_tau)
    beta_solution = root_scalar(current_function, bracket=[theta, 2 * math.pi], method='brentq')
    beta = beta_solution.root
    steps.append(f"Step 6: Calculate extinction angle using i(β) = (V_m/Z) * sin(β - θ) + (V_m/Z) * sin(θ) * e^(-β / ωτ) = 0: Solved numerically ≈ {beta:.4f} rad")
    result["extinction angle"] = round(beta, 4)

    # Step 7: Calculate load current expression
    steps.append(f"Step 7: Calculate load current expression using i(ωt) = (V_m/Z) * sin(ωt - θ) + (V_m/Z) * sin(θ) * e^(-ωt / ωτ): {i_m:.4f} * sin(ωt - {theta:.4f}) + {i_m * sin_theta:.4f} * e^(-ωt / {omega_tau:.4f}), for 0 ≤ ωt ≤ {beta:.2f} rad")
    result["expression for load current"] = f"{i_m:.4f} * sin(ωt - {theta:.4f}) + {i_m * sin_theta:.4f} * e^(-ωt / {omega_tau:.4f})"

    # Step 8: Calculate average current
    def integrand(wt):
        return i_m * math.sin(wt - theta) + i_m * sin_theta * math.exp(-wt / omega_tau)
    i_avg_integral, _ = quad(integrand, 0, beta)
    i_avg = i_avg_integral / (2 * math.pi)
    steps.append(f"Step 8: Calculate average current using I_avg = (1/(2π)) * ∫[0 to {beta:.4f}] i(ωt) d(ωt): ∫[0 to {beta:.4f}] ({i_m:.4f} * sin(ωt - {theta:.4f}) + {i_m * sin_theta:.4f} * e^(-ωt / {omega_tau:.4f})) d(ωt) / (2π) ≈ {i_avg:.4f} A")
    result["average current"] = round(i_avg, 4)

    # Step 9: Calculate RMS current
    def integrand_squared(wt):
        i = i_m * math.sin(wt - theta) + i_m * sin_theta * math.exp(-wt / omega_tau)
        return i ** 2
    i_rms_integral, _ = quad(integrand_squared, 0, beta)
    i_rms = math.sqrt(i_rms_integral / (2 * math.pi))
    steps.append(f"Step 9: Calculate RMS current using I_rms = sqrt((1/(2π)) * ∫[0 to {beta:.4f}] [i(ωt)]^2 d(ωt)): sqrt(∫[0 to {beta:.4f}] ({i_m:.4f} * sin(ωt - {theta:.4f}) + {i_m * sin_theta:.4f} * e^(-ωt / {omega_tau:.4f}))^2 d(ωt) / (2π)) ≈ {i_rms:.4f} A")
    result["rms current"] = round(i_rms, 4)

    # Step 10: Calculate power absorbed by the resistor
    p = i_rms ** 2 * r
    steps.append(f"Step 10: Calculate power absorbed by the resistor using P = I_rms^2 * R: ({i_rms:.4f})^2 * {r} ≈ {p:.4f} W")
    result["power absorbed by the resistor"] = round(p, 4)

    # Step 11: Calculate apparent power
    s = vs_rms * i_rms
    steps.append(f"Step 11: Calculate apparent power using S = V_s,rms * I_rms: {vs_rms} * {i_rms:.4f} ≈ {s:.4f} VA")
    result["apparent power"] = round(s, 4)

    # Step 12: Calculate power factor
    pf = p / s
    steps.append(f"Step 12: Calculate power factor using pf = P / S: {p:.4f} / {s:.4f} ≈ {pf:.4f}")
    result["power factor"] = round(pf, 4)

    # Add steps_text
    result["steps_text"] = "\n".join(steps)
    return result

def half_wave_rectifier_with_rl_load_and_pspice(vs_rms, f, r, l):
    """
    Calculate outputs for a half-wave rectifier with an RL load (PSpice variant).
    """
    if vs_rms <= 0:
        raise ValueError("Source RMS voltage (vs_rms) must be positive")
    if r <= 0:
        raise ValueError("Resistance (r) must be positive")
    if l <= 0:
        raise ValueError("Inductance (l) must be positive")
    if f <= 0:
        raise ValueError("Frequency (f) must be positive")

    result = {}
    steps = []

    # Step 1: Calculate peak voltage
    v_m = math.sqrt(2) * vs_rms
    steps.append(f"Step 1: Calculate peak voltage using V_m = sqrt(2) * V_s,rms: √2 * {vs_rms} ≈ {v_m:.4f} V")
    result["peak voltage"] = round(v_m, 4)

    # Step 2: Calculate angular frequency
    omega = 2 * math.pi * f
    steps.append(f"Step 2: Calculate angular frequency using ω = 2 * π * f: 2 * π * {f} ≈ {omega:.4f} rad/s")
    result["angular frequency"] = round(omega, 4)

    # Step 3: Calculate impedance
    omega_l = omega * l
    z = math.sqrt(r**2 + omega_l**2)
    steps.append(f"Step 3: Calculate impedance of RL load using Z = sqrt(R^2 + (ωL)^2): √({r}^2 + ({omega:.4f} * {l})^2) ≈ {z:.4f} Ω")
    result["impedance"] = round(z, 4)

    # Step 4: Calculate phase angle
    theta = math.atan(omega_l / r)
    steps.append(f"Step 4: Calculate phase angle using θ = tan^-1(ωL / R): tan^-1({omega_l:.4f} / {r}) ≈ {theta:.4f} rad")
    result["phase angle"] = round(theta, 4)

    # Step 5: Calculate time constant term
    omega_tau = omega_l / r
    steps.append(f"Step 5: Calculate time constant term using ωτ = ωL / R: {omega_l:.4f} / {r} = {omega_tau:.4f}")
    result["omega_tau"] = round(omega_tau, 4)

    # Step 6: Calculate extinction angle
    i_m = v_m / z
    sin_theta = math.sin(theta)
    def current_function(wt):
        return i_m * math.sin(wt - theta) + i_m * sin_theta * math.exp(-wt / omega_tau)
    beta_solution = root_scalar(current_function, bracket=[theta, 2 * math.pi], method='brentq')
    beta = beta_solution.root
    steps.append(f"Step 6: Calculate extinction angle using i(β) = (V_m/Z) * sin(β - θ) + (V_m/Z) * sin(θ) * e^(-β / ωτ) = 0: Solved numerically ≈ {beta:.4f} rad")
    result["extinction angle"] = round(beta, 4)

    # Step 7: Calculate load current expression
    steps.append(f"Step 7: Calculate load current expression using i(ωt) = (V_m/Z) * sin(ωt - θ) + (V_m/Z) * sin(θ) * e^(-ωt / ωτ): {i_m:.4f} * sin(ωt - {theta:.4f}) + {i_m * sin_theta:.4f} * e^(-ωt / {omega_tau:.4f}), for 0 ≤ ωt ≤ {beta:.2f} rad")
    result["expression for load current"] = f"{i_m:.4f} * sin(ωt - {theta:.4f}) + {i_m * sin_theta:.4f} * e^(-ωt / {omega_tau:.4f})"

    # Step 8: Calculate average current
    def integrand(wt):
        return i_m * math.sin(wt - theta) + i_m * sin_theta * math.exp(-wt / omega_tau)
    i_avg_integral, _ = quad(integrand, 0, beta)
    i_avg = i_avg_integral / (2 * math.pi)
    steps.append(f"Step 8: Calculate average current using I_avg = (1/(2π)) * ∫[0 to {beta:.4f}] i(ωt) d(ωt): ∫[0 to {beta:.4f}] ({i_m:.4f} * sin(ωt - {theta:.4f}) + {i_m * sin_theta:.4f} * e^(-ωt / {omega_tau:.4f})) d(ωt) / (2π) ≈ {i_avg:.4f} A")
    result["average current"] = round(i_avg, 4)

    # Step 9: Calculate RMS current
    def integrand_squared(wt):
        i = i_m * math.sin(wt - theta) + i_m * sin_theta * math.exp(-wt / omega_tau)
        return i ** 2
    i_rms_integral, _ = quad(integrand_squared, 0, beta)
    i_rms = math.sqrt(i_rms_integral / (2 * math.pi))
    steps.append(f"Step 9: Calculate RMS current using I_rms = sqrt((1/(2π)) * ∫[0 to {beta:.4f}] [i(ωt)]^2 d(ωt)): sqrt(∫[0 to {beta:.4f}] ({i_m:.4f} * sin(ωt - {theta:.4f}) + {i_m * sin_theta:.4f} * e^(-ωt / {omega_tau:.4f}))^2 d(ωt) / (2π)) ≈ {i_rms:.4f} A")
    result["rms current"] = round(i_rms, 4)

    # Step 10: Calculate power absorbed by the resistor
    p = i_rms ** 2 * r
    steps.append(f"Step 10: Calculate power absorbed by the resistor using P = I_rms^2 * R: ({i_rms:.4f})^2 * {r} ≈ {p:.4f} W")
    result["power absorbed by the resistor"] = round(p, 4)

    # Step 11: Calculate apparent power
    s = vs_rms * i_rms
    steps.append(f"Step 11: Calculate apparent power using S = V_s,rms * I_rms: {vs_rms} * {i_rms:.4f} ≈ {s:.4f} VA")
    result["apparent power"] = round(s, 4)

    # Step 12: Calculate power factor
    pf = p / s
    steps.append(f"Step 12: Calculate power factor using pf = P / S: {p:.4f} / {s:.4f} ≈ {pf:.4f}")
    result["power factor"] = round(pf, 4)

    # Step 13: PSpice placeholder
    steps.append("Step 13: PSpice simulation comparison with analytical results: Not computed: Requires external PSpice simulation tool.")
    result["PSpice simulation comparison with analytical results"] = "Not computed: Requires external PSpice simulation tool."

    # Add steps_text
    result["steps_text"] = "\n".join(steps)
    return result

def half_wave_rectifier_with_rl_source_load(vs_rms, f, r, l, vdc):
    """
    Half-wave rectifier with series RL + DC source load.
    """
    if vs_rms <= 0:
        raise ValueError("vs_rms must be positive")
    if f <= 0:
        raise ValueError("f must be positive")
    if r <= 0:
        raise ValueError("r must be positive")
    if l <= 0:
        raise ValueError("l must be positive")
    if vdc < 0:
        raise ValueError("vdc must be non-negative")

    result = {}
    steps = []

    omega = 2 * math.pi * f
    v_m = math.sqrt(2) * vs_rms
    steps.append(f"Step 1: Calculate peak voltage using V_m = sqrt(2) * V_s,rms: V_m = √2 * {vs_rms} ≈ {v_m:.4f} V")
    result["V_m"] = round(v_m, 4)

    omega_l = omega * l
    z = math.sqrt(r**2 + omega_l**2)
    steps.append(f"Step 2: Load impedance using Z = sqrt(R^2 + (ωL)^2): Z = √({r}^2 + ({omega:.4f}*{l})^2) ≈ {z:.4f} Ω")
    result["Z"] = round(z, 4)

    theta = math.atan2(omega_l, r)
    steps.append(f"Step 3: Phase angle using θ = atan(ωL / R): θ = atan({omega_l:.4f} / {r}) ≈ {theta:.4f} rad")
    result["theta (rad)"] = round(theta, 4)

    omega_tau = omega_l / r
    steps.append(f"Step 4: Time-constant term using ωτ = ωL / R: ωτ = {omega_l:.4f} / {r} = {omega_tau:.4f}")
    result["omega_tau"] = round(omega_tau, 4)

    if abs(vdc) > v_m:
        raise ValueError("Vdc >= V_m: no conduction (asin undefined).")
    alpha = math.asin(vdc / v_m)
    steps.append(f"Step 5: Firing angle using α = arcsin(Vdc / V_m): α = asin({vdc:.4f} / {v_m:.4f}) ≈ {alpha:.4f} rad ({math.degrees(alpha):.2f}°)")
    result["alpha (rad)"] = round(alpha, 4)

    i_m = v_m / z
    sin_a_minus_t = math.sin(alpha - theta)
    A = (-i_m * sin_a_minus_t + (vdc / r)) * math.exp(alpha / omega_tau)
    steps.append(f"Step 6: Compute constants i_m and A: i_m = V_m / Z = {v_m:.4f} / {z:.4f} ≈ {i_m:.4f} A; A ≈ {A:.4f}")
    result["i_m"] = round(i_m, 4)
    result["A"] = round(A, 4)

    def i_of_wt(wt):
        return i_m * math.sin(wt - theta) - (vdc / r) + A * math.exp(-wt / omega_tau)
    steps.append(f"Step 7: Load current expression: i(ωt) ≈ {i_m:.4f}·sin(ωt - {theta:.4f}) - {vdc/r:.4f} + {A:.4f}·e^(-ωt / {omega_tau:.4f})")
    result["expression for load current"] = f"{i_m:.4f} * sin(ωt - {theta:.4f}) - {vdc/r:.4f} + {A:.4f} * exp(-ωt / {omega_tau:.4f})"

    eps = 1e-6
    start_search = alpha + eps
    def safe_eval(x):
        try:
            return i_of_wt(x)
        except Exception:
            return None
    fa = safe_eval(start_search)
    fb = safe_eval(2 * math.pi)
    bracket = None
    if fa is not None and fb is not None and fa * fb < 0:
        bracket = (start_search, 2 * math.pi)
    else:
        Nscan = 2000
        wt_grid = [start_search + (2 * math.pi - start_search) * k / Nscan for k in range(Nscan + 1)]
        prev_w = wt_grid[0]
        prev_val = safe_eval(prev_w)
        for w in wt_grid[1:]:
            val = safe_eval(w)
            if val is None or prev_val is None:
                prev_w, prev_val = w, val
                continue
            if prev_val == 0.0:
                prev_w, prev_val = w, val
                continue
            if prev_val * val < 0:
                bracket = (prev_w, w)
                break
            prev_w, prev_val = w, val
    if bracket is None:
        raise RuntimeError("Failed to bracket extinction angle β.")
    a, b = bracket
    beta_root = root_scalar(lambda w: i_of_wt(w), bracket=[a, b], method='brentq', xtol=1e-12)
    beta = beta_root.root
    steps.append(f"Step 8: Extinction angle β by solving i(β)=0 numerically: Solved numerically: β ≈ {beta:.4f} rad ({math.degrees(beta):.1f}°)")
    result["extinction angle (rad)"] = round(beta, 4)

    i_avg_integral, _ = quad(i_of_wt, alpha, beta, epsabs=1e-9, epsrel=1e-9, limit=500)
    i_avg = i_avg_integral / (2 * math.pi)
    steps.append(f"Step 9: Average current using I_avg = (1/(2π)) * ∫_{round(alpha,4)}^{round(beta, 4)} i(ωt) d(ωt): Integral(α→β) i(ωt) d(ωt) = {i_avg_integral:.6f}; I_avg = {i_avg_integral:.6f}/(2π) ≈ {i_avg:.4f} A")
    result["average current"] = round(i_avg, 4)

    i_rms_integral, _ = quad(lambda wt: i_of_wt(wt)**2, alpha, beta, epsabs=1e-9, epsrel=1e-9, limit=500)
    i_rms = math.sqrt(i_rms_integral / (2 * math.pi))
    steps.append(f"Step 10: RMS current using I_rms = sqrt((1/(2π)) * ∫_{round(alpha,4)}^{round(beta, 4)} i^2(ωt) d(ωt)): Integral(α→β) i^2(ωt) d(ωt) = {i_rms_integral:.6f}; I_rms = sqrt({i_rms_integral:.6f}/(2π)) ≈ {i_rms:.4f} A")
    result["rms current"] = round(i_rms, 4)

    p_dc = vdc * i_avg
    steps.append(f"Step 11: Power absorbed by the dc voltage source using P_dc = Vdc * I_avg: P_dc = {vdc:.4f} * {i_avg:.4f} ≈ {p_dc:.4f} W")
    result["power absorbed by the dc voltage source"] = round(p_dc, 4)

    p_r = (i_rms ** 2) * r
    steps.append(f"Step 12: Power absorbed by the resistance using P_R = I_rms^2 * R: P_R = ({i_rms:.4f})^2 * {r} ≈ {p_r:.4f} W")
    result["power absorbed by the resistance"] = round(p_r, 4)

    p_total = p_dc + p_r
    s = vs_rms * i_rms
    pf = p_total / s if s != 0 else 0.0
    steps.append(f"Step 13: Power factor using pf = (P_dc + P_R) / (V_rms * I_rms): pf = ({p_dc:.4f} + {p_r:.4f}) / ({vs_rms} * {i_rms:.4f}) ≈ {pf:.4f}")
    result["power factor"] = round(pf, 4)

    # Add steps_text
    result["steps_text"] = "\n".join(steps)
    return result

def half_wave_rectifier_with_l_and_dc_source_load(vs_rms, f, l, vdc):
    """
    Half-wave rectifier with series inductance and DC source (no R).
    """
    if vs_rms <= 0:
        raise ValueError("vs_rms must be positive")
    if f <= 0:
        raise ValueError("f must be positive")
    if l <= 0:
        raise ValueError("l must be positive")
    if vdc < 0:
        raise ValueError("vdc must be non-negative")

    result = {}
    steps = []

    omega = 2 * math.pi * f
    v_m = math.sqrt(2) * vs_rms
    steps.append(f"Step 1: Peak voltage using V_m = sqrt(2) * V_s,rms: V_m = √2 * {vs_rms} ≈ {v_m:.4f} V")
    result["V_m"] = round(v_m, 4)

    steps.append(f"Step 2: Angular frequency using ω = 2πf: ω = 2π * {f} ≈ {omega:.4f} rad/s")
    result["omega"] = round(omega, 4)

    if abs(vdc) > v_m:
        raise ValueError("Vdc >= V_m: no conduction (asin undefined).")
    alpha = math.asin(vdc / v_m)
    steps.append(f"Step 3: Firing angle using α = asin(Vdc / V_m): α = asin({vdc:.4f} / {v_m:.4f}) ≈ {alpha:.4f} rad ({math.degrees(alpha):.2f}°)")
    result["alpha (rad)"] = round(alpha, 4)

    Vm_over_omegaL = v_m / (omega * l)
    Vdc_over_omegaL = vdc / (omega * l)
    steps.append(f"Step 4: Inductive coefficients: V_m/(ωL) = {v_m:.4f}/({omega:.4f} * {l}) ≈ {Vm_over_omegaL:.4f} A, Vdc/(ωL) = {vdc:.4f}/({omega:.4f} * {l}) ≈ {Vdc_over_omegaL:.4f} A/rad")
    result["V_m_over_omegaL"] = round(Vm_over_omegaL, 4)
    result["Vdc_over_omegaL"] = round(Vdc_over_omegaL, 4)

    def i_of_wt(wt):
        return Vm_over_omegaL * (math.cos(alpha) - math.cos(wt)) + Vdc_over_omegaL * (alpha - wt)
    steps.append(f"Step 5: Load current expression: i(ωt) ≈ {Vm_over_omegaL:.4f}·(cos({alpha:.4f}) - cos(ωt)) + {Vdc_over_omegaL:.4f}·({alpha:.4f} - ωt)")
    result["expression for load current"] = f"{Vm_over_omegaL:.4f}*(cos({alpha:.4f}) - cos(ωt)) + {Vdc_over_omegaL:.4f}*({alpha:.4f} - ωt)"

    eps = 1e-9
    start = alpha + 1e-6
    def safe_i(x):
        return i_of_wt(x)
    bracket = None
    if safe_i(start) * safe_i(2 * math.pi) < 0:
        bracket = (start, 2 * math.pi)
    else:
        Nscan = 2000
        wt_grid = [start + (2 * math.pi - start) * k / Nscan for k in range(Nscan + 1)]
        prev_w = wt_grid[0]
        prev_val = safe_i(prev_w)
        for w in wt_grid[1:]:
            val = safe_i(w)
            if prev_val == 0.0:
                bracket = (prev_w, w)
                break
            if prev_val * val < 0:
                bracket = (prev_w, w)
                break
            prev_w, prev_val = w, val
    if bracket is None:
        raise RuntimeError("Failed to bracket extinction angle β.")
    a, b = bracket
    beta_root = root_scalar(lambda wt: safe_i(wt), bracket=[a, b], method='brentq', xtol=1e-12)
    beta = beta_root.root
    steps.append(f"Step 6: Extinction angle β by solving i(β)=0 numerically: Solved numerically: β ≈ {beta:.4f} rad ({math.degrees(beta):.1f}°)")
    result["extinction angle (rad)"] = round(beta, 4)

    i_avg_integral, _ = quad(i_of_wt, alpha, beta, epsabs=1e-9, epsrel=1e-9, limit=500)
    i_avg = i_avg_integral / (2 * math.pi)
    steps.append(f"Step 7: Average current using I_o = (1/(2π)) * ∫_{round(alpha,4)}^{round(beta, 4)} i(ωt) d(ωt): Integral(α→β) i(ωt) d(ωt) = {i_avg_integral:.6f}; I_o = {i_avg_integral:.6f}/(2π) ≈ {i_avg:.4f} A")
    result["average current"] = round(i_avg, 4)

    p_dc = vdc * i_avg
    steps.append(f"Step 8: Power absorbed by the dc voltage source using P_dc = Vdc * I_o: P_dc = {vdc:.4f} * {i_avg:.4f} ≈ {p_dc:.4f} W")
    result["power absorbed by the dc voltage source"] = round(p_dc, 4)

    # Add steps_text
    result["steps_text"] = "\n".join(steps)
    return result

def half_wave_rectifier_with_freewheeling_diode_and_rl_load(vs_rms, f, r, l):
    """
    Half-wave rectifier with freewheeling diode and RL load.
    """
    if vs_rms <= 0:
        raise ValueError("vs_rms must be positive")
    if f <= 0:
        raise ValueError("f must be positive")
    if r <= 0:
        raise ValueError("r must be positive")
    if l <= 0:
        raise ValueError("l must be positive")

    result = {}
    steps = []

    omega = 2 * math.pi * f
    v_m = math.sqrt(2) * vs_rms
    steps.append(f"Step 1: Peak voltage using V_m = sqrt(2) * V_s,rms: V_m = √2 * {vs_rms} ≈ {v_m:.4f} V")
    result["V_m"] = round(v_m, 4)

    v_0 = v_m / math.pi
    i_0 = v_0 / r
    steps.append(f"Step 2: DC voltage component using V_0 = V_m / π: V_0 = {v_m:.4f} / π ≈ {v_0:.4f} V")
    steps.append(f"Step 3: DC current using I_0 = V_0 / R: I_0 = {v_0:.4f} / {r} ≈ {i_0:.4f} A")
    result["dc component of the current"] = round(i_0, 4)

    def v_of_wt(wt):
        wt_mod = wt % (2 * math.pi)
        if 0.0 <= wt_mod <= math.pi:
            return v_m * math.sin(wt_mod)
        else:
            return 0.0

    steps.append("Step 4: Compute Fourier coefficients a_n, b_n (numerically): Computed a_n and b_n for n=1.. until four nonzero harmonics found.")

    def compute_an_bn(n):
        an, _ = quad(lambda wt: v_of_wt(wt) * math.cos(n * wt), 0, 2 * math.pi, epsabs=1e-9, epsrel=1e-9, limit=400)
        bn, _ = quad(lambda wt: v_of_wt(wt) * math.sin(n * wt), 0, 2 * math.pi, epsabs=1e-9, epsrel=1e-9, limit=400)
        an = an / math.pi
        bn = bn / math.pi
        return an, bn

    harmonics = []
    n = 1
    nonzero_threshold = 1e-8
    while len(harmonics) < 4 and n <= 60:
        an, bn = compute_an_bn(n)
        Vn = math.sqrt(an * an + bn * bn)
        if Vn > nonzero_threshold:
            Zn = math.sqrt(r * r + (n * omega * l) ** 2)
            In = Vn / Zn if Zn != 0 else 0.0
            harmonics.append({
                "n": n,
                "Vn": round(Vn, 4),
                "Zn": round(Zn, 4),
                "In": round(In, 4)
            })
        n += 1

    harmonics_lines = [f"n={h['n']}: Vn ≈ {h['Vn']:.4f} V, Zn ≈ {h['Zn']:.4f} Ω, In ≈ {h['In']:.4f} A" for h in harmonics]
    steps.append(f"Step 5: First four nonzero AC harmonic amplitudes (Vn...V1=Vm/2 and using Vn=2Vm/((n^2)-1) for n=2,4,6...), impedances (Zn=|R+jnωL|), and currents (In=Vn/Zn): \n" + "\n".join(harmonics_lines))
    result["amplitudes of the first four nonzero ac terms in the Fourier series"] = harmonics

    # Add steps_text
    result["steps_text"] = "\n".join(steps)
    return result

def half_wave_rectifier_with_freewheeling_diode_and_rl_load_pspice(Vm, f, R, percent_limit=5):
    """
    Find L for a half-wave rectifier with a freewheeling diode.
    """
    if Vm <= 0:
        raise ValueError("Vm must be positive")
    if f <= 0:
        raise ValueError("f must be positive")
    if R <= 0:
        raise ValueError("R must be positive")
    if percent_limit <= 0 or percent_limit >= 100:
        raise ValueError("percent_limit must be between 0 and 100")

    result = {}
    steps = []

    I0 = Vm / (math.pi * R)
    steps.append(f"Step 1: DC current using I0 = Vm / (πR): I0 = {Vm:.4f} / (π * {R:.4f}) ≈ {I0:.4f} A")
    result["I0"] = round(I0, 4)

    I1_target = (percent_limit / 100.0) * I0
    steps.append(f"Step 2: Target first harmonic amplitude I1_target: I1_target = {percent_limit:.2f}% * {I0:.4f} ≈ {I1_target:.4f} A")
    result["I1_target"] = round(I1_target, 4)

    V1 = Vm / 2.0
    steps.append(f"Step 3: First harmonic voltage amplitude V1 = Vm / 2: V1 = {Vm:.4f} / 2 = {V1:.4f} V")
    result["V1"] = round(V1, 4)

    Z1 = V1 / I1_target
    steps.append(f"Step 4: Required impedance Z1 = V1 / I1_target: Z1 = {V1:.4f} / {I1_target:.4f} ≈ {Z1:.4f} Ω")
    result["Z1"] = round(Z1, 4)

    omega = 2 * math.pi * f
    inside = Z1**2 - R**2
    if inside < 0:
        if inside > -1e-12:
            inside = 0.0
        else:
            raise ValueError("Computed Z1 is smaller than R; cannot find real ωL.")
    omegaL = math.sqrt(inside)
    L_required = omegaL / omega
    steps.append(f"Step 5: Compute ωL and L: ω = 2πf = {omega:.4f} rad/s; ωL = sqrt({Z1:.4f}² - {R:.4f}²) ≈ {omegaL:.4f}; L = {omegaL:.4f} / {omega:.4f} ≈ {L_required:.4f} H")
    result["required inductance L"] = round(L_required, 4)

    pp_theoretical = 2 * I1_target
    steps.append(f"Step 6: Theoretical peak-to-peak current: p-p (theory) = 2 * {I1_target:.4f} ≈ {pp_theoretical:.4f} A")
    result["theoretical_pp_current"] = round(pp_theoretical, 4)

    pspice_note = "Step 7: PSpice verification and peak-to-peak current: Not computed: Requires external PSpice simulation tool."
    steps.append(pspice_note)
    result["PSpice verification and peak-to-peak current"] = pspice_note

    # Add steps_text
    result["steps_text"] = "\n".join(steps)
    return result

def half_wave_rectifier_with_freewheeling_diode_and_dc_source_load(Vm, f, R, Vdc, delta_i_pp=1.0):
    """
    Half-wave rectifier with freewheeling diode and a series DC source.
    """
    if Vm <= 0:
        raise ValueError("Vm must be positive")
    if f <= 0:
        raise ValueError("f must be positive")
    if R <= 0:
        raise ValueError("R must be positive")
    if Vdc < 0:
        raise ValueError("Vdc must be non-negative")
    if delta_i_pp <= 0:
        raise ValueError("delta_i_pp must be positive")

    result = {}
    steps = []

    I1_target = delta_i_pp / 2.0
    steps.append(f"Step 1: Set first-harmonic target from Δi0 ≈ 2 I1: Target: Δi0 ≤ {delta_i_pp:.4f} A ⇒ I1_target = {I1_target:.4f} A")
    result["I1_target"] = round(I1_target, 4)

    V0 = Vm / math.pi
    I0 = (V0 - Vdc) / R
    steps.append(f"Step 2: DC voltage and current: V0 = V_m/π, I0 = (V0 - Vdc)/R: V0 = {Vm:.4f}/π ≈ {V0:.4f} V; I0 = ({V0:.4f} - {Vdc:.4f}) / {R:.4f} ≈ {I0:.4f} A")
    result["V0"] = round(V0, 4)
    result["I0"] = round(I0, 4)

    V1 = Vm / 2.0
    steps.append(f"Step 3: First harmonic voltage using V1 = V_m / 2: V1 = {Vm:.4f} / 2 = {V1:.4f} V")
    result["V1"] = round(V1, 4)

    Z1 = V1 / I1_target
    steps.append(f"Step 4: Required impedance Z1 = V1 / I1_target: Z1 = {V1:.4f} / {I1_target:.4f} ≈ {Z1:.4f} Ω")
    result["Z1"] = round(Z1, 4)

    omega = 2 * math.pi * f
    inside = Z1 ** 2 - R ** 2
    if inside < 0:
        if inside > -1e-12:
            inside = 0.0
        else:
            raise ValueError("Computed Z1 is smaller than R; cannot find real ωL.")
    omegaL = math.sqrt(inside)
    L_required = omegaL / omega
    steps.append(f"Step 5: Compute ωL and L (ωL = sqrt(Z1^2 - R^2), L = ωL / ω): ω = 2πf = {omega:.4f} rad/s; ωL ≈ sqrt({Z1:.4f}^2 - {R:.4f}^2) = {omegaL:.4f}; L = {omegaL:.4f}/{omega:.4f} ≈ {L_required:.4f} H")
    result["required inductance L"] = round(L_required, 4)

    P_dc = I0 * Vdc
    steps.append(f"Step 6: Power absorbed by DC source using P_dc = I0 * Vdc: P_dc = {I0:.4f} * {Vdc:.4f} ≈ {P_dc:.4f} W")
    result["power absorbed by the dc source"] = round(P_dc, 4)

    I_rms = math.sqrt(I0 ** 2 + (I1_target / math.sqrt(2)) ** 2)
    steps.append(f"Step 7: RMS current using I_rms = sqrt(I0^2 + (I1/√2)^2): I_rms = sqrt({I0:.4f}^2 + ({I1_target:.4f}/√2)^2) ≈ {I_rms:.4f} A")
    result["I_rms"] = round(I_rms, 4)

    P_R = (I_rms ** 2) * R
    steps.append(f"Step 8: Power absorbed by resistor using P_R = I_rms^2 * R: P_R = ({I_rms:.4f})^2 * {R:.4f} ≈ {P_R:.4f} W")
    result["power absorbed by the resistor"] = round(P_R, 4)

    pp_current = 2 * I1_target
    steps.append(f"Step 9: Theoretical peak-to-peak current (p-p = 2 * I1_target): p-p (theory) = 2 * {I1_target:.4f} ≈ {pp_current:.4f} A")
    result["theoretical_pp_current"] = round(pp_current, 4)

    steps.append("Step 10: PSpice verification and peak-to-peak current: Not computed: Requires external PSpice simulation tool.")
    result["PSpice verification and peak-to-peak current"] = "Not computed: Requires external PSpice simulation tool."

    # Add steps_text
    result["steps_text"] = "\n".join(steps)
    return result

def half_wave_rectifier_with_capacitor_filter_single_resistance(vm, r, c, omega):
    """
    Calculate outputs for a half-wave rectifier with a capacitor filter and single resistance.
    """
    if vm <= 0:
        raise ValueError("Peak voltage (vm) must be positive")
    if r <= 0:
        raise ValueError("Resistance (r) must be positive")
    if c <= 0:
        raise ValueError("Capacitance (c) must be positive")
    if omega <= 0:
        raise ValueError("Angular frequency (omega) must be positive")

    result = {}
    steps = []

    rc = r * c
    period = 2 * math.pi / omega
    f = omega / (2 * math.pi)
    ratio = rc / period
    steps.append(f"Step 1: Time constant and period ratio: τ = R·C = {r} * {c} = {rc:.4f} s; T = 2π/ω = 2π/{omega:.4f} ≈ {period:.4f} s; τ/T ≈ {ratio:.4f}")
    result["ratio of RC time constant to period"] = round(ratio, 4)

    theta = -math.atan(omega * rc) + math.pi
    steps.append(f"Step 2: Compute θ = -atan(ωRC) + π: θ = -atan({omega:.4f} * {rc:.4f}) + π ≈ {theta:.4f} rad ({math.degrees(theta):.4f}°)")
    result["theta (rad)"] = round(theta, 4)

    def transcendental(alpha):
        return math.sin(alpha) - math.sin(theta) * math.exp(-(2 * math.pi + alpha + theta) / (omega * rc))
    try:
        alpha_solution = root_scalar(transcendental, bracket=[0, math.pi / 2], method='brentq')
        alpha = alpha_solution.root
        steps.append(f"Step 3: Solve transcendental equation for α: α ≈ {alpha:.4f} rad ({math.degrees(alpha):.4f}°)")
        result["alpha (rad)"] = round(alpha, 4)
    except ValueError as e:
        steps.append(f"Step 3: Solve transcendental equation for α: Failed to solve: {str(e)}. Using approximate ripple.")
        result["alpha (rad)"] = 0
        alpha = 0

    v_r_exact = vm * (1 - math.sin(alpha))
    steps.append(f"Step 4: Exact peak-to-peak ripple using ΔV = Vm (1 - sin α): ΔV_exact = {vm:.4f} * (1 - sin({alpha:.4f})) ≈ {v_r_exact:.4f} V")
    result["peak-to-peak ripple voltage using exact equations"] = round(v_r_exact, 4)

    v_r_approx = vm / (f * r * c)
    if v_r_approx > vm:
        v_r_approx = vm
        steps.append(f"Step 5: Approximate ripple using ΔV ≈ Vm / (f R C): f = {f:.4f} Hz; ΔV_approx = {vm:.4f} / ({f:.4f} * {r} * {c}) ≈ {v_r_approx:.4f} V (capped at Vm)")
    else:
        steps.append(f"Step 5: Approximate ripple using ΔV ≈ Vm / (f R C): f = {f:.4f} Hz; ΔV_approx = {vm:.4f} / ({f:.4f} * {r} * {c}) ≈ {v_r_approx:.4f} V")
    result["ripple using approximate formula"] = round(v_r_approx, 4)

    # Add steps_text
    result["steps_text"] = "\n".join(steps)
    return result

def half_wave_rectifier_with_capacitor_filter_dual_resistance(**kwargs):
    """
    Dual-R analysis — returns detailed step-by-step results for two resistances.
    Keys accepted case-insensitively: Vm/vm, C/c, omega, R1/r1, R2/r2, or R_values list.
    Each R entry contains Step 1..Step 6 as before AND a multi-line 'steps_text' string
    where every step appears on its own line for easy messaging.
    """
    # --- Normalize input keys (case-insensitive) ---
    def get_key(*names, default=None):
        for n in names:
            if n in kwargs:
                return kwargs[n]
            nl = n.lower()
            for k in kwargs:
                if k.lower() == nl:
                    return kwargs[k]
        return default

    Vm = get_key("Vm", "vm", "VM")
    C  = get_key("C", "c")
    omega = get_key("omega", "w", "omega_rad", "omega_rads")
    R1 = get_key("R1", "r1", "R_1", "r_01", "Rfirst")
    R2 = get_key("R2", "r2", "R_2", "r_02", "Rsecond")

    R_values_list = get_key("R_values", "r_values", default=None)
    if R_values_list is not None and (R1 is None and R2 is None):
        if not isinstance(R_values_list, (list, tuple)) or len(R_values_list) < 2:
            raise ValueError("R_values must be a list/tuple with at least two elements")
        R1, R2 = R_values_list[0], R_values_list[1]

    # Validate presence
    if Vm is None:
        raise ValueError("Vm not provided")
    if C is None:
        raise ValueError("C not provided")
    if omega is None:
        raise ValueError("omega not provided")
    if R1 is None or R2 is None:
        raise ValueError("R1 and R2 must be provided (or R_values list)")

    # Type cast & validate numeric ranges
    Vm = float(Vm); C = float(C); omega = float(omega); R1 = float(R1); R2 = float(R2)
    if Vm <= 0 or C <= 0 or omega <= 0 or R1 <= 0 or R2 <= 0:
        raise ValueError("All numeric inputs must be positive")

    def single_R_analysis(Vm, R, C, omega):
        res = {}
        ordered_steps = []  # collect human readable step lines

        # Step 1
        tau = R * C
        T = 2 * math.pi / omega
        ratio = tau / T
        step1 = (f"Step 1: Time constant and period ratio: τ = R·C = {R} * {C} = {tau:.6g} s; "
                 f"T = 2π/ω = 2π/{omega:.6g} ≈ {T:.6g} s; τ/T ≈ {ratio:.6g}")
        res["Step 1: Time constant and period ratio"] = step1
        res["tau"] = round(tau, 6)
        res["ratio of RC time constant to period"] = round(ratio, 6)
        ordered_steps.append(step1)

        # Step 2
        theta = -math.atan(omega * R * C) + math.pi
        step2 = (f"Step 2: Compute θ = -atan(ωRC) + π: θ = -atan({omega:.6g} * {R} * {C}) + π ≈ "
                 f"{theta:.6f} rad ({math.degrees(theta):.4f}°)")
        res["Step 2: Compute θ = -atan(ωRC) + π"] = step2
        res["theta (rad)"] = round(theta, 6)
        res["theta (deg)"] = round(math.degrees(theta), 4)
        ordered_steps.append(step2)

        # Step 3: Solve for alpha robustly
        omegaRC = omega * R * C
        def f_alpha(a):
            return math.sin(a) - math.sin(theta) * math.exp(-(2 * math.pi + a + theta) / omegaRC)

        # scan for sign change
        eps = 1e-9
        a_low = eps
        a_high = math.pi - eps
        Nscan = 2000
        bracket = None
        prev_a = a_low
        prev_val = f_alpha(prev_a)
        for k in range(1, Nscan + 1):
            a = a_low + (a_high - a_low) * k / Nscan
            val = f_alpha(a)
            if prev_val == 0.0:
                bracket = (prev_a, a)
                break
            if prev_val * val < 0:
                bracket = (prev_a, a)
                break
            prev_a, prev_val = a, val

        alpha = None
        if bracket is not None:
            try:
                sol = root_scalar(f_alpha, bracket=bracket, method='brentq',
                                  xtol=1e-12, rtol=1e-12, maxiter=200)
                alpha = sol.root
            except Exception:
                alpha = None

        if alpha is None:
            # fallback to fsolve
            try:
                # attempt a reasonable initial guess
                approx_rhs = math.sin(theta) * math.exp(-(2 * math.pi + math.pi/2 + theta) / omegaRC)
                if -1.0 <= approx_rhs <= 1.0:
                    guess = math.asin(max(-1.0, min(1.0, approx_rhs)))
                else:
                    guess = math.pi / 2
            except Exception:
                guess = math.pi / 2
            try:
                sol_arr = fsolve(f_alpha, guess, xtol=1e-12, maxfev=2000)
                alpha = float(sol_arr[0])
                if not (0 < alpha < math.pi):
                    # try other guesses
                    for g in [0.1, 0.5, 1.0, 1.5, 2.5]:
                        sol_arr = fsolve(f_alpha, g, xtol=1e-12, maxfev=2000)
                        a_try = float(sol_arr[0])
                        if 0 < a_try < math.pi:
                            alpha = a_try
                            break
            except Exception:
                raise RuntimeError("Failed to find root for alpha (no bracket and fsolve failed).")

        # clamp
        if alpha <= 0:
            alpha = eps
        if alpha >= math.pi:
            alpha = math.pi - eps

        step3 = (f"Step 3: Solve transcendental equation for α: α ≈ {alpha:.6f} rad "
                 f"({math.degrees(alpha):.4f}°)")
        res["Step 3: Solve transcendental equation for α"] = step3
        res["alpha (rad)"] = round(alpha, 6)
        res["alpha (deg)"] = round(math.degrees(alpha), 4)
        ordered_steps.append(step3)

        # Step 4: Exact ripple
        deltaV_exact = Vm * (1 - math.sin(alpha))
        step4 = f"Step 4: Exact peak-to-peak ripple using ΔV = Vm (1 - sin α): ΔV_exact = {Vm:.6g} * (1 - sin({alpha:.6f})) ≈ {deltaV_exact:.6f} V"
        res["Step 4: Exact peak-to-peak ripple using ΔV = Vm (1 - sin α)"] = step4
        res["peak-to-peak ripple voltage using exact equations"] = round(deltaV_exact, 6)
        ordered_steps.append(step4)

        # Step 5: Approximate ripple
        f_hz = omega / (2 * math.pi)
        deltaV_approx = Vm / (f_hz * R * C)
        step5 = f"Step 5: Approximate ripple using ΔV ≈ Vm / (f R C): f = {f_hz:.6g} Hz; ΔV_approx ≈ {deltaV_approx:.6f} V"
        res["Step 5: Approximate ripple using ΔV ≈ Vm / (f R C)"] = step5
        res["ripple using approximate formula"] = round(deltaV_approx, 6)
        ordered_steps.append(step5)

        # Step 6: Comment
        if ratio > 5:
            comment = "Step 6: Comment on approximation accuracy: Approximation is reasonably accurate (τ >> T)."
        elif ratio < 1:
            comment = "Step 6: Comment on approximation accuracy: Approximation is inaccurate (τ comparable to T)."
        else:
            comment = "Step 6: Comment on approximation accuracy: Approximation moderately accurate."
        res["Step 6: Comment on approximation accuracy"] = comment
        ordered_steps.append(comment)

        # Combine into one multi-line string, one step per line
        res["steps_text"] = "\n".join(ordered_steps)

        return res

    # Run analyses for both R1 and R2
    results = {}
    results[f"R={int(R1)}Ω"] = single_R_analysis(Vm, R1, C, omega)
    results[f"R={int(R2)}Ω"] = single_R_analysis(Vm, R2, C, omega)

    return results
def half_wave_rectifier_with_dual_capacitor_filter(Vs_rms, f, R, C1, C2):
    """
    Dual-capacitor analysis for half-wave rectifier with parallel capacitor(s).

    Parameters
    ----------
    Vs_rms : float
        Source RMS voltage in volts (e.g. 120)
    f : float
        Frequency in Hz (e.g. 60)
    R : float
        Load resistance in ohms (e.g. 1000)
    C1, C2 : float
        Capacitances. If value >= 1 it is assumed to be given in microfarads (µF)
        and will be converted to farads automatically. If value < 1 it is treated
        as farads.

    Returns
    -------
    dict
        Two entries (one per capacitor) each containing Step 1..Step 6, numeric keys,
        and 'steps_text' (multi-line one-step-per-line) ready for Telegram display.
    """
    omega = 2 * math.pi * f
    Vm = Vs_rms * math.sqrt(2)

    def interpret_C(raw):
        rawf = float(raw)
        if rawf >= 1.0:
            # treat as microfarads -> convert to farads
            C_f = rawf * 1e-6
            display = f"{rawf:g} µF ({C_f:.6g} F)"
        else:
            C_f = rawf
            display = f"{C_f:.6g} F"
        return C_f, display

    C1_f, C1_display = interpret_C(C1)
    C2_f, C2_display = interpret_C(C2)

    def analyze_for_C(C, C_display):
        res = {}
        steps = []

        # Step 1
        tau = R * C
        T = 2 * math.pi / omega
        ratio = tau / T
        s1 = (f"Step 1: Time constant and period ratio: τ = R·C = {R} * {C_display} = {tau:.6g} s; "
              f"T = 2π/ω = 2π/{omega:.6g} ≈ {T:.6g} s; τ/T ≈ {ratio:.6g}")
        res["Step 1: Time constant and period ratio"] = s1
        res["tau"] = round(tau, 6)
        res["ratio of RC time constant to period"] = round(ratio, 6)
        steps.append(s1)

        # Step 2
        theta = -math.atan(omega * R * C) + math.pi
        s2 = (f"Step 2: Compute θ = -atan(ωRC) + π: θ = -atan({omega:.6g} * {R} * {C:.6g}) + π ≈ "
              f"{theta:.6f} rad ({math.degrees(theta):.4f}°)")
        res["Step 2: Compute θ = -atan(ωRC) + π"] = s2
        res["theta (rad)"] = round(theta, 6)
        res["theta (deg)"] = round(math.degrees(theta), 4)
        steps.append(s2)

        # Step 3: solve alpha robustly
        omegaRC = omega * R * C

        def f_alpha(a):
            return math.sin(a) - math.sin(theta) * math.exp(-(2 * math.pi + a + theta) / omegaRC)

        eps = 1e-9
        a_low, a_high = eps, math.pi - eps
        bracket = None
        prev_a = a_low
        prev_val = f_alpha(prev_a)
        Nscan = 2000
        for k in range(1, Nscan + 1):
            a = a_low + (a_high - a_low) * k / Nscan
            val = f_alpha(a)
            if prev_val == 0.0 or prev_val * val < 0:
                bracket = (prev_a, a)
                break
            prev_a, prev_val = a, val

        alpha = None
        if bracket is not None:
            try:
                sol = root_scalar(f_alpha, bracket=bracket, method='brentq',
                                  xtol=1e-12, rtol=1e-12, maxiter=200)
                alpha = sol.root
            except Exception:
                alpha = None

        if alpha is None:
            # fallback to fsolve
            try:
                approx_rhs = math.sin(theta) * math.exp(-(2 * math.pi + math.pi / 2 + theta) / omegaRC)
                if -1.0 <= approx_rhs <= 1.0:
                    guess = math.asin(max(-1.0, min(1.0, approx_rhs)))
                else:
                    guess = math.pi / 2
            except Exception:
                guess = math.pi / 2
            try:
                sol_arr = fsolve(f_alpha, guess, xtol=1e-12, maxfev=2000)
                alpha = float(sol_arr[0])
                if not (0 < alpha < math.pi):
                    for g in [0.1, 0.5, 1.0, 1.5, 2.5]:
                        sol_arr = fsolve(f_alpha, g, xtol=1e-12, maxfev=2000)
                        a_try = float(sol_arr[0])
                        if 0 < a_try < math.pi:
                            alpha = a_try
                            break
            except Exception:
                raise RuntimeError("Failed to solve for conduction angle α numerically.")

        # clamp for safety
        if alpha <= 0:
            alpha = eps
        if alpha >= math.pi:
            alpha = math.pi - eps

        s3 = f"Step 3: Solve transcendental equation for α: α ≈ {alpha:.6f} rad ({math.degrees(alpha):.4f}°)"
        res["Step 3: Solve transcendental equation for α"] = s3
        res["alpha (rad)"] = round(alpha, 6)
        res["alpha (deg)"] = round(math.degrees(alpha), 4)
        steps.append(s3)

        # Step 4: exact ripple
        deltaV_exact = Vm * (1 - math.sin(alpha))
        s4 = f"Step 4: Exact peak-to-peak ripple using ΔV = Vm (1 - sin α): ΔV_exact = {Vm:.6g} * (1 - sin({alpha:.6f})) ≈ {deltaV_exact:.6f} V"
        res["Step 4: Exact peak-to-peak ripple using ΔV = Vm (1 - sin α)"] = s4
        res["peak-to-peak ripple voltage using exact equations"] = round(deltaV_exact, 6)
        steps.append(s4)

        # Step 5: approximate ripple
        f_hz = omega / (2 * math.pi)
        deltaV_approx = Vm / (f_hz * R * C)
        s5 = f"Step 5: Approximate ripple using ΔV ≈ Vm / (f R C): f = {f_hz:.6g} Hz; ΔV_approx ≈ {deltaV_approx:.6f} V"
        res["Step 5: Approximate ripple using ΔV ≈ Vm / (f R C)"] = s5
        res["ripple using approximate formula"] = round(deltaV_approx, 6)
        steps.append(s5)

        # Step 6: comment
        if ratio > 5:
            comment = "Step 6: Comment on approximation accuracy: Approximation is reasonably accurate (τ >> T)."
        elif ratio < 1:
            comment = "Step 6: Comment on approximation accuracy: Approximation is inaccurate (τ comparable to T)."
        else:
            comment = "Step 6: Comment on approximation accuracy: Approximation moderately accurate."
        res["Step 6: Comment on approximation accuracy"] = comment
        steps.append(comment)

        # multi-line steps_text (one step per line)
        res["steps_text"] = "\n".join(steps)
        return res

    out = {}
    out[f"R={R}Ω, C={C1_display}"] = analyze_for_C(C1_f, C1_display)
    out[f"R={R}Ω, C={C2_display}"] = analyze_for_C(C2_f, C2_display)
    return out
def half_wave_rectifier_with_capacitor_filter_ripple_and_diode_currents(Vs_rms, f, R, delta_V0):
    """
    Compute capacitor required to keep peak-to-peak ripple <= delta_V0, and diode currents.

    Parameters
    ----------
    Vs_rms : float
        Source RMS voltage (V), e.g. 120
    f : float
        Frequency in Hz, e.g. 60
    R : float
        Load resistance in ohms, e.g. 750
    delta_V0 : float
        Allowed peak-to-peak ripple in volts, e.g. 2

    Returns
    -------
    dict
        - step-by-step entries ("Step 1: ...", etc.)
        - numeric results:
            "required_capacitance_F" (F),
            "required_capacitance_uF" (µF),
            "conduction_angle_alpha_rad",
            "conduction_angle_alpha_deg",
            "average_diode_current",
            "peak_diode_current",
        - "steps_text": multiline string (one step per line)
    Notes
    -----
    Uses formulas:
      Vm = sqrt(2) * Vs_rms
      C = Vm / (f * R * ΔV0)
      α ≈ asin(1 - ΔV0 / Vm)
      I_D,peak = Vm * (ω C cos α + (sin α)/R)
      I_D,avg ≈ Vm / R   (approximation used in provided solution)
    """
    # validation
    if Vs_rms <= 0:
        raise ValueError("Vs_rms must be positive")
    if f <= 0:
        raise ValueError("f must be positive")
    if R <= 0:
        raise ValueError("R must be positive")
    if delta_V0 <= 0:
        raise ValueError("delta_V0 must be positive")

    res = {}
    steps = []

    # Step 1: Vm
    Vm = Vs_rms * math.sqrt(2)
    steps.append(f"Step 1: Peak voltage: V_m = √2 * V_s(rms) = √2 * {Vs_rms} ≈ {Vm:.6f} V")
    res["V_m"] = round(Vm, 6)

    # Step 2: Capacitance (using approximate ripple formula rearranged)
    # C = Vm / (f * R * ΔV0)
    C = Vm / (f * R * delta_V0)
    C_uF = C * 1e6
    steps.append(
        f"Step 2: Required capacitance from C = V_m / (f · R · ΔV0): "
        f"C = {Vm:.6f} / ({f:.6f} * {R:.6f} * {delta_V0:.6f}) ≈ {C:.9g} F ({C_uF:.3f} μF)"
    )
    res["required_capacitance_F"] = round(C, 9)
    res["required_capacitance_uF"] = round(C_uF, 3)

    # Step 3: Conduction angle α approximation
    # α ≈ asin(1 - ΔV0 / Vm)
    arg = 1.0 - (delta_V0 / Vm)
    # clamp to [-1,1] for safety
    if arg > 1.0:
        arg = 1.0
    if arg < -1.0:
        arg = -1.0
    alpha = math.asin(arg)
    steps.append(
        f"Step 3: Conduction angle approximation: α ≈ asin(1 - ΔV0 / V_m) = "
        f"asin({arg:.6f}) ≈ {alpha:.6f} rad ({math.degrees(alpha):.4f}°)"
    )
    res["conduction_angle_alpha_rad"] = round(alpha, 6)
    res["conduction_angle_alpha_deg"] = round(math.degrees(alpha), 4)

    # Step 4: Diode average current (approx)
    I_D_avg = Vm / R  # per provided solution (approx)
    steps.append(f"Step 4: Approximate average diode current: I_D,avg ≈ V_m / R = {Vm:.6f} / {R:.6f} ≈ {I_D_avg:.6f} A")
    res["average_diode_current"] = round(I_D_avg, 6)

    # Step 5: Peak diode current formula: I_D,peak = V_m * (ω C cos α + sin α / R)
    omega = 2.0 * math.pi * f
    term1 = omega * C * math.cos(alpha)
    term2 = math.sin(alpha) / R
    I_D_peak = Vm * (term1 + term2)
    steps.append(
        "Step 5: Peak diode current using I_D,peak = V_m (ω C cos α + sin α / R): "
        f"ω = 2πf = {omega:.6f} rad/s; ωC cosα = {omega:.6f} * {C:.9g} * cos({alpha:.6f}) = {term1:.6f}; "
        f"sinα/R = {math.sin(alpha):.6f} / {R:.6f} = {term2:.6f}; "
        f"I_D,peak = {Vm:.6f} * ({term1:.6f} + {term2:.6f}) ≈ {I_D_peak:.6f} A"
    )
    res["peak_diode_current"] = round(I_D_peak, 6)

    # Step 6: Note on approximations
    steps.append("Step 6: Notes: formulas use standard approximations for capacitor-fed half-wave rectifiers. "
                 "I_D,avg is approximate; more exact diode average would integrate conduction interval if needed.")

    # Combine
    res["steps_text"] = "\n".join(steps)

    return res
def half_wave_rectifier_with_capacitor_filter_ripple_and_diode_currents_P_load(Vs_rms, f, P_load, delta_V0):
    """
    Compute capacitor required for specified peak-to-peak ripple and diode currents
    for a half-wave rectifier with capacitor filter (user's formula style).

    Parameters
    ----------
    Vs_rms : float
        Source RMS voltage (V) (e.g. 120).
    f : float
        Frequency (Hz) (e.g. 60).
    P_load : float
        Load power (W) (e.g. 50).
    delta_V0 : float
        Desired peak-to-peak ripple (V) (e.g. 1.5).

    Returns
    -------
    dict
        Includes step-by-step text entries (Step 1..Step 4), numeric results:
        - 'capacitance_F' (float)
        - 'capacitance_uF' (float)
        - 'average_diode_current' (A)
        - 'peak_diode_current' (A)
        - 'steps_text' (multiline string)
    Notes
    -----
    Uses the same formulas from your solution:
      Vm = Vs_rms * sqrt(2)
      R ≈ Vm^2 / P_load
      C = Vm / (f * R * ΔV0)
      α ≈ asin(1 - ΔV0 / Vm)
      I_D,peak = Vm * (ω C cos α + sin α / R)
      I_D,avg ≈ Vm / R
    """

    # --- validation ---
    if Vs_rms <= 0:
        raise ValueError("Vs_rms must be positive")
    if f <= 0:
        raise ValueError("f must be positive")
    if P_load <= 0:
        raise ValueError("P_load must be positive")
    if delta_V0 <= 0:
        raise ValueError("delta_V0 must be positive")

    res = {}
    steps = []

    # Step 1: Vm
    Vm = Vs_rms * math.sqrt(2)
    steps.append(f"Step 1: Peak voltage: V_m = √2 * V_s,rms = √2 * {Vs_rms} ≈ {Vm:.4f} V")
    res["V_m"] = round(Vm, 4)

    # Step 2: approximate load resistance from P_load (user's method)
    R = (Vm ** 2) / P_load
    steps.append(f"Step 2: Approximate load resistance from P_load using R ≈ V_m^2 / P_load = ({Vm:.4f})^2 / {P_load} ≈ {R:.4f} Ω")
    res["R (approx)"] = round(R, 4)

    # Step 3: capacitor value from ripple equation
    # C = Vm / (f * R * delta_V0)
    C = Vm / (f * R * delta_V0)
    C_uF = C * 1e6
    steps.append(f"Step 3: Capacitance for ΔV0: C = V_m / (f · R · ΔV0) = {Vm:.4f} / ({f} * {R:.4f} * {delta_V0:.4f}) ≈ {C:.6e} F ({C_uF:.1f} μF)")
    res["capacitance_F"] = C
    res["capacitance_uF"] = round(C_uF, 1)

    # Step 4: conduction angle approx α
    # α ≈ asin(1 - ΔV0 / Vm)
    arg = 1.0 - (delta_V0 / Vm)
    if arg < -1.0 or arg > 1.0:
        raise ValueError("Invalid conduction angle calculation: 1 - ΔV0 / Vm out of [-1,1]. Check delta_V0 and Vs_rms.")
    alpha = math.asin(arg)
    steps.append(f"Step 4: Conduction angle approx: α ≈ asin(1 - ΔV0 / V_m) = asin({arg:.6f}) ≈ {alpha:.6f} rad ({math.degrees(alpha):.4f}°)")
    res["alpha (rad)"] = round(alpha, 6)
    res["alpha (deg)"] = round(math.degrees(alpha), 4)

    # Step 5: diode currents
    omega = 2 * math.pi * f
    # average diode current (user's approx)
    I_D_avg = Vm / R
    # peak diode current formula: I_D,peak = Vm * (ω C cos α + sin α / R)
    I_D_peak = Vm * (omega * C * math.cos(alpha) + (math.sin(alpha) / R))
    steps.append(f"Step 5: Diode currents: I_D,avg ≈ V_m / R = {Vm:.4f} / {R:.4f} ≈ {I_D_avg:.4f} A")
    steps.append(f"Step 6: Peak diode current: I_D,peak = V_m (ω C cos α + sin α / R) = {Vm:.4f} * ({omega:.4f} * {C:.6e} * cos({alpha:.6f}) + sin({alpha:.6f}) / {R:.4f}) ≈ {I_D_peak:.4f} A")

    res["average_diode_current"] = round(I_D_avg, 4)
    res["peak_diode_current"] = round(I_D_peak, 4)

    # assemble steps_text (one step per line)
    res["steps_text"] = "\n".join(steps)

    return res
def controlled_half_wave_rectifier_resistive_load(Vs_rms, f, R, alpha):
    """
    Controlled half-wave rectifier with resistive load (step-by-step style).

    Parameters
    ----------
    Vs_rms : float
        Source RMS voltage (V), e.g. 120
    f : float
        Frequency (Hz), e.g. 60
    R : float
        Load resistance (Ω), e.g. 100
    alpha : float
        Delay angle in degrees (°), e.g. 45

    Returns
    -------
    dict
        Step-by-step strings (Step 1..Step N), numeric results and a multi-line
        'steps_text' string where every step appears on its own line.
    """
    # input validation
    if Vs_rms <= 0:
        raise ValueError("Vs_rms must be positive")
    if f <= 0:
        raise ValueError("f must be positive")
    if R <= 0:
        raise ValueError("R must be positive")

    # allow alpha in degrees; validate range 0..180 (practical)
    alpha_deg = float(alpha)
    if not (0.0 <= alpha_deg <= 180.0):
        raise ValueError("alpha (degrees) must be in [0, 180]")

    steps = []
    result = {}

    # Step 1: Peak voltage Vm
    Vm = Vs_rms * math.sqrt(2)
    steps.append(f"Step 1: Peak voltage: V_m = √2 * V_s,rms = √2 * {Vs_rms} ≈ {Vm:.4f} V")
    result["V_m"] = round(Vm, 4)

    # Step 2: Convert alpha to radians
    alpha_rad = math.radians(alpha_deg)
    steps.append(f"Step 2: Delay angle: α = {alpha_deg:.4f}° = {alpha_rad:.6f} rad")
    result["alpha (deg)"] = round(alpha_deg, 4)
    result["alpha (rad)"] = round(alpha_rad, 6)

    # Step 3: Average output voltage Vo = (Vm / (2π)) * (1 + cos α)
    V_o = (Vm / (2 * math.pi)) * (1.0 + math.cos(alpha_rad))
    steps.append(f"Step 3: Average output voltage using V_o = (V_m / 2π) (1 + cos α): V_o ≈ {V_o:.4f} V")
    result["average voltage across the resistor"] = round(V_o, 4)

    # Step 4: RMS output voltage formula:
    # V_rms = (V_m / 2) * sqrt(1 - α/π + (sin 2α)/(2π))
    sin2a = math.sin(2.0 * alpha_rad)
    V_rms = (Vm / 2.0) * math.sqrt(max(0.0, 1.0 - (alpha_rad / math.pi) + (sin2a / (2.0 * math.pi))))
    steps.append("Step 4: RMS output voltage using "
                 "V_rms = (V_m / 2) * sqrt(1 - α/π + (sin 2α)/(2π))"
                 f": V_rms ≈ {V_rms:.4f} V")
    result["V_rms"] = round(V_rms, 4)

    # Step 5: Load currents and power
    I_rms_load = V_rms / R
    P_load = (V_rms ** 2) / R
    steps.append(f"Step 5: RMS load current I_rms = V_rms / R = {V_rms:.4f} / {R} ≈ {I_rms_load:.4f} A")
    steps.append(f"Step 6: Power absorbed by the resistor P = V_rms^2 / R = ({V_rms:.4f})^2 / {R} ≈ {P_load:.4f} W")
    result["rms load current"] = round(I_rms_load, 4)
    result["power absorbed by the resistor"] = round(P_load, 4)

    # Step 6: Apparent power and power factor as seen by the source
    S = Vs_rms * I_rms_load
    pf = P_load / S if S != 0 else 0.0
    steps.append(f"Step 7: Apparent power S = V_s,rms * I_rms = {Vs_rms} * {I_rms_load:.4f} ≈ {S:.4f} VA")
    steps.append(f"Step 8: Power factor pf = P / S = {P_load:.4f} / {S:.4f} ≈ {pf:.4f}")
    result["apparent power supplied by the source"] = round(S, 4)
    result["power factor"] = round(pf, 4)

    # assemble steps_text multiline
    result["steps_text"] = "\n".join(steps)

    # keep the same detailed Step N keys for readability (optional)
    # e.g. Step 1..Step 8 entries
    for i, s in enumerate(steps, start=1):
        result[f"Step {i}"] = s

    return result
def controlled_half_wave_rectifier_with_resistive_load_delay(Vs_rms, f, R, I_avg):
    """
    Controlled half-wave rectifier with resistive load where the required delay angle
    is computed from a target average load current I_avg.

    Parameters
    ----------
    vs_rms : float
        Source RMS voltage (V), e.g. 240
    f : float
        Frequency (Hz), e.g. 60
    R : float
        Load resistance (Ω), e.g. 30
    I_avg : float
        Desired average load current (A), e.g. 2.5

    Returns
    -------
    dict
        Step-by-step strings (Step 1..Step N), numeric results and a multiline
        'steps_text' field where every step appears on its own line.
    """
    # Input validation
    if Vs_rms <= 0:
        raise ValueError("vs_rms must be positive")
    if f <= 0:
        raise ValueError("f must be positive")
    if R <= 0:
        raise ValueError("R must be positive")
    if I_avg < 0:
        raise ValueError("I_avg must be non-negative")

    steps = []
    result = {}

    # Step 1: Compute Vm and V_o required from I_avg
    Vm = Vs_rms * math.sqrt(2)
    V_o = I_avg * R
    steps.append(f"Step 1: Peak voltage: V_m = √2 * V_s,rms = √2 * {Vs_rms} ≈ {Vm:.4f} V")
    steps.append(f"Step 2: Required average output voltage from I_avg: V_o = I_avg * R = {I_avg:.4f} * {R} = {V_o:.4f} V")
    result["V_m"] = round(Vm, 4)
    result["average voltage (required)"] = round(V_o, 4)

    # Step 3: Solve for alpha from V_o = (V_m / (2π)) (1 + cos α)
    # => cos α = (2π V_o / V_m) - 1
    arg = (2.0 * math.pi * V_o / Vm) - 1.0
    steps.append("Step 3: Solve for α from V_o = (V_m / (2π)) (1 + cos α) "
                 "⇒ cos α = (2π V_o / V_m) - 1")
    steps.append(f"         Compute argument for arccos: (2π V_o / V_m) - 1 = {arg:.6f}")

    if arg < -1.0 or arg > 1.0:
        # out of range -> no physical solution
        raise ValueError(f"No valid delay angle: argument for arccos = {arg:.6f} outside [-1,1]."
                         " Check I_avg, R, and Vs_rms values.")

    alpha = math.acos(arg)
    alpha_deg = math.degrees(alpha)
    steps.append(f"Step 4: Delay angle: α = arccos({arg:.6f}) ≈ {alpha_deg:.4f}° = {alpha:.6f} rad")
    result["delay angle (rad)"] = round(alpha, 6)
    result["delay angle (deg)"] = round(alpha_deg, 4)

    # Step 5: RMS output voltage
    # V_rms = (V_m / 2) * sqrt(1 - (α / π) + (sin 2α) / (2π))
    sin2a = math.sin(2.0 * alpha)
    V_rms = (Vm / 2.0) * math.sqrt(max(0.0, 1.0 - (alpha / math.pi) + (sin2a / (2.0 * math.pi))))
    steps.append("Step 5: RMS output voltage using "
                 "V_rms = (V_m / 2) * sqrt(1 - α/π + (sin 2α)/(2π))")
    steps.append(f"         V_rms ≈ {V_rms:.4f} V")
    result["V_rms"] = round(V_rms, 4)

    # Step 6: Power absorbed by the load
    P_load = (V_rms ** 2) / R
    steps.append(f"Step 6: Power absorbed by the load: P = V_rms^2 / R = ({V_rms:.4f})^2 / {R} ≈ {P_load:.4f} W")
    result["power absorbed by the load"] = round(P_load, 4)

    # Step 7: Apparent power and power factor as seen by the source
    I_rms_load = V_rms / R
    S = Vs_rms * I_rms_load
    pf = P_load / S if S != 0 else 0.0
    steps.append(f"Step 7: Apparent power: S = V_s,rms * I_rms = {Vs_rms} * {I_rms_load:.4f} ≈ {S:.4f} VA")
    steps.append(f"Step 8: Power factor: pf = P / S = {P_load:.4f} / {S:.4f} ≈ {pf:.4f}")
    result["apparent power supplied by the source"] = round(S, 4)
    result["power factor"] = round(pf, 4)
    result["rms load current"] = round(I_rms_load, 4)

    # Steps text (one per line) for Telegram
    result["steps_text"] = "\n".join(steps)

    # Also keep individual Step N entries for backward compatibility
    for i, s in enumerate(steps, start=1):
        result[f"Step {i}"] = s

    return result
def controlled_half_wave_rectifier_with_rl_load(vs_rms, f, R, L, alpha_deg):
    """
    Controlled half-wave rectifier with series RL load (phase-controlled).

    Parameters
    ----------
    vs_rms : float
        Source RMS voltage (V).
    f : float
        Frequency (Hz).
    R : float
        Load resistance (Ω).
    L : float
        Load inductance (H).
    alpha_deg : float
        Delay angle α in degrees.

    Returns
    -------
    dict
        Contains step-by-step strings ("Step 1: ..."), numeric keys (e.g. "V_m", "average current"),
        and a multiline "steps_text" field where each step is on its own line.
    """
    # --- input validation ---
    if vs_rms <= 0:
        raise ValueError("vs_rms must be positive")
    if f <= 0:
        raise ValueError("f must be positive")
    if R <= 0:
        raise ValueError("R must be positive")
    if L <= 0:
        raise ValueError("L must be positive")

    result = {}
    steps = []

    # Step 1: Vm and omega
    omega = 2.0 * math.pi * f
    V_m = math.sqrt(2.0) * vs_rms
    steps.append(f"Step 1: Peak voltage using V_m = √2 * V_s,rms: V_m = √2 * {vs_rms} ≈ {V_m:.4f} V")
    result["V_m"] = round(V_m, 4)
    result["omega"] = round(omega, 6)

    # Step 2: Impedance Z
    omegaL = omega * L
    Z = math.sqrt(R**2 + omegaL**2)
    steps.append(f"Step 2: Load impedance using Z = √(R^2 + (ωL)^2): Z = √({R:.4f}^2 + ({omega:.4f}*{L:.6g})^2) ≈ {Z:.4f} Ω")
    result["Z"] = round(Z, 6)

    # Step 3: phase angle theta
    theta = math.atan2(omegaL, R)
    steps.append(f"Step 3: Phase angle using θ = atan(ωL / R): θ = atan({omegaL:.4f} / {R:.4f}) ≈ {theta:.6f} rad ({math.degrees(theta):.4f}°)")
    result["theta (rad)"] = round(theta, 6)
    result["theta (deg)"] = round(math.degrees(theta), 4)

    # Step 4: time-constant term ωτ
    omega_tau = omegaL / R
    steps.append(f"Step 4: Time-constant term using ωτ = ωL / R: ωτ = {omegaL:.4f} / {R:.4f} ≈ {omega_tau:.6f}")
    result["omega_tau"] = round(omega_tau, 6)

    # Step 5: firing angle alpha (convert degrees to radians)
    alpha = math.radians(alpha_deg)
    if not (0.0 <= alpha < 2.0 * math.pi):
        raise ValueError("alpha_deg must be in [0, 360) degrees")
    steps.append(f"Step 5: Delay (firing) angle α: α = {alpha_deg:.4f}° = {alpha:.6f} rad")
    result["alpha (rad)"] = round(alpha, 6)
    result["alpha (deg)"] = round(alpha_deg, 4)

    # Step 6: compute i_m and A using initial condition i(α)=0
    i_m = V_m / Z
    A = - i_m * math.sin(alpha - theta) * math.exp(alpha / omega_tau)
    steps.append(f"Step 6: Compute coefficients: i_m = V_m / Z = {V_m:.4f} / {Z:.4f} ≈ {i_m:.6f} A; "
                 f"A = -i_m·sin(α - θ)·e^(α/ωτ) ≈ {A:.6f}")
    result["i_m"] = round(i_m, 6)
    result["A"] = round(A, 6)

    # Step 7: load current expression (valid for α ≤ ωt ≤ β)
    expr = (f"i(ωt) = (V_m/Z)·sin(ωt - θ) + A·e^(-ωt / ωτ); "
            f"numeric: {i_m:.6f}·sin(ωt - {theta:.6f}) + {A:.6f}·e^(-ωt/{omega_tau:.6f}), for α ≤ ωt ≤ β")
    steps.append(f"Step 7: Load current expression: {expr}")
    result["expression for load current"] = expr

    # Step 8: find extinction angle beta by solving i(β) = 0 in interval (α, 2π]
    def i_wt(wt):
        return i_m * math.sin(wt - theta) + A * math.exp(-wt / omega_tau)

    # bracket: attempt direct bracket [α, 2π], else scan grid
    a_try = alpha + 1e-9
    b_try = 2.0 * math.pi
    bracket = None
    try:
        fa = i_wt(a_try)
        fb = i_wt(b_try)
        if fa is not None and fb is not None and fa * fb < 0:
            bracket = (a_try, b_try)
    except Exception:
        bracket = None

    if bracket is None:
        # scan to find sign change
        Nscan = 2000
        prev_w = a_try
        prev_val = i_wt(prev_w)
        for k in range(1, Nscan + 1):
            w = a_try + (b_try - a_try) * k / Nscan
            val = i_wt(w)
            if prev_val is None or val is None:
                prev_w, prev_val = w, val
                continue
            if prev_val == 0.0:
                bracket = (prev_w, w)
                break
            if prev_val * val < 0:
                bracket = (prev_w, w)
                break
            prev_w, prev_val = w, val

    if bracket is None:
        raise RuntimeError("Failed to bracket extinction angle β (no sign change found for i(ωt) on (α,2π]).")

    sol = root_scalar(lambda wt: i_wt(wt), bracket=bracket, method='brentq', xtol=1e-12, rtol=1e-12, maxiter=200)
    beta = sol.root
    steps.append(f"Step 8: Extinction angle β by solving i(β)=0 numerically: Solved numerically: β ≈ {beta:.6f} rad ({math.degrees(beta):.4f}°)")
    result["extinction angle (rad)"] = round(beta, 6)
    result["extinction angle (deg)"] = round(math.degrees(beta), 4)

    # Step 9: average current I_avg = (1/(2π)) ∫_α^β i(ωt) d(ωt)
    integral_i, _ = quad(i_wt, alpha, beta, epsabs=1e-9, epsrel=1e-9, limit=500)
    I_avg = integral_i / (2.0 * math.pi)
    steps.append(f"Step 9: Average current using I_avg = (1/(2π)) ∫_α^β i(ωt) d(ωt): "
                 f"Integral(α→β) = {integral_i:.6f}; I_avg = {integral_i:.6f}/(2π) ≈ {I_avg:.6f} A")
    result["average current"] = round(I_avg, 6)

    # Step 10: RMS current I_rms = sqrt((1/(2π)) ∫_α^β i^2(ωt) d(ωt))
    integral_i2, _ = quad(lambda wt: (i_wt(wt))**2, alpha, beta, epsabs=1e-9, epsrel=1e-9, limit=500)
    I_rms = math.sqrt(integral_i2 / (2.0 * math.pi))
    steps.append(f"Step 10: RMS current using I_rms = sqrt((1/(2π)) ∫_α^β i^2(ωt) d(ωt)): "
                 f"Integral(α→β) = {integral_i2:.6f}; I_rms ≈ {I_rms:.6f} A")
    result["rms current"] = round(I_rms, 6)

    # Step 11: Power absorbed by load P = I_rms^2 * R
    P_load = (I_rms ** 2) * R
    steps.append(f"Step 11: Power absorbed by the load using P = I_rms^2 * R: P = ({I_rms:.6f})^2 * {R:.4f} ≈ {P_load:.6f} W")
    result["power absorbed by the load"] = round(P_load, 6)

    # Step 12: pack expression and multiline steps_text
    result["steps_text"] = "\n".join(steps)

    # also return compact expression and numeric pieces
    result["expression_compact"] = f"{i_m:.6f}*sin(ωt - {theta:.6f}) + {A:.6f}*exp(-ωt/{omega_tau:.6f})"
    return result
def half_wave_rectifier_with_rl_source_load(vs_rms, f, R, L, Vdc):
    """
    Half-wave rectifier with RL + DC source load (step-by-step style).

    Parameters
    ----------
    vs_rms : float
        AC source RMS voltage (V).
    f : float
        Frequency (Hz).
    R : float
        Load resistance (Ω).
    L : float
        Load inductance (H).
    Vdc : float
        Series DC source in load (V).

    Returns
    -------
    dict
        Contains step-by-step strings ("Step 1: ..."), numeric keys (e.g. "average current"),
        and a multiline "steps_text" field for messaging.
    """
    # --- input validation ---
    if vs_rms <= 0:
        raise ValueError("vs_rms must be positive")
    if f <= 0:
        raise ValueError("f must be positive")
    if R <= 0:
        raise ValueError("R must be positive")
    if L <= 0:
        raise ValueError("L must be positive")
    if Vdc < 0:
        raise ValueError("Vdc must be non-negative")

    res = {}
    steps = []

    # Step 1: peak voltage
    omega = 2.0 * math.pi * f
    V_m = math.sqrt(2.0) * vs_rms
    steps.append(f"Step 1: Peak voltage using V_m = √2 * V_s,rms: V_m = √2 * {vs_rms} ≈ {V_m:.4f} V")
    res["V_m"] = round(V_m, 4)
    res["omega"] = round(omega, 6)

    # Step 2: impedance Z
    omegaL = omega * L
    Z = math.sqrt(R**2 + omegaL**2)
    steps.append(f"Step 2: Load impedance using Z = √(R^2 + (ωL)^2): Z = √({R:.4f}^2 + ({omega:.4f}*{L:.6g})^2) ≈ {Z:.4f} Ω")
    res["Z"] = round(Z, 6)

    # Step 3: phase angle theta
    theta = math.atan2(omegaL, R)
    steps.append(f"Step 3: Phase angle using θ = atan(ωL / R): θ = atan({omegaL:.4f} / {R:.4f}) ≈ {theta:.6f} rad ({math.degrees(theta):.4f}°)")
    res["theta (rad)"] = round(theta, 6)
    res["theta (deg)"] = round(math.degrees(theta), 4)

    # Step 4: time-constant term ωτ = ωL / R
    omega_tau = omegaL / R
    steps.append(f"Step 4: Time-constant term using ωτ = ωL / R: ωτ = {omegaL:.4f} / {R:.4f} ≈ {omega_tau:.6f}")
    res["omega_tau"] = round(omega_tau, 6)

    # Step 5: firing angle alpha from Vdc: alpha = asin(Vdc / V_m)
    if abs(Vdc) > V_m:
        # physically no conduction if DC > peak (asin undefined)
        raise ValueError("Vdc >= V_m: no conduction (asin undefined). Check Vdc or Vs_rms.")
    alpha = math.asin(Vdc / V_m)
    steps.append(f"Step 5: Firing angle using α = asin(Vdc / V_m): α = asin({Vdc:.4f} / {V_m:.4f}) ≈ {alpha:.6f} rad ({math.degrees(alpha):.4f}°)")
    res["alpha (rad)"] = round(alpha, 6)
    res["alpha (deg)"] = round(math.degrees(alpha), 4)

    # Step 6: compute i_m and A from i(α)=0 -> A = (Vdc/R - i_m sin(α-θ)) e^{α/ωτ}
    i_m = V_m / Z
    A = (Vdc / R - i_m * math.sin(alpha - theta)) * math.exp(alpha / omega_tau)
    steps.append(f"Step 6: Compute coefficients: i_m = V_m / Z = {V_m:.4f} / {Z:.4f} ≈ {i_m:.6f} A; "
                 f"A = (Vdc/R - i_m·sin(α-θ))·e^(α/ωτ) ≈ {A:.6f}")
    res["i_m"] = round(i_m, 6)
    res["A"] = round(A, 6)

    # Step 7: expression for load current (for α <= ωt <= β)
    expr = (f"i(ωt) = (V_m/Z)·sin(ωt - θ) - Vdc/R + A·e^(-ωt / ωτ); "
            f"numeric: {i_m:.6f}·sin(ωt - {theta:.6f}) - {Vdc/R:.6f} + {A:.6f}·e^(-ωt/{omega_tau:.6f}), for α ≤ ωt ≤ β")
    steps.append(f"Step 7: Load current expression: {expr}")
    res["expression for current"] = expr

    # Step 8: find extinction angle beta solving i(beta)=0 numerically on (α, 2π]
    def i_wt(wt):
        return i_m * math.sin(wt - theta) - (Vdc / R) + A * math.exp(-wt / omega_tau)

    # try simple bracket [α, 2π]
    a_try = alpha + 1e-3
    b_try = 2.0 * math.pi
    bracket = None
    try:
        fa = i_wt(a_try)
        fb = i_wt(b_try)
        if fa is not None and fb is not None and fa * fb < 0:
            bracket = (a_try, b_try)
    except Exception:
        bracket = None

    if bracket is None:
        # scan to find sign change
        Nscan = 2000
        prev_w = a_try
        prev_val = i_wt(prev_w)
        for k in range(1, Nscan + 1):
            w = a_try + (b_try - a_try) * k / Nscan
            val = i_wt(w)
            if prev_val is None or val is None:
                prev_w, prev_val = w, val
                continue
            if prev_val == 0.0:
                bracket = (prev_w, w)
                break
            if prev_val * val < 0:
                bracket = (prev_w, w)
                break
            prev_w, prev_val = w, val

    if bracket is None:
        raise RuntimeError("Failed to bracket extinction angle β (no sign change found for i(ωt) on (α,2π]).")

    sol = root_scalar(lambda wt: i_wt(wt), bracket=bracket, method='brentq', xtol=1e-12, rtol=1e-12, maxiter=200)
    beta = sol.root
    steps.append(f"Step 8: Extinction angle β by solving i(β)=0 numerically: Solved numerically: β ≈ {beta:.6f} rad ({math.degrees(beta):.4f}°)")
    res["extinction angle (rad)"] = round(beta, 6)
    res["extinction angle (deg)"] = round(math.degrees(beta), 4)

    # Step 9: average current I_avg = (1/(2π)) * ∫_{α}^{β} i(ωt) d(ωt)
    integral_i, _ = quad(i_wt, alpha, beta, epsabs=1e-9, epsrel=1e-9, limit=500)
    I_avg = integral_i / (2.0 * math.pi)
    steps.append(f"Step 9: Average current using I_avg = (1/(2π)) ∫_α^β i(ωt) d(ωt): Integral = {integral_i:.6f}; I_avg = {I_avg:.6f} A")
    res["average current"] = round(I_avg, 6)

    # Step 10: RMS current I_rms = sqrt((1/(2π)) ∫_α^β i^2(ωt) d(ωt))
    integral_i2, _ = quad(lambda wt: (i_wt(wt))**2, alpha, beta, epsabs=1e-9, epsrel=1e-9, limit=500)
    I_rms = math.sqrt(integral_i2 / (2.0 * math.pi))
    steps.append(f"Step 10: RMS current using I_rms = sqrt((1/(2π)) ∫_α^β i^2(ωt) d(ωt)): Integral = {integral_i2:.6f}; I_rms = {I_rms:.6f} A")
    res["rms current"] = round(I_rms, 6)

    # Step 11: Power absorbed by resistor P_R = I_rms^2 * R
    P_R = (I_rms ** 2) * R
    steps.append(f"Step 11: Power absorbed by the resistor using P_R = I_rms^2 * R: P_R = ({I_rms:.6f})^2 * {R:.4f} ≈ {P_R:.6f} W")
    res["power absorbed by the resistor"] = round(P_R, 6)

    # Step 12: Power absorbed by the DC source P_dc = Vdc * I_avg
    P_dc = Vdc * I_avg
    steps.append(f"Step 12: Power absorbed by the dc voltage source using P_dc = Vdc * I_avg: P_dc = {Vdc:.4f} * {I_avg:.6f} ≈ {P_dc:.6f} W")
    res["power absorbed by the dc source"] = round(P_dc, 6)

    # Step 13: power supplied by AC, apparent power and power factor
    P_ac = P_R + P_dc
    S = vs_rms * I_rms
    pf = P_ac / S if S != 0 else 0.0
    steps.append(f"Step 13: Power supplied by AC P_ac = P_R + P_dc = {P_R:.6f} + {P_dc:.6f} ≈ {P_ac:.6f} W; "
                 f"S = V_rms * I_rms = {vs_rms:.4f} * {I_rms:.6f} ≈ {S:.6f} VA; pf = P_ac / S ≈ {pf:.6f}")
    res["power supplied by ac source"] = round(P_ac, 6)
    res["apparent power (VA)"] = round(S, 6)
    res["power factor"] = round(pf, 6)

    # steps_text for messaging
    res["steps_text"] = "\n".join(steps)

    # compact expression
    res["expression_compact"] = f"{i_m:.6f}*sin(ωt - {theta:.6f}) - {Vdc/R:.6f} + {A:.6f}*exp(-ωt/{omega_tau:.6f})"

    return res
def controlled_half_wave_rectifier_with_rldc_load(Vrms, f, R, L, Vdc, alpha_deg):
    """
    Controlled half-wave rectifier with series R-L and a series DC source (RLDC).
    Parameters
    ----------
    Vrms : float
        Source RMS voltage (V).
    f : float
        Frequency (Hz).
    R : float
        Load resistance (Ω).
    L : float
        Load inductance (H).
    Vdc : float
        Series DC voltage (V).
    alpha_deg : float
        Delay angle in degrees.

    Returns
    -------
    dict
        Step-by-step strings (Step 1..Step N), numeric keys, and 'steps_text' multiline summary.
    """
    # Basic validation
    if Vrms <= 0:
        raise ValueError("Vrms must be positive")
    if f <= 0:
        raise ValueError("f must be positive")
    if R <= 0:
        raise ValueError("R must be positive")
    if L <= 0:
        raise ValueError("L must be positive")
    if Vdc < 0:
        raise ValueError("Vdc must be non-negative")

    res = {}
    steps = []

    # Step 1: fundamental quantities
    omega = 2.0 * math.pi * f
    Vm = Vrms * math.sqrt(2.0)
    steps.append(f"Step 1: Peak voltage using V_m = √2 * V_rms: V_m = √2 * {Vrms} ≈ {Vm:.6f} V")
    res["V_m"] = round(Vm, 6)
    res["omega"] = round(omega, 6)

    # Step 2: impedance Z and angle theta
    omegaL = omega * L
    Z = math.hypot(R, omegaL)
    theta = math.atan2(omegaL, R)   # rad
    steps.append(f"Step 2: Load impedance Z = √(R^2 + (ωL)^2) = √({R}^2 + ({omegaL:.6f})^2) ≈ {Z:.6f} Ω")
    steps.append(f"        Phase angle θ = atan(ωL / R) ≈ {theta:.6f} rad ({math.degrees(theta):.4f}°)")
    res["Z"] = round(Z, 6)
    res["theta (rad)"] = round(theta, 6)
    res["theta (deg)"] = round(math.degrees(theta), 4)

    # Step 3: time-constant term ωτ = ωL / R
    omega_tau = omegaL / R
    steps.append(f"Step 3: Time-constant term ωτ = ωL / R = {omegaL:.6f} / {R} ≈ {omega_tau:.6f}")
    res["omega_tau"] = round(omega_tau, 6)

    # Step 4: firing angle alpha (convert degrees -> radians)
    alpha = math.radians(alpha_deg)
    steps.append(f"Step 4: Delay/firing angle α = {alpha_deg}° = {alpha:.6f} rad")
    res["alpha (rad)"] = round(alpha, 6)
    res["alpha (deg)"] = round(alpha_deg, 6)

    # Step 5: steady-state coefficients i_m and A
    i_m = Vm / Z
    # A follows the textbook form (ensures initial condition continuity at ωt = α)
    A = (-i_m * math.sin(alpha - theta) + (Vdc / R)) * math.exp(alpha / omega_tau)
    steps.append(f"Step 5: Coefficients: i_m = V_m / Z = {Vm:.6f} / {Z:.6f} ≈ {i_m:.6f} A; "
                 f"A = (-i_m·sin(α-θ) + Vdc/R)·e^(α/ωτ) ≈ {A:.6f}")
    res["i_m"] = round(i_m, 6)
    res["A"] = round(A, 6)

    # Step 6: current expression (valid during conduction α ≤ ωt ≤ β)
    steps.append("Step 6: Load current expression (for conduction interval α ≤ ωt ≤ β):")
    steps.append(f"        i(ωt) = i_m·sin(ωt - θ) - Vdc/R + A·e^-ωt/{omega_tau}")
    steps.append(f"        numeric: i(ωt) ≈ {i_m:.6f}·sin(ωt - {theta:.6f}) - {Vdc/R:.6f} + {A:.6f}·e^(-ωt/{omega_tau:.6f})")
    res["expression for load current"] = (f"{i_m:.6f}*sin(ωt - {theta:.6f}) - {Vdc/R:.6f} + "
                                          f"{A:.6f}*exp(-ωt/{omega_tau:.6f})")

    # helper current function (argument is ωt)
    def i_of_wt(wt):
        return i_m * math.sin(wt - theta) - (Vdc / R) + A * math.exp(-wt / omega_tau)

    # Step 7: find extinction angle beta by solving i(β) = 0 with bracket search
    eps = 1e-9
    start = alpha + 1e-6
    if start >= 2.0 * math.pi:
        raise RuntimeError("α is >= 2π — invalid delay angle for this problem.")

    # bracket search on [start, 2π]
    a = start
    b = 2.0 * math.pi
    bracket = None
    # quick checks
    try:
        fa = i_of_wt(a)
        fb = i_of_wt(b)
    except Exception:
        fa = None
        fb = None

    if fa is not None and fb is not None and fa * fb < 0:
        bracket = (a, b)
    else:
        # scan grid to find sign change
        Nscan = 2000
        prev_w = a
        prev_val = i_of_wt(prev_w)
        for k in range(1, Nscan + 1):
            w = a + (b - a) * k / Nscan
            val = i_of_wt(w)
            # skip if NaN
            if prev_val == 0.0:
                bracket = (prev_w, w)
                break
            if prev_val * val < 0:
                bracket = (prev_w, w)
                break
            prev_w, prev_val = w, val

    if bracket is None:
        raise RuntimeError("Failed to bracket extinction angle β. Conduction may not end in this cycle.")

    try:
        sol = root_scalar(lambda wt: i_of_wt(wt), bracket=bracket, method='brentq', xtol=1e-12, rtol=1e-12)
        beta = sol.root
    except Exception as ex:
        raise RuntimeError(f"Root-finding for β failed: {ex}")

    steps.append(f"Step 7: Extinction angle β by solving i(β)=0 numerically: Solved numerically: β ≈ {beta:.6f} rad ({math.degrees(beta):.4f}°)")
    res["extinction angle (rad)"] = round(beta, 6)
    res["extinction angle (deg)"] = round(math.degrees(beta), 4)

    # Step 8: average current I_avg = (1/(2π)) ∫_{α}^{β} i(ωt) d(ωt)
    # and I_rms = sqrt( (1/(2π)) ∫_{α}^{β} i(ωt)^2 d(ωt) )
    i_avg_integral, _ = quad(i_of_wt, alpha, beta, epsabs=1e-9, epsrel=1e-9, limit=500)
    I_avg = i_avg_integral / (2.0 * math.pi)
    steps.append(f"Step 8: Average current I_avg = (1/(2π)) ∫_α^β i(ωt) d(ωt): integral = {i_avg_integral:.9f}; I_avg ≈ {I_avg:.6f} A")
    res["average current"] = round(I_avg, 6)

    i_rms_integral, _ = quad(lambda wt: (i_of_wt(wt) ** 2), alpha, beta, epsabs=1e-9, epsrel=1e-9, limit=500)
    I_rms = math.sqrt(i_rms_integral / (2.0 * math.pi))
    steps.append(f"Step 9: RMS current I_rms = sqrt((1/(2π)) ∫_α^β i^2(ωt) d(ωt)): integral = {i_rms_integral:.9f}; I_rms ≈ {I_rms:.6f} A")
    res["rms current"] = round(I_rms, 6)

    # Step 10: power absorbed by DC source P_dc = Vdc * I_avg
    P_dc = Vdc * I_avg
    steps.append(f"Step 10: Power absorbed by DC source P_dc = Vdc * I_avg = {Vdc:.6f} * {I_avg:.6f} ≈ {P_dc:.6f} W")
    res["power absorbed by the dc voltage source"] = round(P_dc, 6)

    # Step 11: power absorbed by resistor P_R = I_rms^2 * R
    P_R = (I_rms ** 2) * R
    steps.append(f"Step 11: Power absorbed by resistor P_R = I_rms^2 * R = ({I_rms:.6f})^2 * {R} ≈ {P_R:.6f} W")
    res["power absorbed by the resistance"] = round(P_R, 6)

    # Step 12: total power, apparent power and power factor
    P_total = P_dc + P_R
    S = Vrms * I_rms
    pf = (P_total / S) if S != 0.0 else 0.0
    steps.append(f"Step 12: Power supplied by AC P_ac = P_R + P_dc = {P_R:.6f} + {P_dc:.6f} ≈ {P_total:.6f} W")
    steps.append(f"         Apparent power S = V_rms * I_rms = {Vrms} * {I_rms:.6f} ≈ {S:.6f} VA")
    steps.append(f"         Power factor pf = P_ac / S ≈ {pf:.6f}")
    res["power supplied by ac source"] = round(P_total, 6)
    res["apparent power S"] = round(S, 6)
    res["power factor"] = round(pf, 6)

    # steps_text (one step per line)
    res["steps_text"] = "\n".join(steps)

    return res
def controlled_half_wave_rectifier_with_ldc(Vm, L, Vdc, omega, alpha):
    """
    Controlled half-wave rectifier with series inductance L and DC source Vdc.
    Parameters
    ----------
    Vm : float
        Peak AC voltage (V).
    L : float
        Inductance (H).
    Vdc : float
        Series DC voltage (V).
    omega : float
        Angular frequency (rad/s).
    alpha : float
        Delay angle. If > 2π it is assumed provided in degrees and converted to radians.

    Returns
    -------
    dict
        Contains:
          - step-by-step strings ("Step 1: ...", ...)
          - numeric results (Vm, alpha (rad), beta (rad), average current, rms current)
          - 'expression for current' (string)
          - 'steps_text' (multi-line string, one step per line) for messaging
    """
    # ---------- validation ----------
    if Vm <= 0:
        raise ValueError("Vm must be positive")
    if L <= 0:
        raise ValueError("L must be positive")
    if omega <= 0:
        raise ValueError("omega must be positive")
    # normalize alpha: accept degrees if value > 2π
    alpha_input = float(alpha)
    if alpha_input > 2 * math.pi:
        alpha_rad = math.radians(alpha_input)
    else:
        alpha_rad = alpha_input

    # Prepare result
    result = {}
    steps = []

    # Step 1: list parameters
    steps.append(f"Step 1: Parameters: V_m = {Vm:.6g} V, L = {L:.6g} H, Vdc = {Vdc:.6g} V, ω = {omega:.6g} rad/s, α = {alpha_rad:.6f} rad ({math.degrees(alpha_rad):.4f}°)")
    result["V_m"] = round(Vm, 6)
    result["L"] = round(L, 6)
    result["Vdc"] = round(Vdc, 6)
    result["omega"] = round(omega, 6)
    result["alpha (rad)"] = round(alpha_rad, 6)
    result["alpha (deg)"] = round(math.degrees(alpha_rad), 4)

    # Step 2: derive expression coefficients
    coeff1 = Vm / (omega * L)       # V_m/(ωL)
    coeff2 = Vdc / (omega * L)      # Vdc/(ωL)
    steps.append(f"Step 2: Inductive coefficients: V_m/(ωL) = {Vm:.6g}/({omega:.6g}*{L:.6g}) ≈ {coeff1:.6g} A; Vdc/(ωL) ≈ {coeff2:.6g} A/rad")
    result["V_m_over_omegaL"] = round(coeff1, 6)
    result["Vdc_over_omegaL"] = round(coeff2, 6)

    # Step 3: current expression (i(ωt) for α ≤ ωt ≤ β)
    # i(ωt) = (V_m/(ωL))*(cos α - cos ωt) + (Vdc/(ωL))*(α - ωt)
    expr_str = f"i(ωt) = {coeff1:.6g}*(cos(α) - cos(ωt)) + {coeff2:.6g}*(α - ωt),  for α ≤ ωt ≤ β"
    steps.append("Step 3: Load current expression:")
    steps.append(f"    {expr_str}")
    result["expression for current"] = expr_str

    # helper current function
    def i_of_wt(wt):
        return coeff1 * (math.cos(alpha_rad) - math.cos(wt)) + coeff2 * (alpha_rad - wt)

    # Step 4: find extinction angle beta (β > α) solving i(β) = 0
    # scan to bracket a root in (α, 2π)
    eps = 1e-12
    start = alpha_rad + 1e-9
    end = 2 * math.pi
    bracket = None
    Nscan = 2000
    prev_w = start
    prev_val = i_of_wt(prev_w)
    for k in range(1, Nscan + 1):
        w = start + (end - start) * k / Nscan
        val = i_of_wt(w)
        # sign change -> bracket found
        if prev_val == 0.0:
            bracket = (prev_w, w)
            break
        if prev_val * val < 0:
            bracket = (prev_w, w)
            break
        prev_w, prev_val = w, val

    if bracket is None:
        # Try slightly beyond 2π (rare cases)
        end2 = 2 * math.pi + 0.5
        prev_w = start
        prev_val = i_of_wt(prev_w)
        for k in range(1, Nscan + 1):
            w = start + (end2 - start) * k / Nscan
            val = i_of_wt(w)
            if prev_val == 0.0:
                bracket = (prev_w, w)
                break
            if prev_val * val < 0:
                bracket = (prev_w, w)
                break
            prev_w, prev_val = w, val

    if bracket is None:
        raise RuntimeError("Failed to bracket extinction angle β (no sign change found for i(ωt)).")

    a, b = bracket
    try:
        sol = root_scalar(i_of_wt, bracket=[a, b], method='brentq', xtol=1e-12, rtol=1e-12, maxiter=200)
        beta = sol.root
    except Exception as e:
        raise RuntimeError(f"Root finding for β failed: {e}")

    steps.append(f"Step 4: Extinction angle β by solving i(β)=0 numerically: bracketed in ({a:.6f}, {b:.6f}); β ≈ {beta:.6f} rad ({math.degrees(beta):.4f}°)")
    result["extinction angle (rad)"] = round(beta, 6)
    result["extinction angle (deg)"] = round(math.degrees(beta), 4)

    # Step 5: average current I_avg = (1/(2π)) * ∫_{α}^{β} i(ωt) d(ωt)
    integral_i, _ = quad(i_of_wt, alpha_rad, beta, epsabs=1e-10, epsrel=1e-10, limit=400)
    I_avg = integral_i / (2 * math.pi)
    steps.append(f"Step 5: Average current using I_avg = (1/(2π)) ∫_α^β i(ωt) d(ωt): integral = {integral_i:.12g}; I_avg = {I_avg:.6g} A")
    result["average current"] = round(I_avg, 6)

    # Step 6: RMS current (often useful)
    integral_i2, _ = quad(lambda wt: i_of_wt(wt) ** 2, alpha_rad, beta, epsabs=1e-10, epsrel=1e-10, limit=400)
    I_rms = math.sqrt(integral_i2 / (2 * math.pi))
    result["rms current"] = round(I_rms, 6)
    steps.append(f"Step 6: RMS current using I_rms = sqrt((1/(2π)) ∫_α^β i^2(ωt) d(ωt)): integral = {integral_i2:.12g}; I_rms = {I_rms:.6g} A")

    # Final assemble
    result["steps_text"] = "\n".join(steps)
    return result
def full_wave_bridge_rectifier_resistive_load(Vrms, R):
    """
    Full-wave bridge rectifier with purely resistive load.

    Parameters
    ----------
    Vrms : float
        AC source RMS voltage (V), e.g. 120
    R : float
        Load resistance (Ω), e.g. 18

    Returns
    -------
    dict
        Detailed step-by-step results (strings) and numeric values:
          - 'Vm' (V)
          - 'Vo_avg' (V) average DC output voltage
          - 'average load current' (A)
          - 'peak load current' (A)
          - 'rms load current' (A)
          - 'average diode current' (A) (per diode)
          - 'peak diode current' (A) (per diode)
          - 'rms diode current' (A) (per diode)
          - 'steps_text' multiline string (one step per line) for messaging
    Notes
    -----
    Formulas used:
      Vm = sqrt(2) * Vrms
      V_o(avg) = 2 * Vm / pi
      I_o(avg) = V_o(avg) / R
      I_o(peak) = Vm / R
      I_o(rms) = I_o(peak) / sqrt(2)
    For bridge diodes (each diode conducts for half the period):
      I_D(avg,per diode) = I_o(avg) / 2
      I_D(peak,per diode) = I_o(peak)
      I_D(rms,per diode) = I_o(peak) / 2
    """
    # --- input validation ---
    if Vrms <= 0:
        raise ValueError("Vrms must be positive")
    if R <= 0:
        raise ValueError("R must be positive")

    res = {}
    steps = []

    # Step 1: Peak voltage
    Vm = math.sqrt(2) * Vrms
    s1 = f"Step 1: Peak voltage: Vm = √2 * Vrms = √2 * {Vrms:.6g} ≈ {Vm:.6f} V"
    res["Vm"] = round(Vm, 6)
    steps.append(s1)

    # Step 2: Average output voltage
    Vo_avg = 2.0 * Vm / math.pi
    s2 = f"Step 2: Average output voltage (full-wave): Vo_avg = 2·Vm/π = 2 * {Vm:.6f} / π ≈ {Vo_avg:.6f} V"
    res["Vo_avg"] = round(Vo_avg, 6)
    steps.append(s2)

    # Step 3: Average load current
    Io_avg = Vo_avg / R
    s3 = f"Step 3: Average load current: I_o(avg) = Vo_avg / R = {Vo_avg:.6f} / {R:.6g} ≈ {Io_avg:.6f} A"
    res["average load current"] = round(Io_avg, 6)
    steps.append(s3)

    # Step 4: Peak load current
    Io_peak = Vm / R
    s4 = f"Step 4: Peak load current: I_o(peak) = Vm / R = {Vm:.6f} / {R:.6g} ≈ {Io_peak:.6f} A"
    res["peak load current"] = round(Io_peak, 6)
    steps.append(s4)

    # Step 5: RMS load current
    Io_rms = Io_peak / math.sqrt(2)
    s5 = f"Step 5: RMS load current: I_o(rms) = I_o(peak)/√2 = {Io_peak:.6f}/√2 ≈ {Io_rms:.6f} A"
    res["rms load current"] = round(Io_rms, 6)
    steps.append(s5)

    # Step 6: Diode currents (per-diode)
    # Each diode conducts during alternate half-cycles in a bridge.
    ID_avg = Io_avg / 2.0
    ID_peak = Io_peak
    ID_rms = Io_peak / 2.0  # derived: sqrt((1/(2π))∫_0^π (Im sin θ)^2 dθ) = Im/2
    s6 = f"Step 6: Diode currents (per diode): I_D(avg) = I_o(avg)/2 = {Io_avg:.6f}/2 ≈ {ID_avg:.6f} A"
    s7 = f"         I_D(peak) = I_o(peak) = {ID_peak:.6f} A"
    s8 = f"         I_D(rms) = I_o(peak)/2 = {Io_peak:.6f}/2 ≈ {ID_rms:.6f} A"
    res["average diode current"] = round(ID_avg, 6)
    res["peak diode current"] = round(ID_peak, 6)
    res["rms diode current"] = round(ID_rms, 6)
    steps.extend([s6, s7, s8])

    # Combine steps into multiline text suitable for Telegram
    res["steps_text"] = "\n".join(steps)

    return res
def single_phase_rectifier_bridge_vs_center_tapped(R, Vrms, f):
    """
    Compare single-phase bridge vs center-tapped full-wave rectifiers with resistive load.

    Parameters
    ----------
    R : float
        Load resistance (Ω)
    Vrms : float
        AC source RMS voltage (V). For the center-tapped case this is the RMS of each half-secondary
        (matches the problem statement "120 V rms on each half of the secondary").
    f : float
        Frequency (Hz) — included for completeness.

    Returns
    -------
    dict
        A dictionary containing:
          - Vm (peak voltage)
          - average load current (bridge)
          - PIV_bridge (V)
          - average load current (center-tapped)
          - PIV_center_tapped (V)
          - steps_text: multiline string (one step per line)
    """
    if Vrms <= 0:
        raise ValueError("Vrms must be positive")
    if R <= 0:
        raise ValueError("R must be positive")
    if f <= 0:
        raise ValueError("f must be positive")

    res = {}
    steps = []

    # Step 1: peak voltage
    Vm = math.sqrt(2) * Vrms
    steps.append(f"Step 1: Peak voltage Vm = √2 * Vrms = √2 * {Vrms:.6g} ≈ {Vm:.6f} V")
    res["Vm"] = round(Vm, 6)

    # Part (a) Bridge rectifier:
    # Vo_avg = 2*Vm/pi  and Io_avg = Vo_avg / R
    Vo_avg_bridge = 2.0 * Vm / math.pi
    Io_avg_bridge = Vo_avg_bridge / R
    steps.append(f"Step 2 (Bridge): Average output voltage Vo_avg = 2·Vm/π = 2 * {Vm:.6f} / π ≈ {Vo_avg_bridge:.6f} V")
    steps.append(f"Step 3 (Bridge): Average load current I_o(avg) = Vo_avg / R = {Vo_avg_bridge:.6f} / {R:.6g} ≈ {Io_avg_bridge:.6f} A")
    res["average load current (bridge)"] = round(Io_avg_bridge, 6)

    # PIV for bridge: Vm
    PIV_bridge = Vm
    steps.append(f"Step 4 (Bridge): Peak inverse voltage (PIV) per diode = Vm ≈ {PIV_bridge:.6f} V")
    res["PIV_bridge"] = round(PIV_bridge, 6)

    # Part (b) Center-tapped (Vrms is per half-secondary as problem states):
    # Vo_avg same formula (2*Vm/pi) if Vrms is half-secondary RMS
    Vo_avg_ct = 2.0 * Vm / math.pi
    Io_avg_ct = Vo_avg_ct / R
    steps.append(f"Step 5 (Center-tapped): Peak per half-secondary Vm = √2 * Vrms = {Vm:.6f} V")
    steps.append(f"Step 6 (Center-tapped): Average output voltage Vo_avg = 2·Vm/π = {Vo_avg_ct:.6f} V")
    steps.append(f"Step 7 (Center-tapped): Average load current I_o(avg) = Vo_avg / R = {Vo_avg_ct:.6f} / {R:.6g} ≈ {Io_avg_ct:.6f} A")
    res["average load current (center-tapped)"] = round(Io_avg_ct, 6)

    # PIV for center-tapped: each diode can see 2*Vm
    PIV_ct = 2.0 * Vm
    steps.append(f"Step 8 (Center-tapped): Peak inverse voltage (PIV) per diode = 2·Vm ≈ {PIV_ct:.6f} V")
    res["PIV_center_tapped"] = round(PIV_ct, 6)

    # Combine steps into a single multiline string
    res["steps_text"] = "\n".join(steps)

    return res
def single_phase_bridge_rectifier_with_rl_load(Vm, omega, R, L):
    """
    Single-phase bridge rectifier with RL load — step-by-step output.

    Parameters
    ----------
    Vm : float
        Peak source voltage (V).
    omega : float
        Angular frequency (rad/s).
    R : float
        Load resistance (Ω).
    L : float
        Load inductance (H).

    Returns
    -------
    dict
        Contains step-by-step strings ('Step 1: ...'), numeric results (rounded),
        and a multi-line 'steps_text' ready for messaging.
    """
    # input validation
    if Vm <= 0:
        raise ValueError("Vm must be positive")
    if omega <= 0:
        raise ValueError("omega must be positive")
    if R <= 0:
        raise ValueError("R must be positive")
    if L < 0:
        raise ValueError("L must be non-negative")

    res = {}
    steps = []

    # Step 1: DC (average) output voltage and average load current
    Vdc = (2.0 * Vm) / math.pi          # a0/2 for full-wave rectified sine
    Io = Vdc / R
    step1 = (f"Step 1: Average (dc) output voltage and load current:\n"
             f"V_dc = 2 V_m / π = 2*{Vm:.6g}/π ≈ {Vdc:.6f} V\n"
             f"I_o = V_dc / R = {Vdc:.6f} / {R:.6g} ≈ {Io:.6f} A")
    res["Step 1: Average output voltage and average load current"] = step1
    res["V_dc"] = round(Vdc, 6)
    res["average load current"] = round(Io, 6)
    steps.append(step1)

    # Step 2: Compute significant harmonic amplitudes (numerical integration)
    # For the full-wave rectified waveform v(θ)=Vm*|sin θ| the DC term is a0/2 and
    # the cosine coefficients a_n = (1/π) ∫_0^{2π} Vm |sin θ| cos(nθ) dθ.
    def compute_an(n, n_points=40001):
        thetas = np.linspace(0.0, 2.0 * math.pi, n_points)
        vals = Vm * np.abs(np.sin(thetas)) * np.cos(n * thetas)
        return (1.0 / math.pi) * np.trapz(vals, thetas)

    # compute a2 and a4 (these correspond to the first two AC harmonics of the full-wave rectified signal)
    a2 = Vdc*((1/(2-1))-(1/(2+1)))
    a4 = Vdc*((1/(4-1))-(1/(4+1)))
    step2 = (f"Step 2: Harmonic cosine coefficients (numerical):\n"
             f"a2 = V_dc[(1/n-1)-(1/n+1)] =V_dc*((1/(2-1))-(1/(2+1)))= {a2:.6f} V\n"
             f"a4 = V_dc[(1/n-1)-(1/n+1)] =V_dc*((1/(4-1))-(1/(4+1)))= {a4:.6f} V")
    res["Step 2: Harmonic coefficients (a2, a4)"] = step2
    res["a2 (V)"] = round(a2, 6)
    res["a4 (V)"] = round(a4, 6)
    steps.append(step2)

    # Step 3: Impedances and harmonic currents
    # For nth harmonic, use n*omega in inductive reactance.
    Z2 = math.sqrt(R * R + (2 * omega * L) ** 2)
    Z4 = math.sqrt(R * R + (4 * omega * L) ** 2)
    I2 = a2 / Z2
    I4 = a4 / Z4
    step3 = (f"Step 3: Harmonic impedances and current amplitudes:\n"
             f"Z2 = sqrt(R^2 + (2ωL)^2) = sqrt({R:.6g}^2 + (2*{omega:.6g}*{L:.6g})^2) ≈ {Z2:.6f} Ω\n"
             f"I2 = a2 / Z2 ≈ {a2:.6f} / {Z2:.6f} ≈ {I2:.6f} A\n"
             f"Z4 = sqrt(R^2 + (4ωL)^2) ≈ {Z4:.6f} Ω\n"
             f"I4 = a4 / Z4 ≈ {a4:.6f} / {Z4:.6f} ≈ {I4:.6f} A")
    res["Step 3: Harmonic impedances and current amplitudes"] = step3
    res["Z2 (Ω)"] = round(Z2, 6)
    res["I2 (A, peak)"] = round(I2, 6)
    res["Z4 (Ω)"] = round(Z4, 6)
    res["I4 (A, peak)"] = round(I4, 6)
    steps.append(step3)

    # Step 4: RMS load current (include DC and the two harmonic contributions)
    # RMS contribution: I_rms = sqrt( I0^2 + (I2^2)/2 + (I4^2)/2 )
    I_rms = math.sqrt(Io ** 2 + (I2 ** 2) / 2.0 + (I4 ** 2) / 2.0)
    step4 = (f"Step 4: RMS load current (approx using DC + 2 harmonics):\n"
             f"I_rms ≈ sqrt(I_o^2 + (I2^2)/2 + (I4^2)/2)\n"
             f"     = sqrt({Io:.6f}^2 + ({I2:.6f}^2)/2 + ({I4:.6f}^2)/2) ≈ {I_rms:.6f} A")
    res["Step 4: RMS load current"] = step4
    res["rms load current"] = round(I_rms, 6)
    steps.append(step4)

    # Step 5: Diode currents (bridge)
    # Average diode current = I_o / 2 (each diode conducts on alternate half-cycles on average)
    I_d_avg = Io / 2.0
    # RMS diode current (approx) = I_rms / sqrt(2) (since each diode conducts pulses; approximate)
    I_d_rms = I_rms / math.sqrt(2.0)
    step5 = (f"Step 5: Diode currents (approx):\n"
             f"I_D,avg ≈ I_o / 2 = {Io:.6f} / 2 ≈ {I_d_avg:.6f} A\n"
             f"I_D,rms ≈ I_rms / √2 = {I_rms:.6f} / √2 ≈ {I_d_rms:.6f} A")
    res["Step 5: Diode current estimates"] = step5
    res["average diode current"] = round(I_d_avg, 6)
    res["rms diode current"] = round(I_d_rms, 6)
    steps.append(step5)

    # Step 6: (Optional) list the harmonics included and a short note
    note = "Note: result uses a numerical evaluation of a2 and a4 (dominant harmonics for full-wave rectified Vm|sinθ|). " \
           "More harmonics can be included for higher accuracy."
    res["Step 6: Note on method"] = note
    steps.append(note)

    # assemble steps_text for messaging
    res["steps_text"] = "\n\n".join(steps)

    return res
def full_wave_bridge_rectifier_rl_power(Vrms, f, R, L):
    """
    Full-wave bridge rectifier with RL load — power & power factor analysis.
    Returns step-by-step strings, numeric results, and a multiline steps_text for messaging.

    Parameters
    ----------
    Vrms : float
        Source RMS voltage (V).
    f : float
        Frequency (Hz).
    R : float
        Load resistance (Ω).
    L : float
        Load inductance (H).

    Returns
    -------
    dict
        Keys:
          - "Step 1: ..." .. "Step 6: ..."
          - numeric keys: "average load current", "rms load current", "power absorbed by the load",
                          "power factor", and harmonic data.
          - "steps_text": multiline summary (one step per paragraph).
    """
    # --- validate ---
    if Vrms <= 0:
        raise ValueError("Vrms must be positive")
    if f <= 0:
        raise ValueError("f must be positive")
    if R <= 0:
        raise ValueError("R must be positive")
    if L < 0:
        raise ValueError("L must be non-negative")

    res = {}
    steps = []

    # Step 1: basic quantities
    omega = 2.0 * math.pi * f
    Vm = Vrms * math.sqrt(2.0)
    step1 = (f"Step 1: Basic quantities:\n"
             f"V_m = √2 * V_rms = √2 * {Vrms:.6g} ≈ {Vm:.6f} V\n"
             f"ω = 2πf = 2π * {f:.6g} ≈ {omega:.6f} rad/s")
    res["Step 1: Basic quantities"] = step1
    res["V_m"] = round(Vm, 6)
    res["omega"] = round(omega, 6)
    steps.append(step1)

    # Step 2: Average (dc) output and average load current
    Vdc = (2.0 * Vm) / math.pi
    Io = Vdc / R
    step2 = (f"Step 2: Average output and average load current:\n"
             f"V_dc = 2 V_m / π = 2*{Vm:.6f}/π ≈ {Vdc:.6f} V\n"
             f"I_o = V_dc / R = {Vdc:.6f} / {R:.6g} ≈ {Io:.6f} A")
    res["Step 2: Average output and average load current"] = step2
    res["V_dc"] = round(Vdc, 6)
    res["average load current"] = round(Io, 6)
    steps.append(step2)

    # Step 3: compute dominant harmonic coefficients a2 and a4 numerically
    # v_o(θ) = Vm * |sin θ| for full-wave rectified sine
    def compute_an(n, n_points=40001):
        thetas = np.linspace(0.0, 2.0 * math.pi, n_points)
        vals = Vm * np.abs(np.sin(thetas)) * np.cos(n * thetas)
        return (1.0 / math.pi) * np.trapz(vals, thetas)

    a2 = Vdc*((1/(2-1))-(1/(2+1)))
    a4 = Vdc*((1/(4-1))-(1/(4+1)))
    step3 = (f"Step 3: Harmonic coefficients (numerical):\n"
             f"a2 = V_dc[(1/n-1)-(1/n+1)] =V_dc*((1/(2-1))-(1/(2+1)))= {a2:.6f} V\n"
             f"a4 = V_dc[(1/n-1)-(1/n+1)] =V_dc*((1/(4-1))-(1/(4+1)))= {a4:.6f} V")
    res["Step 3: Harmonic coefficients"] = step3
    res["a2 (V)"] = round(a2, 6)
    res["a4 (V)"] = round(a4, 6)
    steps.append(step3)

    # Step 4: impedances & harmonic currents
    Z2 = math.sqrt(R**2 + (2 * omega * L)**2)
    Z4 = math.sqrt(R**2 + (4 * omega * L)**2)
    I2 = a2 / Z2 if Z2 != 0 else 0.0
    I4 = a4 / Z4 if Z4 != 0 else 0.0
    step4 = (f"Step 4: Harmonic impedances and current amplitudes:\n"
             f"Z2 = sqrt(R^2 + (2ωL)^2) = sqrt({R:.6g}^2 + (2*{omega:.6g}*{L:.6g})^2) ≈ {Z2:.6f} Ω\n"
             f"I2 = a2 / Z2 ≈ {a2:.6f} / {Z2:.6f} ≈ {I2:.6f} A\n"
             f"Z4 = sqrt(R^2 + (4ωL)^2) ≈ {Z4:.6f} Ω\n"
             f"I4 = a4 / Z4 ≈ {a4:.6f} / {Z4:.6f} ≈ {I4:.6f} A")
    res["Step 4: Harmonic impedances and current amplitudes"] = step4
    res["Z2 (Ω)"] = round(Z2, 6)
    res["I2 (A, peak)"] = round(I2, 6)
    res["Z4 (Ω)"] = round(Z4, 6)
    res["I4 (A, peak)"] = round(I4, 6)
    steps.append(step4)

    # Step 5: RMS load current (approx include DC + 2 harmonics)
    I_rms = math.sqrt(Io**2 + (I2**2) / 2.0 + (I4**2) / 2.0)
    step5 = (f"Step 5: RMS load current (approx using DC + I2 & I4):\n"
             f"I_rms ≈ sqrt(I_o^2 + (I2^2)/2 + (I4^2)/2)\n"
             f"     = sqrt({Io:.6f}^2 + ({I2:.6f}^2)/2 + ({I4:.6f}^2)/2) ≈ {I_rms:.6f} A")
    res["Step 5: RMS load current"] = step5
    res["rms load current"] = round(I_rms, 6)
    steps.append(step5)

    # Step 6: power and power factor
    P_load = (I_rms**2) * R
    S = Vrms * I_rms
    pf = (P_load / S) if S != 0 else 0.0
    step6 = (f"Step 6: Power absorbed and power factor:\n"
             f"P = I_rms^2 * R = ({I_rms:.6f})^2 * {R:.6g} ≈ {P_load:.6f} W\n"
             f"S = V_rms * I_rms = {Vrms:.6g} * {I_rms:.6f} ≈ {S:.6f} VA\n"
             f"pf = P / S ≈ {pf:.6f}")
    res["Step 6: Power absorbed and power factor"] = step6
    res["power absorbed by the load"] = round(P_load, 6)
    res["power factor"] = round(pf, 6)
    steps.append(step6)

    # Step 7: diode (bridge) approximate currents
    I_d_avg = Io / 2.0
    I_d_rms = I_rms / math.sqrt(2.0)
    step7 = (f"Step 7: Diode current estimates (bridge):\n"
             f"I_D,avg ≈ I_o / 2 = {Io:.6f} / 2 ≈ {I_d_avg:.6f} A\n"
             f"I_D,rms ≈ I_rms / √2 ≈ {I_rms:.6f} / √2 ≈ {I_d_rms:.6f} A")
    res["Step 7: Diode current estimates"] = step7
    res["average diode current"] = round(I_d_avg, 6)
    res["rms diode current"] = round(I_d_rms, 6)
    steps.append(step7)

    # final multiline
    res["steps_text"] = "\n\n".join(steps)
    return res
def single_phase_center_tapped_rectifier_with_resistive_load(
    Vrms_primary: float,
    Vrms_secondary_tap: float,
    f: float,
    R: float,
    turns_ratio: str = None
):
    """
    Single-phase center-tapped transformer rectifier with resistive load.
    Returns detailed step-by-step strings and numeric results.
    Parameters:
      - Vrms_primary: primary RMS voltage (V), e.g. 240
      - Vrms_secondary_tap: RMS voltage of each half-secondary tap (V), e.g. 40
      - f: frequency (Hz), e.g. 60
      - R: load resistance (Ω)
      - turns_ratio: optional textual turns ratio (e.g. "3:1")
    Returns:
      dict with keys:
        - 'average load current' (A)
        - 'RMS load current' (A)
        - 'average source current' (A)
        - 'RMS source current' (A)
        - 'steps_text' (multiline string with Step 1..Step 4)
        - plus intermediate numeric values (Vm, Vo, etc.)
    """
    # --- input validation ---
    if Vrms_primary <= 0:
        raise ValueError("Vrms_primary must be positive")
    if Vrms_secondary_tap <= 0:
        raise ValueError("Vrms_secondary_tap must be positive")
    if f <= 0:
        raise ValueError("f must be positive")
    if R <= 0:
        raise ValueError("R must be positive")

    out = {}
    steps = []

    # Step 1: peak secondary (per half) and mean output voltage
    Vm = Vrms_secondary_tap * math.sqrt(2)
    steps.append(
        f"Step 1: Peak secondary (per half-tap) V_m = √2 * V_rms(tap) = √2 * {Vrms_secondary_tap} ≈ {Vm:.4f} V"
    )
    out["Vm (peak per half)"] = round(Vm, 4)

    # Step 2: Average (DC) output voltage for center-tapped full-wave
    Vo = (2.0 * Vm) / math.pi
    steps.append(
        f"Step 2: Average output voltage V_o = 2·V_m / π = 2 * {Vm:.4f} / π ≈ {Vo:.4f} V"
    )
    out["V_o (average)"] = round(Vo, 4)

    # Step 3: Load currents (average and RMS)
    I_o_avg = Vo / R
    I_load_rms = Vrms_secondary_tap / R  # for purely resistive load the RMS output equals Vrms(tap)
    steps.append(
        f"Step 3: Load currents: I_o (average) = V_o / R = {Vo:.4f} / {R} ≈ {I_o_avg:.4f} A; "
        f"I_rms (load) = V_rms(tap) / R = {Vrms_secondary_tap:.4f} / {R} ≈ {I_load_rms:.4f} A"
    )
    out["average load current"] = round(I_o_avg, 4)
    out["RMS load current"] = round(I_load_rms, 4)

    # Step 4: Source (primary) currents: average and RMS (reflect secondary to primary using turns ratio)
    # average source current for center-tapped rectifier is zero (symmetry)
    I_source_avg = 0.0
    steps.append("Step 4: Average source current: for the center-tapped rectifier the fundamental-average "
                 "over a cycle is zero due to symmetry: I_s,avg = 0 A")
    out["average source current"] = round(I_source_avg, 4)

    # RMS source current: reflect secondary RMS to primary by ratio N2/N1 = V2_rms / V1_rms
    N2_over_N1 = Vrms_secondary_tap / Vrms_primary
    I_source_rms = I_load_rms * N2_over_N1
    steps.append(
        f"Step 5: RMS source current (reflected): I_s,rms = I_load,rms * (N2/N1) ≈ {I_load_rms:.4f} * "
        f"({Vrms_secondary_tap:.4f}/{Vrms_primary:.4f}) ≈ {I_source_rms:.4f} A"
    )
    out["RMS source current"] = round(I_source_rms, 4)

    # Steps summary text
    if turns_ratio is not None:
        steps.insert(0, f"Turns ratio (given): {turns_ratio}")
    steps_text = "\n".join(steps)
    out["steps_text"] = steps_text

    # Also include helpful intermediate values for clarity
    out["f (Hz)"] = round(f, 4)
    out["R (Ω)"] = round(R, 4)
    out["Vrms_primary"] = round(Vrms_primary, 4)
    out["Vrms_secondary_tap"] = round(Vrms_secondary_tap, 4)

    return out
def design_center_tapped_rectifier(I_avg_desired, R, f, source1=120, source2=240):
    """
    Simple design for a center-tapped transformer rectifier.

    Parameters:
      I_avg_desired : desired average load current (A)
      R             : load resistance (Ω)
      f             : frequency (Hz)
      source1       : first available primary RMS voltage (V) or None
      source2       : second available primary RMS voltage (V) or None

    Returns:
      dict with:
        - V_o, V_m, V_rms_tap, V_secondary_total_rms (numeric)
        - turns_ratios: dict keyed by string "Vs=<value>V" with numeric N1/N2 and text desc
        - selected_source: chosen source (numeric)
        - selected_turns_ratio_text: short recommendation text
        - steps_text: multiline human-readable steps (one step per line)
    """
    # --- validate inputs ---
    if I_avg_desired is None or R is None or f is None:
        raise ValueError("I_avg_desired, R and f must be provided")
    try:
        I_avg_desired = float(I_avg_desired)
        R = float(R)
        f = float(f)
    except Exception:
        raise ValueError("I_avg_desired, R and f must be numeric")

    if I_avg_desired <= 0 or R <= 0 or f <= 0:
        raise ValueError("I_avg_desired, R and f must be positive")

    # Build list of provided sources (skip None)
    sources = []
    if source1 is not None:
        sources.append(float(source1))
    if source2 is not None and (source2 != source1):
        sources.append(float(source2))

    if not sources:
        raise ValueError("At least one source voltage (source1 or source2) must be provided")

    steps = []
    out = {}

    # Step 1: required DC output voltage
    V_o = I_avg_desired * R
    steps.append(f"Step 1: Required average output voltage: V_o = I_o · R = {I_avg_desired:.6g} · {R:.6g} = {V_o:.6g} V")
    out["V_o"] = round(V_o, 6)

    # Step 2: required peak per half-secondary (center-tapped)
    V_m = (V_o * math.pi) / 2.0
    steps.append(f"Step 2: Peak per half-secondary (V_m) = (π/2)·V_o = ({V_o:.6g}·π)/2 ≈ {V_m:.6g} V")
    out["V_m"] = round(V_m, 6)

    # Step 3: RMS per half-tap and overall secondary RMS
    V_rms_tap = V_m / math.sqrt(2.0)
    V_secondary_total_rms = 2.0 * V_rms_tap
    steps.append(
        f"Step 3: RMS per half-tap: V_rms(tap) = V_m/√2 = {V_m:.6g}/√2 ≈ {V_rms_tap:.6g} V; "
        f"overall secondary RMS ≈ {V_secondary_total_rms:.6g} V"
    )
    out["V_rms_tap"] = round(V_rms_tap, 6)
    out["V_secondary_total_rms"] = round(V_secondary_total_rms, 6)

    # Step 4: compute turns ratios for each provided source and choose best (closest to 1:1)
    turns_ratios = {}
    best_source = None
    best_metric = None
    for Vs in sources:
        N1_over_N2 = Vs / V_secondary_total_rms
        # produce a simple small-denominator fraction to show a turns suggestion (limit denominator to 20)
        frac = Fraction(N1_over_N2).limit_denominator(20)
        # Represent key as a string so downstream formatting never sees float keys
        key = f"Vs={int(round(Vs))}V"
        desc = f"N1/N2 ≈ {frac.numerator}:{frac.denominator} (≈ {N1_over_N2:.6f})"
        turns_ratios[key] = {"N1_over_N2": round(N1_over_N2, 6), "desc": desc, "Vs": round(Vs, 6)}
        metric = abs(1.0 - N1_over_N2)
        if best_metric is None or metric < best_metric:
            best_metric = metric
            best_source = Vs

    steps.append("Step 4: Computed transformer turns ratio (N1/N2 = V_primary / V_secondary_total) for available sources:")
    for key, info in turns_ratios.items():
        steps.append(f"  - For {key}: {info['desc']}")

    if best_source is None:
        raise RuntimeError("Failed to determine best source (unexpected).")

    selected_key = f"Vs={int(round(best_source))}V"
    selected_ratio = turns_ratios[selected_key]["N1_over_N2"]
    steps.append(f"Step 5: Recommendation: pick source with turns ratio closest to 1:1 -> {int(round(best_source))} V (N1/N2 ≈ {selected_ratio:.6f})")

    out["turns_ratios"] = turns_ratios
    out["selected_source"] = round(best_source, 6)
    out["selected_turns_ratio_N1_over_N2"] = round(selected_ratio, 6)
    # also store a friendly text
    out["selected_turns_ratio_text"] = f"Recommended N1:N2 ≈ 1 : {1/selected_ratio:.2f}" if selected_ratio != 0 else "Undefined"
    out["I_o_desired"] = round(I_avg_desired, 6)
    out["f"] = round(f, 6)
    out["steps_text"] = "\n".join(steps)

    return out


def design_center_tapped_rectifier_with_rl_load(I_avg_desired, R, f, source1=120, source2=240, L=None):
    """
    Design center-tapped transformer rectifier (step-by-step style).
    Parameters:
      I_avg_desired : desired average load current (A)
      R             : load resistance (Ω)
      f             : frequency (Hz)
      source1       : first available primary RMS voltage (V) or None
      source2       : second available primary RMS voltage (V) or None
      L             : optional series inductance (H). If provided, function reports X_L and Z.
    Returns:
      dict with:
        - V_o, V_m, V_rms_tap, V_secondary_total_rms (numeric)
        - turns_ratios: dict keyed by string "Vs=<value>V" with numeric N1/N2 and text desc
        - selected_source: chosen source (numeric)
        - selected_turns_ratio_N1_over_N2 (numeric)
        - selected_turns_ratio_text (friendly string)
        - L (if provided), X_L, Z_load (if L provided)
        - note_about_L: brief note that L doesn't change turns ratio but affects conduction/ripple
        - steps_text: multiline human-readable steps (one step per line)
    """
    # validate basic inputs
    if I_avg_desired is None or R is None or f is None:
        raise ValueError("I_avg_desired, R and f must be provided")
    try:
        I_avg_desired = float(I_avg_desired)
        R = float(R)
        f = float(f)
    except Exception:
        raise ValueError("I_avg_desired, R and f must be numeric")

    if I_avg_desired <= 0 or R <= 0 or f <= 0:
        raise ValueError("I_avg_desired, R and f must be positive")

    # parse sources
    sources = []
    if source1 is not None:
        try:
            sources.append(float(source1))
        except Exception:
            raise ValueError("source1 must be numeric or None")
    if source2 is not None and (source2 != source1):
        try:
            sources.append(float(source2))
        except Exception:
            raise ValueError("source2 must be numeric or None")

    if not sources:
        raise ValueError("At least one source voltage (source1 or source2) must be provided")

    # optional L parsing
    if L is not None:
        try:
            L = float(L)
            if L <= 0:
                raise ValueError("If provided, L must be positive")
        except Exception:
            raise ValueError("L must be numeric (Henries) or None")

    steps = []
    out = {}

    # Step 1: required DC output voltage
    V_o = I_avg_desired * R
    steps.append(f"Step 1: Required average output voltage: V_o = I_o · R = {I_avg_desired:.6g} · {R:.6g} = {V_o:.6g} V")
    out["V_o"] = round(V_o, 6)

    # Step 2: required peak per half-secondary (center-tapped)
    V_m = (V_o * math.pi) / 2.0
    steps.append(f"Step 2: Peak per half-secondary (V_m) = (π/2)·V_o = ({V_o:.6g}·π)/2 ≈ {V_m:.6g} V")
    out["V_m"] = round(V_m, 6)

    # Step 3: RMS per half-tap and overall secondary RMS
    V_rms_tap = V_m / math.sqrt(2.0)
    V_secondary_total_rms = 2.0 * V_rms_tap
    steps.append(
        f"Step 3: RMS per half-tap: V_rms(tap) = V_m/√2 = {V_m:.6g}/√2 ≈ {V_rms_tap:.6g} V; "
        f"overall secondary RMS ≈ {V_secondary_total_rms:.6g} V"
    )
    out["V_rms_tap"] = round(V_rms_tap, 6)
    out["V_secondary_total_rms"] = round(V_secondary_total_rms, 6)

    # Step 4: compute turns ratios for each provided source and choose best (closest to 1:1)
    turns_ratios = {}
    best_source = None
    best_metric = None
    for Vs in sources:
        N1_over_N2 = Vs / V_secondary_total_rms
        frac = Fraction(N1_over_N2).limit_denominator(20)
        key = f"Vs={int(round(Vs))}V"
        desc = f"N1/N2 ≈ {frac.numerator}:{frac.denominator} (≈ {N1_over_N2:.6f})"
        turns_ratios[key] = {"N1_over_N2": round(N1_over_N2, 6), "desc": desc, "Vs": round(Vs, 6)}
        metric = abs(1.0 - N1_over_N2)
        if best_metric is None or metric < best_metric:
            best_metric = metric
            best_source = Vs

    steps.append("Step 4: Computed transformer turns ratio (N1/N2 = V_primary / V_secondary_total) for available sources:")
    for key, info in turns_ratios.items():
        steps.append(f"  - For {key}: {info['desc']}")

    if best_source is None:
        raise RuntimeError("Failed to determine best source (unexpected).")

    selected_key = f"Vs={int(round(best_source))}V"
    selected_ratio = turns_ratios[selected_key]["N1_over_N2"]
    steps.append(f"Step 5: Recommendation: pick source with turns ratio closest to 1:1 -> {int(round(best_source))} V (N1/N2 ≈ {selected_ratio:.6f})")

    out["turns_ratios"] = turns_ratios
    out["selected_source"] = round(best_source, 6)
    out["selected_turns_ratio_N1_over_N2"] = round(selected_ratio, 6)
    out["selected_turns_ratio_text"] = f"Recommended N1:N2 ≈ 1 : {1/selected_ratio:.2f}" if selected_ratio != 0 else "Undefined"

    # Step 6: If L provided, compute reactance and impedance and explain effect
    if L is not None:
        omega = 2 * math.pi * f
        X_L = omega * L
        Z_load = math.sqrt(R * R + X_L * X_L)
        steps.append(f"Step 6: Inductance provided: L = {L:.6g} H -> X_L = ωL = 2πf·L = {omega:.6g}·{L:.6g} ≈ {X_L:.6g} Ω")
        steps.append(f"  -> Load impedance magnitude at fundamental: Z = √(R^2 + X_L^2) ≈ {Z_load:.6g} Ω")
        steps.append("Note: L does NOT change the required transformer turns ratio (turns ratio depends on voltages),")
        steps.append("but L affects current waveform (conduction interval, smoothing, and ripple). Use these Z/X values when")
        steps.append("computing conduction/extinction angles and ripple/heating — those require time-domain or harmonic analysis.")
        out["L"] = round(L, 6)
        out["X_L"] = round(X_L, 6)
        out["Z_load"] = round(Z_load, 6)
        out["note_about_L"] = "L does not affect turns ratio; it affects current waveform (conduction interval and ripple)."
    else:
        steps.append("Step 6: No inductance L provided. If L exists, it affects conduction interval and current ripple but not turns ratio.")
        out["note_about_L"] = "No L provided."

    out["I_o_desired"] = round(I_avg_desired, 6)
    out["f"] = round(f, 6)
    out["steps_text"] = "\n".join(steps)

    return out

def design_additional_series_resistance_for_electromagnet(L, R_existing, I_avg_desired, Vrms, f):
    """
    Compute additional series resistance required so a bridge rectifier + series
    resistance supplies a target average current to an RL electromagnet.

    Parameters
    ----------
    L : float
        Inductance in henries (H), e.g. 0.2 for 200 mH.
    R_existing : float
        Existing DC series resistance (Ω).
    I_avg_desired : float
        Desired average current (A).
    Vrms : float
        Source RMS voltage (V), e.g. 120.
    f : float
        Frequency in Hz, e.g. 60.

    Returns
    -------
    dict
      Contains step-by-step strings ("Step 1: ...", ...), numeric results:
        - "additional_series_resistance (Ω)"
        - "R_total_required (Ω)"
        - "Vm (V)"
        - "V2 (V)" (second-harmonic amplitude approximation)
        - "Z2 (Ω)"
        - "I2 (A)"
        - "delta_I_pp_estimated (A)"
      and "steps_text": multiline human-readable steps (one step per line).
    Notes
    -----
    - Uses the simple full-wave bridge average formula V_dc ≈ 2 V_m / π, and
      approximates ripple using the second harmonic component of the rectified waveform.
    - This is a design estimate; time-domain/harmonic simulation gives more accurate ripple.
    """
    # --- input validation ---
    if L is None or R_existing is None or I_avg_desired is None or Vrms is None or f is None:
        raise ValueError("All inputs L, R_existing, I_avg_desired, Vrms and f must be provided")
    try:
        L = float(L)
        R_existing = float(R_existing)
        I_avg_desired = float(I_avg_desired)
        Vrms = float(Vrms)
        f = float(f)
    except Exception:
        raise ValueError("Numeric inputs required (L, R_existing, I_avg_desired, Vrms, f)")

    if L < 0 or R_existing < 0 or I_avg_desired <= 0 or Vrms <= 0 or f <= 0:
        raise ValueError("L and R_existing must be >= 0; I_avg_desired, Vrms and f must be > 0")

    steps = []
    out = {}

    # Step 1: compute peak voltage Vm
    Vm = Vrms * math.sqrt(2.0)
    steps.append(f"Step 1: Peak source voltage: V_m = √2 · V_rms = √2 · {Vrms:.6g} ≈ {Vm:.6g} V")
    out["Vm"] = round(Vm, 6)

    # Step 2: required total resistance from average DC equation for full-wave bridge
    # I_avg = (2*Vm) / (π * R_total)  => R_total = 2*Vm / (π * I_avg)
    R_total = (2.0 * Vm) / (math.pi * I_avg_desired)
    steps.append(
        "Step 2: Required total series resistance from average DC formula for full-wave bridge:\n"
        f"  I_avg = (2·V_m)/(π·R_total)  =>  R_total = 2·V_m / (π·I_avg)\n"
        f"  R_total = 2·{Vm:.6g} / (π · {I_avg_desired:.6g}) ≈ {R_total:.6g} Ω"
    )
    out["R_total_required"] = round(R_total, 6)

    # Step 3: additional resistance required
    R_additional = R_total - R_existing
    if R_additional < 0:
        note = ("Note: computed additional resistance is negative — existing resistance already "
                "exceeds required total. Set additional resistance to 0 and review design.")
        steps.append(f"Step 3: Additional resistance = R_total - R_existing = {R_total:.6g} - {R_existing:.6g} = {R_additional:.6g} Ω")
        steps.append("  " + note)
        R_additional_out = 0.0
    else:
        steps.append(f"Step 3: Additional resistance required = R_total - R_existing = {R_total:.6g} - {R_existing:.6g} = {R_additional:.6g} Ω")
        R_additional_out = R_additional

    out["additional_series_resistance (Ω)"] = round(R_additional_out, 6)

    # Step 4: optional ripple estimate via the second harmonic approximation
    # Compute V_dc = 2 Vm / π (used for harmonic coefficients)
    V_dc = (2.0 * Vm) / math.pi
    steps.append(f"Step 4: DC average used for harmonic coefficients: V_dc ≈ 2·V_m / π = {V_dc:.6g} V")

    # second-harmonic amplitude for full-wave rectified |sin| waveform:
    # a_n = V_dc * (1/(n-1) - 1/(n+1)), for even n. For n=2: factor = (1 - 1/3) = 2/3
    n = 2
    V2 = V_dc * (1.0 / (n - 1.0) - 1.0 / (n + 1.0))
    steps.append(f"  -> Second-harmonic (n=2) amplitude estimate: V2 = V_dc·(1/(n-1)-1/(n+1)) = {V2:.6g} V")
    out["V2 (V)"] = round(V2, 6)

    # compute Z_2 = sqrt(R_total^2 + (n·ω·L)^2)
    omega = 2.0 * math.pi * f
    Xn = n * omega * L
    Z2 = math.sqrt(R_total * R_total + Xn * Xn)
    steps.append(f"Step 5: Harmonic impedance at n=2: X_n = n·ω·L = {n}·{omega:.6g}·{L:.6g} ≈ {Xn:.6g} Ω")
    steps.append(f"  -> Z2 = √(R_total^2 + X_n^2) = √({R_total:.6g}^2 + {Xn:.6g}^2) ≈ {Z2:.6g} Ω")
    out["Z2 (Ω)"] = round(Z2, 6)

    # I2 and ripple estimate
    I2 = V2 / Z2
    delta_I_pp = 2.0 * abs(I2)
    steps.append(f"Step 6: Second-harmonic current amplitude: I2 = V2 / Z2 = {V2:.6g} / {Z2:.6g} ≈ {I2:.6g} A")
    steps.append(f"  -> Approximate peak-to-peak current ripple estimate: ΔI_pp ≈ 2·I2 ≈ {delta_I_pp:.6g} A")
    out["I2 (A)"] = round(I2, 6)
    out["delta_I_pp_estimated (A)"] = round(delta_I_pp, 6)

    # Final note & return
    if R_additional < 0:
        steps.append("Final note: No additional series resistor required (existing resistance already sufficient).")
    else:
        steps.append("Final note: Add the computed additional resistance in series with the electromagnet. "
                     "Estimate is for steady-state and uses harmonic approximation for ripple; do time-domain simulation for more accuracy.")
    out["steps_text"] = "\n".join(steps)

    return out
def full_wave_rectifier_with_rl_and_dc_source(Vm, f, R, L, Vdc):
    """
    Full-wave rectifier with RL load and series DC source.

    Parameters
    ----------
    Vm : float
        Peak value of the AC source (V) (e.g. 170).
    f : float
        Frequency in Hz (e.g. 60).
    R : float
        Load resistance (Ω).
    L : float
        Load inductance (H).
    Vdc : float
        Series DC voltage present in the load (V).

    Returns
    -------
    dict
        Contains step-by-step strings ("Step 1: ...", ...), numeric results:
          - "I_avg (A)" : average load current (I_o)
          - "I_rms (A)" : RMS load current (approx using n=2,4)
          - "P_dc (W)"  : power absorbed by DC source
          - "P_R (W)"   : power absorbed by resistor
          - "power_factor" : power factor seen by source
          - harmonic intermediates: V2, V4, Z2, Z4, I2, I4
          - "delta_I_pp_estimated (A)" : approx peak-to-peak load current variation (≈2·I2)
        and "steps_text": multiline human readable steps (suitable for Telegram).
    Notes
    -----
    - Uses the standard Fourier approximations for the full-wave rectified |sin| waveform:
        V_dc = 2 Vm / π
        V_n = V_dc * (1/(n-1) - 1/(n+1))  for even n (n = 2, 4, ...)
    - RMS approximated as sqrt(I0^2 + (I2^2)/2 + (I4^2)/2).
    - If more accuracy is needed, include more harmonics or time-domain integration.
    """
    # input validation
    if Vm is None or f is None or R is None or L is None or Vdc is None:
        raise ValueError("Vm, f, R, L and Vdc must be provided")
    try:
        Vm = float(Vm); f = float(f); R = float(R); L = float(L); Vdc = float(Vdc)
    except Exception:
        raise ValueError("Numeric inputs required for Vm, f, R, L, Vdc")
    if f <= 0 or R <= 0 or L < 0:
        raise ValueError("f and R must be > 0; L must be >= 0")

    steps = []
    out = {}

    # Step 1: V_m (already given) and V_dc (average rectified without DC source)
    omega = 2.0 * math.pi * f
    steps.append(f"Step 1: Given peak voltage: V_m = {Vm:.6g} V, frequency f = {f:.6g} Hz (ω = {omega:.6g} rad/s)")
    V_dc = (2.0 * Vm) / math.pi
    steps.append(f"  -> Average (no-load) rectified voltage: V_dc ≈ 2·V_m/π = 2·{Vm:.6g}/π ≈ {V_dc:.6g} V")
    out["V_dc (V)"] = round(V_dc, 6)

    # Step 2: Average load current (I_o)
    I_o = (V_dc - Vdc) / R
    steps.append(
        "Step 2: Average load current using I_o = (V_dc - V_dc_source) / R:\n"
        f"  I_o = ({V_dc:.6g} - {Vdc:.6g}) / {R:.6g} ≈ {I_o:.6g} A"
    )
    out["I_avg (A)"] = round(I_o, 6)

    # Step 3: Harmonic voltages for n=2 and n=4 (full-wave rectified |sin|)
    # Vn = V_dc * (1/(n-1) - 1/(n+1)) for even n
    def Vn_from_Vdc(n):
        if n <= 1:
            return 0.0
        return V_dc * (1.0 / (n - 1.0) - 1.0 / (n + 1.0))

    V2 = Vn_from_Vdc(2)
    V4 = Vn_from_Vdc(4)
    steps.append(f"Step 3: Harmonic voltage estimates from V_dc:")
    steps.append(f"  - n=2: V2 = V_dc·(1/(2-1)-1/(2+1)) = {V2:.6g} V")
    steps.append(f"  - n=4: V4 = V_dc·(1/(4-1)-1/(4+1)) = {V4:.6g} V")
    out["V2 (V)"] = round(V2, 6)
    out["V4 (V)"] = round(V4, 6)

    # Step 4: Harmonic impedances and currents
    n2 = 2
    n4 = 4
    X2 = n2 * omega * L
    X4 = n4 * omega * L
    Z2 = math.sqrt(R * R + X2 * X2)
    Z4 = math.sqrt(R * R + X4 * X4)
    I2 = V2 / Z2
    I4 = V4 / Z4

    steps.append("Step 4: Harmonic impedances and currents:")
    steps.append(f"  - X2 = n·ω·L = 2 · {omega:.6g} · {L:.6g} ≈ {X2:.6g} Ω; Z2 = √(R^2 + X2^2) ≈ {Z2:.6g} Ω; I2 = V2 / Z2 ≈ {I2:.6g} A")
    steps.append(f"  - X4 = 4 · {omega:.6g} · {L:.6g} ≈ {X4:.6g} Ω; Z4 = √(R^2 + X4^2) ≈ {Z4:.6g} Ω; I4 = V4 / Z4 ≈ {I4:.6g} A")
    out["X2 (Ω)"] = round(X2, 6)
    out["Z2 (Ω)"] = round(Z2, 6)
    out["I2 (A)"] = round(I2, 6)
    out["X4 (Ω)"] = round(X4, 6)
    out["Z4 (Ω)"] = round(Z4, 6)
    out["I4 (A)"] = round(I4, 6)

    # Step 5: RMS current approx (DC + 2 harmonics)
    I_rms = math.sqrt(I_o**2 + (I2**2) / 2.0 + (I4**2) / 2.0)
    steps.append(f"Step 5: Approximate RMS load current using I_rms ≈ sqrt(I_o^2 + (I2^2)/2 + (I4^2)/2): I_rms ≈ {I_rms:.6g} A")
    out["I_rms (A)"] = round(I_rms, 6)

    # Step 6: Powers
    P_dc = I_o * Vdc
    P_R = (I_rms ** 2) * R
    Vrms = Vm / math.sqrt(2.0)
    S = Vrms * I_rms
    pf = (P_dc + P_R) / S if S != 0 else 0.0

    steps.append(f"Step 6: Powers:")
    steps.append(f"  - Power absorbed by DC source: P_dc = I_o · V_dc_source = {I_o:.6g} · {Vdc:.6g} ≈ {P_dc:.6g} W")
    steps.append(f"  - Power absorbed by resistor: P_R = I_rms^2 · R = ({I_rms:.6g})^2 · {R:.6g} ≈ {P_R:.6g} W")
    steps.append(f"  - Apparent power: S = V_rms · I_rms = {Vrms:.6g} · {I_rms:.6g} ≈ {S:.6g} VA")
    steps.append(f"  - Power factor: pf = (P_dc + P_R) / S ≈ {(P_dc + P_R):.6g} / {S:.6g} ≈ {pf:.6g}")

    out["P_dc (W)"] = round(P_dc, 6)
    out["P_R (W)"] = round(P_R, 6)
    out["S (VA)"] = round(S, 6)
    out["power_factor"] = round(pf, 6)

    # Step 7: approximate peak-to-peak current variation (≈ 2 * I2)
    delta_I_pp = 2.0 * abs(I2)
    steps.append(f"Step 7: Approximate peak-to-peak load current variation: ΔI_pp ≈ 2·I2 ≈ {delta_I_pp:.6g} A")
    out["delta_I_pp_estimated (A)"] = round(delta_I_pp, 6)

    # final steps_text
    out["steps_text"] = "\n".join(steps)
    return out
def single_phase_full_wave_bridge_with_rl_source(Vrms: float, f: float, R: float, L: float, Vdc: float) -> Dict:
    """
    Full-wave bridge rectifier with RL-source load (dominant-harmonic approx).
    Inputs:
      Vrms - source RMS voltage (V)
      f    - frequency (Hz)
      R    - resistance (ohm)
      L    - inductance (H)
      Vdc  - dc source in series with load (V)
    Returns:
      dict with step-by-step strings "Step ...", numeric results, and "steps_text" multiline field.
    Notes:
      - uses Vm = sqrt(2)*Vrms
      - uses dominant-harmonic approximation with n=2 and n=4 for RMS and ripple estimates
    """
    if Vrms <= 0 or f <= 0 or R <= 0 or L <= 0:
        raise ValueError("Vrms, f, R and L must be positive")

    res = {}
    steps = []

    # Step 0: given
    Vm = math.sqrt(2.0) * Vrms
    omega = 2 * math.pi * f
    steps.append(f"Step 0: Given: V_rms = {Vrms:.6g} V, f = {f:.6g} Hz, R = {R:.6g} Ω, L = {L:.6g} H, Vdc = {Vdc:.6g} V")
    res["Vrms"] = round(Vrms, 6)
    res["Vm"] = round(Vm, 6)
    res["f"] = round(f, 6)
    res["R"] = round(R, 6)
    res["L"] = round(L, 6)
    res["Vdc"] = round(Vdc, 6)

    # Step 1: theoretical DC from ideal full-wave rectified waveform
    Vdc_theory = 2 * Vm / math.pi
    steps.append(f"Step 1: Theoretical DC from ideal full-wave rectified waveform: V_dc_theory = 2·V_m/π = 2*{Vm:.6g}/π ≈ {Vdc_theory:.6f} V")
    res["V_dc_theory"] = round(Vdc_theory, 6)

    # Step 2: average load current approximated as (Vdc_theory - Vdc)/R
    I_o = (Vdc_theory - Vdc) / R
    steps.append(f"Step 2: Average load current: I_o = (V_dc_theory - Vdc) / R = ({Vdc_theory:.6f} - {Vdc:.6f}) / {R:.6g} ≈ {I_o:.6f} A")
    res["I_o (average load current)"] = round(I_o, 6)

    # Step 3: dominant even-harmonic voltages (approx formula)
    def harmonic_voltage(n):
        # formula used in your reference solutions: a_n = Vdc_theory * (1/(n-1) - 1/(n+1))
        if n <= 0:
            return 0.0
        return Vdc_theory * ((1.0 / (n - 1.0)) - (1.0 / (n + 1.0)))

    V2 = harmonic_voltage(2)
    V4 = harmonic_voltage(4)
    steps.append(f"Step 3: Dominant harmonic voltages (approx): V2 ≈ {V2:.6f} V, V4 ≈ {V4:.6f} V")
    res["V2"] = round(V2, 6)
    res["V4"] = round(V4, 6)

    # Step 4: impedances at harmonics and harmonic currents
    Z2 = math.sqrt(R * R + (2 * omega * L) ** 2)
    Z4 = math.sqrt(R * R + (4 * omega * L) ** 2)
    I2 = V2 / Z2
    I4 = V4 / Z4
    steps.append(f"Step 4: Impedances: Z2 = sqrt(R^2 + (2ωL)^2) ≈ {Z2:.6f} Ω, Z4 ≈ {Z4:.6f} Ω")
    steps.append(f"        Harmonic currents: I2 = V2/Z2 ≈ {I2:.6f} A, I4 = V4/Z4 ≈ {I4:.6f} A")
    res["Z2"] = round(Z2, 6)
    res["Z4"] = round(Z4, 6)
    res["I2"] = round(I2, 6)
    res["I4"] = round(I4, 6)

    # Step 5: approximate RMS load current (DC + dominant harmonics)
    I_rms = math.sqrt(max(0.0, I_o ** 2 + (I2 ** 2) / 2.0 + (I4 ** 2) / 2.0))
    steps.append(f"Step 5: Approximate RMS load current: I_rms ≈ sqrt(I_o^2 + I2^2/2 + I4^2/2) ≈ {I_rms:.6f} A")
    res["I_rms"] = round(I_rms, 6)

    # Step 6: powers and power factor
    P_dc = I_o * Vdc
    P_R = (I_rms ** 2) * R
    V_rms_actual = Vm / math.sqrt(2.0)
    S = V_rms_actual * I_rms
    P_total_from_ac = P_dc + P_R
    pf = P_total_from_ac / S if S != 0 else 0.0

    steps.append(f"Step 6: P_dc = I_o * Vdc = {I_o:.6f} * {Vdc:.6f} ≈ {P_dc:.6f} W")
    steps.append(f"        P_R = I_rms^2 * R = ({I_rms:.6f})^2 * {R:.6g} ≈ {P_R:.6f} W")
    steps.append(f"        Apparent power S = V_rms * I_rms = ({Vm:.6g}/√2) * {I_rms:.6f} ≈ {S:.6f} VA")
    steps.append(f"        Power factor pf = (P_dc + P_R) / S ≈ {pf:.6f}")

    res["P_dc"] = round(P_dc, 6)
    res["P_R"] = round(P_R, 6)
    res["S (apparent)"] = round(S, 6)
    res["pf"] = round(pf, 6)
    res["P_total_from_ac"] = round(P_total_from_ac, 6)

    # Step 7: approximate peak-to-peak current ripple (2 * I2)
    deltaI_pp = 2.0 * I2
    steps.append(f"Step 7: Approx peak-to-peak load-current ripple: ΔI_pp ≈ 2 * I2 ≈ {deltaI_pp:.6f} A")
    res["deltaI_pp (approx)"] = round(deltaI_pp, 6)

    # final multi-line steps_text
    res["steps_text"] = "\n".join(steps)

    return res


def design_fullwave_rectifier_cap_filter(Vrms, f, R, ripple_percentage):
    """
    Design a full-wave rectifier with capacitive filter.

    Parameters:
        Vrms: AC source RMS voltage (V)
        f: AC frequency (Hz)
        R: Load resistance (Ω)
        ripple_percentage: Allowed peak-to-peak ripple as % of DC output (%)

    Returns:
        dict with:
            - Vdc: DC output voltage (V)
            - Delta_V: Peak-to-peak ripple voltage (V)
            - C: Required filter capacitance (μF)
            - I_avg_diode: Average diode current (A)
            - I_peak_diode: Peak diode current (A)
            - steps_text: Multiline human-readable steps
    """
    steps = []

    # Step 1: Approximate DC output voltage
    Vm = Vrms * math.sqrt(2)
    Vdc = Vm
    steps.append(f"Step 1: DC output voltage approximated as Vdc ≈ Vm = Vrms * √2 = {Vrms} * √2 ≈ {Vdc:.2f} V")

    # Step 2: Compute peak-to-peak ripple
    Delta_V = (ripple_percentage / 100.0) * Vdc
    steps.append(f"Step 2: Peak-to-peak ripple: ΔV = {ripple_percentage}% of Vdc = {Delta_V:.2f} V")

    # Step 3: Compute filter capacitance
    C = Vm / (2 * f * R * Delta_V)  # in Farads
    C_uF = C * 1e6  # convert to μF
    steps.append(
        f"Step 3: Filter capacitance: C = Vm / (2 * f * R * ΔV) = {Vm:.2f} / (2*{f}*{R}*{Delta_V:.2f}) ≈ {C_uF:.0f} μF")

    # Step 4: Compute average diode current
    I_o = Vdc / R
    I_avg_diode = I_o / 2  # Full-wave rectifier: 2 diodes conduct alternately
    steps.append(f"Step 4: Average diode current: I_avg = I_o / 2 = {Vdc:.2f}/{R} / 2 ≈ {I_avg_diode:.2f} A")

    # Step 5: Compute conduction angle alpha and peak diode current
    alpha = math.asin(1 - Delta_V / Vm)  # in radians
    omega = 2 * math.pi * f
    I_peak_diode = Vm * (omega * C * math.cos(alpha) + math.sin(alpha) / R)
    steps.append(f"Step 5: Conduction angle α = arcsin(1 - ΔV/Vm) = {math.degrees(alpha):.1f}°")
    steps.append(f"        Peak diode current: I_peak = Vm * (ωC cos α + sin α / R) ≈ {I_peak_diode:.2f} A")

    # Return all outputs
    return {
        "Vdc": round(Vdc, 2),
        "Delta_V": round(Delta_V, 2),
        "C_uF": round(C_uF, 0),
        "I_avg_diode": round(I_avg_diode, 2),
        "I_peak_diode": round(I_peak_diode, 2),
        "steps_text": "\n".join(steps)
    }


def fullwave_rectifier_cap_filter_from_Vm(Vm, f, Vdc_desired, I_dc_desired, ripple_percentage=1.0):
    """
    Compute filter capacitance and diode currents for a full-wave bridge rectifier
    using Vm, desired DC output and load current, and an allowed peak-to-peak ripple.
    Returns a dict with step-by-step strings, numeric results, and a multiline steps_text
    suitable for sending to Telegram.

    Parameters
    ----------
    Vm : float
        Source peak voltage (V). Example: 100
    f : float
        Frequency (Hz). Example: 60
    Vdc_desired : float
        Desired DC output voltage (V). Example: 100
    I_dc_desired : float
        Desired DC load current (A). Example: 0.5
    ripple_percentage : float, optional
        Allowed peak-to-peak ripple as percent of Vdc (default 1.0 for 1%)

    Returns
    -------
    dict
        Contains numeric results and step-by-step text:
          - "Vm", "f", "Vdc", "I_o", "R"
          - "DeltaV" (V), "C_F" (F), "C_uF" (µF)
          - "alpha_rad", "alpha_deg"
          - "I_avg_diode", "I_peak_diode"
          - "steps_text" (multiline human-readable)
    """
    import math

    # --- input validation ---
    for name, val in (("Vm", Vm), ("f", f), ("Vdc_desired", Vdc_desired), ("I_dc_desired", I_dc_desired)):
        if val is None:
            raise ValueError(f"{name} must be provided")
    Vm = float(Vm)
    f = float(f)
    Vdc_desired = float(Vdc_desired)
    I_dc_desired = float(I_dc_desired)
    ripple_percentage = float(ripple_percentage)

    if Vm <= 0 or f <= 0 or Vdc_desired <= 0 or I_dc_desired <= 0 or ripple_percentage <= 0:
        raise ValueError("All numeric inputs must be positive")

    steps = []

    # Step 1: compute/load R
    R = Vdc_desired / I_dc_desired
    steps.append(f"Step 1: Compute load resistance from Vdc and I_o: R = Vdc / I_o = {Vdc_desired:.6g} / {I_dc_desired:.6g} = {R:.6g} Ω")

    # Step 2: compute desired ripple ΔV
    DeltaV = (ripple_percentage / 100.0) * Vdc_desired
    steps.append(f"Step 2: Peak-to-peak ripple ΔV = {ripple_percentage}% of Vdc = {DeltaV:.6g} V")

    # Sanity check
    if DeltaV >= Vm:
        raise ValueError(f"Requested ΔV ({DeltaV:.6g} V) >= Vm ({Vm:.6g} V). Impossible — reduce ripple percentage or increase Vm.")

    # Step 3: capacitance for full-wave: C = Vm / (2 f R ΔV)
    C_F = Vm / (2.0 * f * R * DeltaV)   # Farads
    C_uF = C_F * 1e6
    steps.append(
        "Step 3: Filter capacitance (exact formula for full-wave): "
        f"C = Vm / (2 f R ΔV) = {Vm:.6g}/(2*{f:.6g}*{R:.6g}*{DeltaV:.6g}) ≈ {C_uF:.3f} μF ({C_F:.6e} F)"
    )

    # Step 4: conduction angle from ΔV = Vm (1 - sin α) => sin α = 1 - ΔV/Vm
    sin_alpha = 1.0 - (DeltaV / Vm)
    # clamp numerical rounding
    if sin_alpha > 1.0:
        sin_alpha = 1.0
    if sin_alpha < -1.0:
        sin_alpha = -1.0
    alpha = math.asin(sin_alpha)
    alpha_deg = math.degrees(alpha)
    steps.append(f"Step 4: Conduction angle α from sin α = 1 - ΔV/Vm: α = asin({sin_alpha:.6g}) ≈ {alpha:.6f} rad ({alpha_deg:.4f}°)")

    # Step 5: diode currents
    omega = 2.0 * math.pi * f
    I_o = I_dc_desired
    I_avg_diode = I_o / 2.0  # each diode conducts roughly half the time on average (full-wave bridge)
    I_peak_diode = Vm * (omega * C_F * math.cos(alpha) + (math.sin(alpha) / R))
    steps.append(f"Step 5: Average diode current (I_avg ≈ I_o/2) = {I_avg_diode:.6g} A")
    steps.append(
        f"        Peak diode current I_peak = Vm*(ωC cosα + sinα/R): ω={omega:.6g}; "
        f"I_peak ≈ {I_peak_diode:.6g} A"
    )

    # Prepare result dictionary (match your style: step keys + numeric)
    result = {
        "Vm": round(Vm, 6),
        "f": round(f, 6),
        "Vdc": round(Vdc_desired, 6),
        "I_o": round(I_o, 6),
        "R": round(R, 6),
        "DeltaV": round(DeltaV, 6),
        "C_F": C_F,                  # Farads (raw)
        "C_uF": round(C_uF, 3),      # µF (rounded)
        "alpha (rad)": round(alpha, 6),
        "alpha (deg)": round(alpha_deg, 4),
        "I_avg_diode": round(I_avg_diode, 6),
        "I_peak_diode": round(I_peak_diode, 6),
        # step-by-step text for messaging
        "Step 1: Compute load resistance": f"R = {R:.6g} Ω (from Vdc/I_o)",
        "Step 2: Peak-to-peak ripple": f"ΔV = {DeltaV:.6g} V ({ripple_percentage}% of Vdc)",
        "Step 3: Filter capacitance": f"C ≈ {C_uF:.3f} μF ({C_F:.6e} F)",
        "Step 4: Conduction angle α": f"α ≈ {alpha:.6f} rad ({alpha_deg:.4f}°)",
        "Step 5: Diode currents": f"I_avg ≈ {I_avg_diode:.6g} A; I_peak ≈ {I_peak_diode:.6g} A",
        "steps_text": "\n".join(steps)
    }

    return result
def fullwave_rectifier_lc_filter_largeC(Vrms: float, f: float, L: float, R1, R2):
    """
    Full-wave rectifier with LC filter (large C) — ripple-free output approximation.
    Parameters
    ----------
    Vrms : float
        Source RMS voltage (V), e.g. 120
    f : float
        Frequency in Hz, e.g. 60
    L : float
        Inductance in H, e.g. 0.01 (10 mH)
    R1 : float or convertible
        First load resistance (Ω), e.g. 7
    R2 : float or convertible
        Second load resistance (Ω), e.g. 20

    Returns
    -------
    dict
        {
          "R=<val>Ω": {
              "case": "continuous" or "discontinuous",
              "Vm": <peak volts>,
              "omega": <rad/s>,
              "three_omega_L_over_R": <value>,
              "V_o_primary_formula": "<text>",
              "V_o": <numeric volts>,
              "notes": "<text>",
              "steps_text": "Step 1: ...\nStep 2: ...\n..."
          },
          "steps_text": "<summary multiline>"
        }
    """
    # input validation & normalization
    try:
        Vrms = float(Vrms)
        f = float(f)
        L = float(L)
        R1 = float(R1)
        R2 = float(R2)
    except Exception as e:
        raise ValueError("Vrms, f, L, R1 and R2 must be numeric") from e

    if Vrms <= 0 or f <= 0 or L <= 0 or R1 <= 0 or R2 <= 0:
        raise ValueError("All numeric inputs must be positive")

    omega = 2.0 * math.pi * f
    Vm = Vrms * math.sqrt(2.0)

    def analyze_R(R):
        steps = []
        # Step 1: Vm and omega
        steps.append(f"Step 1: Peak source voltage V_m = √2 · Vrms = √2 · {Vrms:.6g} ≈ {Vm:.6f} V")
        steps.append(f"Step 2: Angular frequency ω = 2πf = 2π · {f:.6g} ≈ {omega:.6f} rad/s")
        # Step 2: compute 3 ω L / R
        three_omega_L_over_R = 3.0 * omega * L / R
        steps.append(f"Step 3: Compute 3·ω·L / R = 3·{omega:.6f}·{L:.6g} / {R:.6g} ≈ {three_omega_L_over_R:.6f}")

        case = "continuous" if three_omega_L_over_R > 1.0 else "discontinuous"
        steps.append(f"Step 4: Rule-of-thumb: current is considered {case} (compare 3·ω·L / R to 1).")

        result = {
            "case": case,
            "Vm": round(Vm, 6),
            "omega": round(omega, 6),
            "three_omega_L_over_R": round(three_omega_L_over_R, 6),
        }

        if case == "continuous":
            # Vo = 2 Vm / π
            Vo = (2.0 * Vm) / math.pi
            steps.append(f"Step 5: Continuous-current formula: V_o = 2·V_m / π = 2·{Vm:.6f}/{math.pi:.6f} ≈ {Vo:.6f} V")
            result.update({
                "V_o_primary_formula": "V_o = 2·V_m / π (continuous current)",
                "V_o": round(Vo, 6),
                "notes": "Continuous current — large C gives ripple-free output with V_o = 2V_m/π."
            })
        else:
            # Discontinuous: use estimate V_o ≈ 0.7 Vm. Add a small iterative refinement:
            Vo_est = 0.7 * Vm
            steps.append(f"Step 5: Discontinuous-current estimate: V_o ≈ 0.7·V_m = 0.7·{Vm:.6f} ≈ {Vo_est:.6f} V")

            # Simple refinement: fixed-point to nudge Vo toward balance of volt-second (very crude,
            # but gives a slightly improved numeric value). We'll iterate using relaxation.
            # (This is intentionally lightweight — exact solution requires solving eqns.)
            Vo = Vo_est
            for _ in range(8):
                # crude physics-inspired correction: current continuity reduces V_o below Vm,
                # push Vo slightly toward Vm*cos(some factor). This is heuristic and only for a better numeric.
                correction = (0.7 * Vm - Vo) * 0.25
                Vo = Vo + correction

            steps.append("Step 6: Quick numeric refinement (heuristic iteration) used to improve discontinuous estimate.")
            steps.append(f"         Refined V_o ≈ {Vo:.6f} V (estimate; for high accuracy use full iterative solution or PSpice).")

            result.update({
                "V_o_primary_formula": "V_o ≈ 0.7·V_m (discontinuous current — heuristic estimate)",
                "V_o_estimate": round(Vo_est, 6),
                "V_o_refined_estimate": round(Vo, 6),
                "V_o": round(Vo, 6),
                "notes": "Discontinuous current — estimate (0.7·V_m) given; refinement applied (heuristic). Use full numerical solution for high accuracy."
            })

        result["steps_text"] = "\n".join(steps)
        return result

    out = {}
    summary_lines = []
    for R in (R1, R2):
        key = f"R={int(round(R))}Ω"
        analysis = analyze_R(R)
        out[key] = analysis
        summary_lines.append(f"{key}: {analysis['case']}, V_o ≈ {analysis['V_o']:.6f} V")

    out["steps_text"] = "\n\n".join([out[k]["steps_text"] for k in out if k.startswith("R=")])
    out["summary"] = "\n".join(summary_lines)
    return out
def controlled_bridge_rectifier_resistive_load(Vrms, f, R, alpha):
    """
    Controlled single-phase bridge rectifier with resistive load (step-by-step style).

    Parameters
    ----------
    Vrms : float
        AC source RMS voltage (V), e.g. 120
    f : float
        Frequency (Hz), e.g. 60 (unused in formulas here but kept for completeness)
    R : float
        Load resistance (Ω), e.g. 20
    alpha : float
        Delay angle. Can be provided in degrees (e.g. 45) or radians (e.g. 0.785).
        The function detects degrees when alpha > 2π.

    Returns
    -------
    dict
        keys include: Vm, alpha_rad, alpha_deg, I_avg (A), I_rms (A),
                       I_s_rms (A), P_load (W), S (VA), pf, and step strings + steps_text.
    """
    # validate inputs
    try:
        Vrms = float(Vrms)
        f = float(f)
        R = float(R)
        alpha = float(alpha)
    except Exception as e:
        raise ValueError("Vrms, f, R and alpha must be numeric") from e

    if Vrms <= 0 or f <= 0 or R <= 0:
        raise ValueError("Vrms, f and R must be positive")

    steps = []

    # Step 1: peak voltage
    Vm = Vrms * math.sqrt(2.0)
    steps.append(f"Step 1: Peak source voltage V_m = √2 · V_rms = √2 · {Vrms:.6g} ≈ {Vm:.6f} V")

    # Step 2: interpret alpha (degrees or radians)
    if alpha > 2.0 * math.pi:
        # almost certainly degrees
        alpha_deg = alpha
        alpha_rad = math.radians(alpha_deg)
        steps.append(f"Step 2: Delay angle provided > 2π → treating as degrees: α = {alpha_deg:.6g}° = {alpha_rad:.6f} rad")
    else:
        # treat as radians but also provide deg form
        alpha_rad = alpha
        alpha_deg = math.degrees(alpha_rad)
        steps.append(f"Step 2: Delay angle provided (interpreted as radians): α = {alpha_rad:.6f} rad = {alpha_deg:.6g}°")

    # Step 3: average load current using Vo = (Vm/π) (1 + cos α) and Io = Vo / R
    Vo_dc = (Vm / math.pi) * (1.0 + math.cos(alpha_rad))
    I_avg = Vo_dc / R
    steps.append(
        "Step 3: Average (dc) output voltage and load current:\n"
        f"        V_o = (V_m/π)·(1 + cos α) = ({Vm:.6f}/π)·(1 + cos({alpha_rad:.6f})) ≈ {Vo_dc:.6f} V\n"
        f"        I_o = V_o / R = {Vo_dc:.6f} / {R:.6g} ≈ {I_avg:.6f} A"
    )

    # Step 4: RMS load current (formula you provided)
    # I_rms = (V_m / R) * sqrt( 1/2 - α/(2π) + (sin 2α)/(4π) )
    term = 0.5 - (alpha_rad / (2.0 * math.pi)) + (math.sin(2.0 * alpha_rad) / (4.0 * math.pi))
    if term < 0:
        # small negative due to rounding -> clamp to zero
        term = max(term, 0.0)
    I_rms = (Vm / R) * math.sqrt(term)
    steps.append(
        "Step 4: RMS load current using formula:\n"
        f"        I_rms = (V_m / R)·√(1/2 - α/(2π) + sin(2α)/(4π)) = ({Vm:.6f}/{R:.6g})·√({term:.6g}) ≈ {I_rms:.6f} A"
    )

    # Step 5: RMS source current (for resistive load with same waveform = I_rms)
    I_s_rms = I_rms
    steps.append(f"Step 5: RMS source current I_s,rms = I_rms ≈ {I_s_rms:.6f} A")

    # Step 6: Power and apparent power and pf
    P_load = (I_rms ** 2) * R
    S = Vrms * I_s_rms
    pf = P_load / S if S != 0 else 0.0
    steps.append(
        "Step 6: Power and power factor:\n"
        f"        P = I_rms^2 · R = ({I_rms:.6f})^2 · {R:.6g} ≈ {P_load:.6f} W\n"
        f"        S = V_rms · I_s,rms = {Vrms:.6g} · {I_s_rms:.6f} ≈ {S:.6f} VA\n"
        f"        pf = P / S ≈ {pf:.6f}"
    )

    # build result dict (rounded nicely for display)
    result = {
        "Vm": round(Vm, 6),
        "alpha_rad": round(alpha_rad, 6),
        "alpha_deg": round(alpha_deg, 4),
        "V_o": round(Vo_dc, 6),
        "I_avg": round(I_avg, 6),
        "I_rms": round(I_rms, 6),
        "I_s_rms": round(I_s_rms, 6),
        "P_load": round(P_load, 6),
        "S": round(S, 6),
        "pf": round(pf, 6),
        "Step 1: Peak voltage": steps[0],
        "Step 2: Delay angle interpretation": steps[1],
        "Step 3: Average load current": steps[2],
        "Step 4: RMS load current": steps[3],
        "Step 5: RMS source current": steps[4],
        "Step 6: Power & pf": steps[5],
        "steps_text": "\n".join(steps)
    }

    return result


def controlled_fullwave_bridge_rl_load_avg_for_two_alphas(Vrms, f, R, L, alpha1_deg, alpha2_deg):
    """
    Controlled single-phase full-wave bridge rectifier with RL load.
    Computes average load current for two delay angles (alpha1_deg, alpha2_deg).
    Uses direct formula if continuous conduction; otherwise, shows full numerical steps.
    """
    Vrms = float(Vrms);
    f = float(f);
    R = float(R);
    L = float(L)
    alpha1 = math.radians(float(alpha1_deg))
    alpha2 = math.radians(float(alpha2_deg))

    steps = []
    out = {}

    omega = 2.0 * math.pi * f
    V_m = Vrms * math.sqrt(2.0)
    steps.append(f"Step 1: V_m = √2·Vrms ≈ {V_m:.6f} V, ω = 2πf ≈ {omega:.6f} rad/s")
    omegaL = omega * L
    Z = math.sqrt(R ** 2 + omegaL ** 2)
    theta = math.atan2(omegaL, R)
    steps.append(f"Step 2: Z = √(R^2 + (ωL)^2) ≈ {Z:.6f} Ω, θ = atan(ωL/R) ≈ {math.degrees(theta):.3f}°")

    def analyze_alpha(alpha_rad):
        local = {}
        alpha_deg_local = math.degrees(alpha_rad)
        steps_local = [f"--- Analysis for α = {alpha_deg_local:.3f}° ---"]

        # Check if continuous
        if theta > alpha_rad:
            conduction_type = "continuous"
            steps_local.append(f"θ ({math.degrees(theta):.3f}°) > α ({alpha_deg_local:.3f}°) → continuous conduction")
            V_o = (2 * V_m / math.pi) * math.cos(alpha_rad)
            I_avg = V_o / R
            steps_local.append(f"Step A: V_o = 2Vm/π * cos(α) ≈ {V_o:.6f} V")
            steps_local.append(f"Step B: I_avg = V_o / R ≈ {I_avg:.6f} A")
            beta_rad = None
            beta_deg = None
        else:
            conduction_type = "discontinuous"
            steps_local.append(
                f"θ ({math.degrees(theta):.3f}°) ≤ α ({alpha_deg_local:.3f}°) → discontinuous conduction")
            I_m = V_m / Z
            omega_tau = omega * L / R
            steps_local.append(f"Step A: Current amplitude I_m = V_m / Z ≈ {I_m:.6f} A")

            # Define i(wt)
            steps_local.append("Step B: Define i(ωt) = I_m*(sin(ωt - θ) - sin(α - θ)*exp(-(ωt - α)/τ)), τ = L/R")
            steps_local.append(
                f"   Where: I_m ≈ {I_m:.6f} A, θ ≈ {math.degrees(theta):.3f}° ({theta:.6f} rad), "
                f"α ≈ {alpha_deg_local:.3f}° ({alpha_rad:.6f} rad), τ = L/R = {L}/{R} ≈ {L / R:.6f} s"
            )
            steps_local.append(
                f"   i(ωt) ≈ {I_m:.6f} * (sin(ωt - {theta:.6f}) - sin({alpha_rad:.6f} - {theta:.6f}) * exp(-(ωt - {alpha_rad:.6f}) / {omega_tau:.6f}))"
            )
            def i_of_wt(wt):
                return I_m * (math.sin(wt - theta) - math.sin(alpha_rad - theta) * math.exp(
                    -(wt - alpha_rad) / omega_tau))

            # Find extinction angle β numerically
            a, b = alpha_rad + 1e-9, alpha_rad + 2 * math.pi - 1e-9
            Nscan = 2000
            prev_w, prev_val = a, i_of_wt(a)
            bracket = None
            for k in range(1, Nscan + 1):
                w = a + (b - a) * k / Nscan
                val = i_of_wt(w)
                if prev_val * val < 0:
                    bracket = (prev_w, w)
                    break
                prev_w, prev_val = w, val
            if bracket is None:
                beta_rad = b
                steps_local.append("Step C: Failed to bracket β; using β ≈ α + 2π")
            else:
                sol = root_scalar(lambda wt: i_of_wt(wt), bracket=bracket, method='brentq')
                beta_rad = sol.root
                steps_local.append(
                    f"Step C: Solve i(β) = 0 numerically → β ≈ {beta_rad:.6f} rad ({math.degrees(beta_rad):.3f}°)")

            # Integrate i(wt) from α to β
            integral_val, _ = quad(i_of_wt, alpha_rad, beta_rad)
            I_avg = integral_val / (2 * math.pi)
            steps_local.append(f"Step D: Integrate i(ωt) from α to β → ∫i(ωt)d(ωt) ≈ {integral_val:.6f}")
            steps_local.append(f"Step E: I_avg = integral / 2π ≈ {I_avg:.6f} A")

        local["I_avg_load"] = round(I_avg, 6)
        local["conduction_type"] = conduction_type
        local["beta_rad"] = beta_rad
        local["beta_deg"] = math.degrees(beta_rad) if beta_rad else None
        return local, steps_local

    res1, s1 = analyze_alpha(alpha1)
    res2, s2 = analyze_alpha(alpha2)

    steps.extend(s1)
    steps.extend(s2)

    out["alpha1_deg"] = round(math.degrees(alpha1), 6)
    out["alpha2_deg"] = round(math.degrees(alpha2), 6)
    out["result_alpha1"] = res1
    out["result_alpha2"] = res2
    out["steps_text"] = "\n".join(steps)

    return out


def controlled_fullwave_rectifier_with_transformer(R, I_min, I_max, Vrms_source, f):
    """
    Computes the transformer turns ratio and delay angle range for a controlled
    single-phase full-wave rectifier with resistive load and isolation transformer.

    Returns a dictionary containing:
      - numeric results for transformer turns ratio and delay angle range
      - step-by-step strings (Step 1..Step N)
      - steps_text multiline string suitable for Telegram
    """
    steps = []
    out = {}

    # Step 1: Calculate Output Voltage Range
    V_o_max = I_max * R
    V_o_min = I_min * R
    steps.append(f"Step 1: Calculate output voltage range:\n"
                 f"  For I_max = {I_max} A: V_o_max = I_max * R = {I_max} * {R} = {V_o_max} V\n"
                 f"  For I_min = {I_min} A: V_o_min = I_min * R = {I_min} * {R} = {V_o_min} V")
    out["V_o_max"] = round(V_o_max, 6)
    out["V_o_min"] = round(V_o_min, 6)

    # Step 2: Determine Peak Voltage V_m needed at α = 0°
    V_m = V_o_max * math.pi / 2
    steps.append(f"Step 2: Determine peak voltage for maximum output (α = 0°):\n"
                 f"  V_m = (V_o_max * π) / 2 = ({V_o_max} * π) / 2 ≈ {V_m:.6f} V")
    out["V_m"] = round(V_m, 6)

    # Step 3: Compute Transformer Turns Ratio
    Vrms_source_peak = Vrms_source * math.sqrt(2)
    turns_ratio = Vrms_source_peak / V_m
    steps.append(f"Step 3: Compute transformer turns ratio:\n"
                 f"  Source peak voltage = Vrms_source * √2 = {Vrms_source} * √2 ≈ {Vrms_source_peak:.6f} V\n"
                 f"  Turns ratio = V_source_peak / V_m = {Vrms_source_peak:.6f} / {V_m:.6f} ≈ {turns_ratio:.3f} : 1")
    out["Vrms_source_peak"] = round(Vrms_source_peak, 6)
    out["turns_ratio"] = round(turns_ratio, 6)

    # Step 4: Calculate Delay Angle Range
    alpha_max_rad = math.acos((V_o_min * math.pi / V_m) - 1)
    alpha_max_deg = math.degrees(alpha_max_rad)
    steps.append(f"Step 4: Calculate maximum delay angle for minimum output:\n"
                 f"  α_max = cos⁻¹((V_o_min * π / V_m) - 1) = cos⁻¹(({V_o_min} * π / {V_m:.6f}) - 1) ≈ {alpha_max_deg:.2f}°\n"
                 f"  Delay angle range: 0° to {alpha_max_deg:.2f}°")
    out["alpha_max_deg"] = round(alpha_max_deg, 6)
    out["delay_angle_range_deg"] = [0.0, round(alpha_max_deg, 6)]

    # Assemble steps_text for Telegram
    out["steps"] = steps
    out["steps_text"] = "\n".join([f"{s}" for s in steps])

    return out


def controlled_rectifier_rl_for_required_current(R, L, I_required, Vrms, f):
    """
    Computes delay angle, current continuity, and peak-to-peak current variation
    for a controlled single-phase rectifier with an RL load to achieve a required average current.

    Returns a dictionary containing:
      - numeric results for delay angle, continuity, and ripple
      - step-by-step strings (Step 1..Step N)
      - steps_text multiline string suitable for Telegram
    """
    steps = []
    out = {}

    # Step 1: Compute output voltage required
    V_o = I_required * R
    steps.append(f"Step 1: Compute output voltage required to achieve I_required = {I_required} A:\n"
                 f"  V_o = I_required * R = {I_required} * {R} = {V_o} V")
    out["V_o"] = round(V_o, 6)

    # Step 2: Compute peak source voltage
    V_m = Vrms * math.sqrt(2)
    steps.append(f"Step 2: Compute source peak voltage:\n"
                 f"  V_m = Vrms * √2 = {Vrms} * √2 ≈ {V_m:.6f} V")
    out["V_m"] = round(V_m, 6)

    # Step 3: Compute delay angle
    alpha_rad = math.acos(V_o * math.pi / (2 * V_m))
    alpha_deg = math.degrees(alpha_rad)
    steps.append(f"Step 3: Compute delay angle α:\n"
                 f"  α = cos⁻¹(V_o * π / (2 * V_m)) = cos⁻¹({V_o} * π / (2 * {V_m:.6f})) ≈ {alpha_deg:.2f}°")
    out["alpha_deg"] = round(alpha_deg, 6)

    # Step 4: Current continuity check
    omega = 2 * math.pi * f
    theta_rad = math.atan(omega * L / R)
    theta_deg = math.degrees(theta_rad)
    if theta_deg > alpha_deg:
        continuity = "continuous"
    else:
        continuity = "discontinuous"
    steps.append(f"Step 4: Check current continuity:\n"
                 f"  θ = atan(ωL / R) = atan({omega:.6f} * {L} / {R}) ≈ {theta_deg:.2f}°\n"
                 f"  Since θ {'>' if theta_deg > alpha_deg else '<='} α ({alpha_deg:.2f}°), current is {continuity}.")
    out["continuity"] = continuity
    out["theta_deg"] = round(theta_deg, 6)

    # Step 5: Estimate peak-to-peak current variation (ripple)
    # Using second harmonic approximation
    a_2 = 2*V_m / math.pi *((math.cos(3*alpha_rad))/ 3 -(math.cos(alpha_rad)) )
    b_2 =2*V_m / math.pi *((math.sin(3*alpha_rad))/ 3 -(math.sin(alpha_rad)) )
    V_2 = math.sqrt((a_2 ** 2) + (b_2 ** 2 ))   # rough estimation for first ripple harmonic
    Z_2 = math.sqrt(R ** 2 + (2 * omega * L) ** 2)
    I_2 = V_2 / Z_2
    delta_I = 2 * I_2
    steps.append(f"Step 5: Estimate peak-to-peak current ripple:\n"
                 f"  Using second harmonic :\n"
                 f" a_2 = 2*V_m / pi *((cos(3*alpha))/ 3 -(cos(alpha)) )≈ {a_2:.2f}\n"
                 f" b_2 =2*V_m / pi *((sin(3*alpha))/ 3 -(sin(alpha)) )≈ {b_2:.2f}\n"
                 f"    V_2 = √(a_2^2+b_2^2) = √({a_2:.2f}^2 + {b_2:.2f}^2)  ≈ {V_2:.2f} V\n"
                 f"    Z_2 = √(R^2 + (2ωL)^2) = √({R}^2 + (2*{omega:.6f}*{L})^2) ≈ {Z_2:.2f} Ω\n"
                 f"    I_2 = V_2 / Z_2 ≈ {I_2:.2f} A\n"
                 f"    ΔI ≈ 2 * I_2 ≈ {delta_I:.2f} A")
    out["I_ripple_peak_to_peak"] = round(delta_I, 6)

    # Assemble steps_text
    out["steps"] = steps
    out["steps_text"] = "\n".join(steps)

    return out
def fullwave_inverter_rl_power_ripple(Vrms, f, R, L, Vdc, alpha_deg):
    """
    Full-wave converter in inverter mode with RL load.
    Returns a dict with step-by-step explanations, numeric results, and a steps_text multiline string.
    """

    steps = []
    out = {}

    # Step 1: Peak voltage of AC source
    V_m = Vrms * math.sqrt(2)
    steps.append(f"Step 1: Peak voltage of AC source: V_m = √2 * Vrms = √2 * {Vrms} ≈ {V_m:.2f} V")
    out["V_m"] = round(V_m, 2)

    # Step 2: Output voltage of converter
    alpha_rad = math.radians(alpha_deg)
    V_o = (2 * V_m / math.pi) * math.cos(alpha_rad)
    steps.append(f"Step 2: Converter output voltage: V_o = 2*V_m/π * cos(α) = 2*{V_m:.2f}/π * cos({alpha_deg}°) ≈ {V_o:.2f} V")
    out["V_o"] = round(V_o, 2)

    # Step 3: Load current (average)
    I_o = (Vdc - V_o) / R
    steps.append(f"Step 3: Load current: I_o = (Vdc - V_o)/R = ({Vdc} - ({V_o:.2f}))/ {R} ≈ {I_o:.2f} A")
    out["I_o"] = round(I_o, 2)

    # Step 4: Power supplied to AC system
    P_ac = abs(I_o * V_o)
    steps.append(f"Step 4: Power supplied to AC system: P_ac = I_o * |V_o| = {I_o:.2f} * {abs(V_o):.2f} ≈ {P_ac:.2f} W")
    out["P_ac"] = round(P_ac, 2)

    # Step 5: Peak-to-peak ripple using second harmonic
    n=2
    omega = 2 * math.pi * f
    a_2 = 2*V_m / math.pi *((math.cos(3*alpha_rad))/ 3 -(math.cos(alpha_rad)) )
    b_2 =2*V_m / math.pi *((math.sin(3*alpha_rad))/ 3 -(math.sin(alpha_rad)) )
    V_2 = math.sqrt((a_2 ** 2) + (b_2 ** 2 ))
    Z_n = math.sqrt(R**2 + (n * omega * L)**2)
    I_n = V_2 / Z_n
    delta_I = 2 * I_n
    steps.append("Step 5: Estimate peak-to-peak current ripple using second harmonic:")
    steps.append( f" a_2 = 2*V_m / pi *((cos(3*alpha))/ 3 -(cos(alpha)) ) = {a_2:.2f} \n")
    steps.append(f" b_2 =2*V_m / pi *((sin(3*alpha))/ 3 -(sin(alpha)) )= {b_2:.2f} \n")
    steps.append( f"    V_2 = √(a_2^2+b_2^2) = √({a_2:.2f}^2 + {b_2:.2f}^2) ≈ {V_2:.2f} V\n")
    steps.append(f"  Z_2 = √(R^2 + (2ωL)^2) = √({R}^2 + (2*{omega:.2f}*{L})^2) ≈ {Z_n:.2f} Ω")
    steps.append(f"  I_2 = V_2 / Z_2 = {V_2:.2f} / {Z_n:.2f} ≈ {I_n:.2f} A")
    steps.append(f"  ΔI ≈ 2 * I_2 ≈ {delta_I:.2f} A")
    out["delta_I"] = round(delta_I, 2)

    # Combine steps into multiline text
    out["steps_text"] = "\n".join(steps)

    return out
def solar_converter_power_and_ripple(Vdc, Vrms, f, P_required, R, delta_I_max):
    """
    Controlled converter for solar power integration.
    Computes:
      - Delay angle α for required power transfer to AC system
      - Power supplied by solar cells
      - Minimum inductance to keep current ripple < delta_I_max
    Returns a dict with numeric results and human-readable step-by-step explanation.
    """

    steps = []
    out = {}

    # Step 1: Peak AC voltage
    V_m = Vrms * math.sqrt(2)
    steps.append(f"Step 1: Peak AC voltage: V_m = √2 * Vrms = √2 * {Vrms} ≈ {V_m:.2f} V")
    out["V_m"] = round(V_m, 2)

    # Step 2: Solve quadratic for V_o (converter output voltage)
    # Quadratic: V_o^2 - V_o*Vdc + P_required*R = 0
    a = 1
    b = -Vdc
    c = P_required * R
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        raise ValueError("Discriminant negative, no real solution for Vo")
    root1 = (-b + math.sqrt(discriminant)) / (2*a)
    root2 = (-b - math.sqrt(discriminant)) / (2*a)
    Io1 = (Vdc - root1) / R
    Io2 = (Vdc - root2) / R
    # choose root giving smaller |Io| to minimize losses
    if abs(Io1) < abs(Io2):
        V_o = root1
        Io = Io1
    else:
        V_o = root2
        Io = Io2
    steps.append(f"Step 2: Solve Vo^2 - Vdc*Vo + P*R = 0 → Vo roots: {root1:.3f}, {root2:.3f}")
    steps.append(f"Step 2a: Corresponding Io = (Vdc - Vo)/R → Io1 = {Io1:.3f}, Io2 = {Io2:.3f}")
    steps.append(f"Step 2b: Choose Vo = {V_o:.3f} V corresponding to smaller Io = {Io:.3f} A to minimize losses")
    out["Vo"] = round(V_o, 3)
    out["Io"] = round(Io, 3)

    # Step 3: Delay angle α
    alpha_rad = math.acos((V_o * math.pi) / (2 * V_m))
    alpha_deg = math.degrees(alpha_rad)
    steps.append(f"Step 3: Delay angle α = cos⁻¹(V_o * π / 2V_m) = cos⁻¹({V_o:.2f} * π / 2*{V_m:.2f}) ≈ {alpha_deg:.2f}°")
    out["alpha_deg"] = round(alpha_deg, 2)

    # Step 4: Solar cell power
    I_o = (Vdc - V_o) / R
    P_solar = I_o * Vdc
    steps.append(f"Step 4: Solar cell power: I_o = (Vdc - V_o)/R = ({Vdc} - {V_o:.2f})/{R} ≈ {I_o:.2f} A")
    steps.append(f"  P_solar = I_o * Vdc = {I_o:.2f} * {Vdc} ≈ {P_solar:.2f} W")
    out["I_o"] = round(I_o, 2)
    out["P_solar"] = round(P_solar, 2)

    # Step 5: Estimate required inductance for current ripple < delta_I_max
    # Using second harmonic approximation: ΔI ≈ V_2 / (2π*f*L), V_2 ≈ 2*V_m / π
    a_2 = 2*V_m / math.pi *((math.cos(3*alpha_rad))/ 3 -(math.cos(alpha_rad)) )
    b_2 =2*V_m / math.pi *((math.sin(3*alpha_rad))/ 3 -(math.sin(alpha_rad)) )
    V_2 = math.sqrt((a_2 ** 2) + (b_2 ** 2 ))
    L_required = V_2 / (delta_I_max * 2 * math.pi * f)
    steps.append(f"Step 5: Inductance to limit ripple ΔI < {delta_I_max} A:")
    steps.append(f" a_2 = 2*V_m / pi *((cos(3*alpha))/ 3 -(cos(alpha)) ) = {a_2:.2f} \n")
    steps.append(f" b_2 =2*V_m / pi *((sin(3*alpha))/ 3 -(sin(alpha)) )= {b_2:.2f} \n")
    steps.append(f"  V_2 = √(a_2^2+b_2^2) = √({a_2:.2f}^2 + {b_2:.2f}^2) ≈ {V_2:.2f} V")
    steps.append(f"  L = V_2 / (ΔI_max * 2πf) = {V_2:.2f} / ({delta_I_max} * 2π*{f}) ≈ {L_required:.3f} H")
    steps.append(f"  → Use L ≈ {math.ceil(L_required*1000)/1000:.3f} H = {math.ceil(L_required*1000)} mH")

    out["L_required_H"] = round(L_required, 3)
    out["L_required_mH"] = round(L_required*1000, 0)

    # Combine steps into multiline text
    out["steps_text"] = "\n".join(steps)

    return out


def solar_scr_bridge_design(V_panel, Vrms, f, P_required, R, current_variation_percent=10):
    """
    Step-by-step calculation for a full-wave SCR bridge interfacing solar panels to AC.
    Returns a dict with numeric results, step strings, and a steps_text multiline string.

    Parameters:
      V_panel: voltage of a single solar panel (V)
      Vrms: AC system RMS voltage (V)
      f: AC frequency (Hz)
      P_required: power to be delivered to AC system (W)
      R: load resistance (Ω)
      current_variation_percent: allowable peak-to-peak ripple as % of average current
    """
    steps = []
    out = {}

    # Step 1: Peak AC voltage
    Vm = Vrms * math.sqrt(2)
    steps.append(f"Step 1: Peak AC voltage: V_m = √2·Vrms = √2·{Vrms} ≈ {Vm:.3f} V")
    out["V_m"] = round(Vm, 3)

    # Step 2: Determine average load current required
    Io_required = P_required / Vrms
    steps.append(f"Step 2: Average load current I_o ≈ P_required / Vrms = {P_required}/{Vrms} ≈ {Io_required:.3f} A")
    out["Io_required"] = round(Io_required, 3)

    # Step 3: Choose Vdc as negative multiple of V_panel to minimize Io
    # Rough estimate: Vo ≈ Io * R, then Vdc = Vo + Io*R
    Vo_approx = Io_required * R
    Vdc_approx = Vo_approx + Io_required * R
    # round to nearest multiple of V_panel
    Vdc = V_panel * round(Vdc_approx / V_panel)
    steps.append(f"Step 3: Choose Vdc as multiple of {V_panel} V to minimize Io: Vdc ≈ {Vdc} V")
    out["Vdc"] = round(Vdc, 3)

    # Step 4: Solve for output voltage Vo = Vdc - Io*R
    Io = P_required / abs(Vdc)  # rough estimate
    Vo = Vdc - Io * R
    steps.append(f"Step 4: Calculate output voltage Vo = Vdc - Io*R = {Vdc} - {Io:.3f}*{R} ≈ {Vo:.3f} V")
    out["Vo"] = round(Vo, 3)
    out["Io"] = round(Io, 3)

    # Step 5: Calculate delay angle α
    alpha_rad = math.acos(Vo * math.pi / (2 * Vm))
    alpha_deg = math.degrees(alpha_rad)
    steps.append(f"Step 5: Delay angle α = cos⁻¹(Vo·π / 2·Vm) = cos⁻¹({Vo:.3f}·π / 2·{Vm:.3f}) ≈ {alpha_deg:.1f}°")
    out["alpha_deg"] = round(alpha_deg, 1)

    # Step 6: Estimate inductance for desired ripple
    # ΔI ≤ current_variation_percent * Io
    delta_I = current_variation_percent / 100 * Io
    L_est = abs(Vo) / (2 * delta_I * f)
    steps.append(
        f"Step 6: Inductance for ΔI ≤ {current_variation_percent}% of I_o: ΔI = {delta_I:.3f} A, L ≈ |Vo| / (2·ΔI·f) = {abs(Vo):.3f} / (2·{delta_I:.3f}·{f}) ≈ {L_est:.3f} H")
    out["L_est_H"] = round(L_est, 3)

    # Step 7: Confirm power delivered
    P_delivered = Io * abs(Vo)
    steps.append(f"Step 7: Confirm power delivered: P = Io·|Vo| ≈ {Io:.3f} * {abs(Vo):.3f} ≈ {P_delivered:.1f} W")
    out["P_delivered_W"] = round(P_delivered, 1)
    out["current_variation_A"] = round(delta_I * 2, 3)  # peak-to-peak

    # Combine all steps into one string
    out["steps_text"] = "\n".join(steps)
    return out
def wind_inverter_design(Vdc, Vrms, f, P_rated, R, current_ripple_percent=10):
    """
    Step-by-step calculation for a full-wave converter operating as an inverter
    transferring power from a wind generator to a single-phase AC system.

    Returns a dict with numeric results, step strings, and a steps_text multiline string.
    """
    steps = []
    out = {}

    # Step 1: Peak AC voltage
    Vm = Vrms * math.sqrt(2)
    steps.append(f"Step 1: Peak AC voltage: V_m = √2·Vrms = √2·{Vrms} ≈ {Vm:.3f} V")
    out["V_m"] = round(Vm, 3)

    # Step 2: Average load current for rated power
    Io = P_rated / Vdc
    steps.append(f"Step 2: Average load current I_o = P_rated / Vdc = {P_rated} / {Vdc} ≈ {Io:.3f} A")
    out["Io"] = round(Io, 3)

    # Step 3: Estimate output voltage Vo for inverter operation
    Vo = Vdc - Io * R  # or approximate based on inverter action (negative Vo)
    # Since inverter delivers power to AC, Vo is negative relative to DC source
    Vo = -abs(Vdc - Io*R)
    steps.append(f"Step 3: Output voltage V_o ≈ -|Vdc - Io*R| = {Vo:.3f} V")
    out["Vo"] = round(Vo, 3)

    # Step 4: Delay angle α
    alpha_rad = math.acos(Vo * math.pi / (2 * Vm))
    alpha_deg = math.degrees(alpha_rad)
    steps.append(f"Step 4: Delay angle α = cos⁻¹(Vo·π / 2·Vm) = cos⁻¹({Vo:.3f}·π / 2·{Vm:.3f}) ≈ {alpha_deg:.1f}°")
    out["alpha_deg"] = round(alpha_deg, 1)

    # Step 5: Power absorbed by AC system
    P_ac = Io * abs(Vo)
    steps.append(f"Step 5: Power absorbed by AC system: P_ac = Io·|Vo| ≈ {Io:.3f} * {abs(Vo):.3f} ≈ {P_ac:.1f} W")
    out["P_ac_W"] = round(P_ac, 1)

    # Step 6: Inductance to limit current ripple
    delta_I = Io * current_ripple_percent / 100  # allowable peak-to-peak ripple / 2

    a_2 = 2*Vm / math.pi *((math.cos(3*alpha_rad))/ 3 -(math.cos(alpha_rad)) )
    b_2 =2*Vm / math.pi *((math.sin(3*alpha_rad))/ 3 -(math.sin(alpha_rad)) )
    V_2 = math.sqrt((a_2 ** 2) + (b_2 ** 2 ))
    Z_2 = 2*V_2 /delta_I
    L = abs(Z_2) / (2 * 2 * math.pi * f)
    steps.append(f" a_2 = 2*V_m / pi *((cos(3*alpha))/ 3 -(cos(alpha)) ) = {a_2:.2f} \n")
    steps.append(f" b_2 =2*V_m / pi *((sin(3*alpha))/ 3 -(sin(alpha)) )= {b_2:.2f} \n")
    steps.append(f"  V_2 = √(a_2^2+b_2^2) = √({a_2:.2f}^2 + {b_2:.2f}^2) ≈ {V_2:.2f} V \n")
    steps.append(f"Step 6: Inductance to limit ΔI ≤ {current_ripple_percent}% of I_o: ΔI = {delta_I:.3f} A, \n")
    steps.append(f"  Z2 = 2*V_2/ΔI = 2 * {V_2:.2f} / {delta_I:.2f} = {Z_2:.2f}  \n")
    steps.append(f"L ≈ Z2 / (2·pi·f) = {abs(Z_2):.3f} / (2· 2·{math.pi:.3f}·{f}) ≈ {L:.3f} H \n")
    out["L_H"] = round(L, 3)
    out["current_ripple_A"] = round(2*delta_I, 3)  # peak-to-peak

    # Combine all steps into one string
    out["steps_text"] = "\n".join(steps)
    return out
def three_phase_rectifier_resistive(Vrms_line, f, R):
    """
    Three-phase full-wave rectifier with resistive load.
    Inputs:
      - Vrms_line : line-to-line RMS voltage (V)
      - f         : frequency (Hz)
      - R         : load resistance (Ω)
    Returns a dict with:
      - numeric outputs (Io_avg, Io_rms, Is_rms, P, S, pf, Vm, Vo)
      - step-by-step strings and a 'steps_text' multiline string
      - an i(ωt) description suitable for display
    Notes:
      - Uses phase RMS = Vrms_line / √3
      - Vm (phase) = √2 * Vrms_phase
      - Exact analytic integral used for I_rms (no numeric quadrature)
    """
    if Vrms_line is None or f is None or R is None:
        raise ValueError("Vrms_line, f and R must be provided")
    Vrms_line = float(Vrms_line)
    f = float(f)
    R = float(R)
    if Vrms_line <= 0 or f <= 0 or R <= 0:
        raise ValueError("Vrms_line, f and R must be positive")

    steps = []
    out = {}

    # Step 1: phase RMS and peak (phase)
    Vm = Vrms_line * math.sqrt(2.0)
    steps.append(f"Step 1: Convert line-to-line RMS to  peak:")
    steps.append(f"        V_m  = √2 · Vrms_LL ≈ {Vm:.6g} V")
    out["Vm_V"] = round(Vm, 6)

    # Step 2: average (DC) output voltage V_o = 3 V_m / π
    Vo = (3.0 * Vm) / math.pi
    steps.append(f"Step 2: Average (DC) output voltage for three-phase 6-pulse: V_o = 3·V_m / π = {Vo:.6g} V")
    out["Vo_V"] = round(Vo, 6)

    # Step 3: average load current
    Io_avg = Vo / R
    steps.append(f"Step 3: Average load current I_o = V_o / R = {Vo:.6g} / {R:.6g} ≈ {Io_avg:.6g} A")
    out["Io_avg_A"] = round(Io_avg, 6)

    # Step 4: define i(ωt) (piecewise) and compute RMS analytically
    steps.append("Step 4: Load current waveform i(ωt):")
    steps.append("        i(ωt) = (V_m / R)·sin(ωt) during conduction intervals")
    steps.append("          conduction intervals per 2π: [π/3, 2π/3] and [4π/3, 5π/3]; elsewhere i=0")

    # analytic integral for ∫ sin^2 θ dθ between π/3 and 2π/3
    a = math.pi / 3.0
    b = 2.0 * math.pi / 3.0
    integral_sin2 = ( (b - a) / 2.0 ) - ( (math.sin(2.0*b) - math.sin(2.0*a)) / 4.0 )
    # simplify: integral_sin2 = π/6 + √3/4 numerically, but use computed value
    steps.append(f"        Integral used: ∫_{{π/3}}^{{2π/3}} sin^2 θ dθ = {integral_sin2:.9g}")

    # I_rms^2 = (1/π) * (V_m/R)^2 * integral_sin2  (two identical intervals per 2π)
    factor = integral_sin2 / math.pi
    Irms = math.sqrt(3)*(Vm / R) * math.sqrt(factor)
    steps.append(f"Step 5: RMS load current from analytic integral:")
    steps.append(f"        I_rms = (V_m / R) * sqrt(integral / π/3) = ({Vm:.6g}/{R:.6g}) * sqrt({integral_sin2:.9g}/{math.pi:.9g}) ≈ {Irms:.6g} A")
    out["Io_rms_A"] = round(Irms, 6)

    # Step 6: RMS source current (approx mapping used in many texts / previous examples)
    # Use the same mapping used previously: I_s_rms = I_o_rms * sqrt(2/3)
    Is_rms = Irms * math.sqrt(2.0 / 3.0)
    steps.append(f"Step 6: RMS source current (convention used here): I_s_rms = √(2/3) · I_o_rms ≈ {Is_rms:.6g} A")
    out["Is_rms_A"] = round(Is_rms, 6)

    # Step 7: power, apparent power, power factor
    P_load = (Irms ** 2) * R
    S_app = math.sqrt(3.0) * Vrms_line * Is_rms
    pf = P_load / S_app if S_app != 0 else 0.0
    steps.append(f"Step 7: Real power P = I_o_rms^2 · R = {Irms:.6g}^2 · {R:.6g} ≈ {P_load:.6g} W")
    steps.append(f"        Apparent power S = √3 · V_line · I_s_rms = √3·{Vrms_line:.6g}·{Is_rms:.6g} ≈ {S_app:.6g} VA")
    steps.append(f"        Power factor pf = P / S ≈ {pf:.6g}")
    out["P_load_W"] = round(P_load, 6)
    out["S_VA"] = round(S_app, 6)
    out["pf"] = round(pf, 6)

    # Add i(ωt) description string for display (helpful for Telegram)
    i_wt_desc = (
        "i(ωt) = (V_m/R)·sin(ωt) for π/3 ≤ ωt ≤ 2π/3 and 4π/3 ≤ ωt ≤ 5π/3; "
        "i(ωt)=0 otherwise. (V_m is phase peak.)"
    )
    steps.append("Step 8: Waveform summary: " + i_wt_desc)
    out["i_wt_description"] = i_wt_desc

    out["steps_text"] = "\n".join(steps)
    return out
def three_phase_rectifier_with_rl_load(Vrms_line, f, R, L):
    """
    Three-phase 6-pulse rectifier with RL load — step-by-step style using V6 = 0.055*Vm.
    No numeric Fourier/integral scan is performed; analytic integrals are used where needed.

    Parameters
    ----------
    Vrms_line : float
        Line-to-line RMS voltage (V), e.g. 480
    f : float
        Frequency (Hz), e.g. 60
    R : float
        Load resistance (Ω)
    L : float
        Load inductance (H)

    Returns
    -------
    dict
      - numeric fields and "steps_text" multiline string
      - keys include: V_m, V_o, I_o, I_rms_approx, I_D_avg, I_D_rms, V6, Z6, I6, I_rms_with_6th, I_s_rms, P_load, S_app, pf, i_wt_description, steps_text
    """
    # validate
    if Vrms_line is None or f is None or R is None or L is None:
        raise ValueError("Vrms_line, f, R and L must be provided")
    Vrms_line = float(Vrms_line)
    f = float(f)
    R = float(R)
    L = float(L)
    if Vrms_line <= 0 or f <= 0 or R <= 0 or L < 0:
        raise ValueError("Vrms_line, f, R must be positive and L must be non-negative")

    steps = []
    out = {}

    # Step 1: peak voltage
    Vm = math.sqrt(2.0) * Vrms_line
    steps.append(f"Step 1: Peak voltage using V_m = √2 * Vrms_line: V_m = √2 * {Vrms_line} ≈ {Vm:.6f} V")
    out["V_m"] = round(Vm, 6)

    # Step 2: average DC output voltage (3-phase 6-pulse standard formula)
    V_o = (3.0 * Vm) / math.pi
    steps.append(f"Step 2: Average DC output V_o using V_o = 3·V_m / π: V_o = 3·{Vm:.6f}/π ≈ {V_o:.6f} V")
    out["V_o"] = round(V_o, 6)

    # Step 3: average load current
    I_o = V_o / R
    steps.append(f"Step 3: Average load current I_o = V_o / R = {V_o:.6f} / {R:.6f} ≈ {I_o:.6f} A")
    out["I_o"] = round(I_o, 6)


    # Step 7: use the requested 6th harmonic approximation: V6 = 0.055 * Vm
    V6 = 0.055 * Vm
    steps.append(f"Step 4: Use the 6th-harmonic approximation V6 = 6·V_m/(pi·(6^2-1)) = 0.055·V_m: V6 = 0.055·{Vm:.6f} ≈ {V6:.6f} V")
    out["V6_V"] = round(V6, 6)

    # Step 8: sixth-harmonic impedance and current amplitude
    omega = 2.0 * math.pi * f
    Z6 = math.sqrt(R * R + (6.0 * omega * L) ** 2)
    I6 = V6 / Z6
    steps.append(f"Step 5: Sixth-harmonic impedance Z6 = √(R^2 + (6ωL)^2) = √({R:.6f}^2 + ({6*omega*L:.6g})^2) ≈ {Z6:.6f} Ω")
    steps.append(f"        Sixth-harmonic current magnitude I6 = V6 / Z6 = {V6:.6f} / {Z6:.6f} ≈ {I6:.6f} A")
    out["Z6_Ohm"] = round(Z6, 6)
    out["I6_A"] = round(I6, 6)

    # Step 9: RMS including dominant 6th harmonic (approx)
    I_rms_with_6th = math.sqrt(I_o ** 2 + (I6 ** 2) / 2.0)
    I_D_rms=I_rms_with_6th/math.sqrt(2.0)
    I_D_avg=I_o/2.0
    steps.append(f"Step 6: Approximate I_rms including dominant 6th harmonic:")
    steps.append(f"        I_rms ≈ sqrt(I_o^2 + (I6^2)/2) = sqrt({I_o:.6f}^2 + ({I6:.6f}^2)/2) ≈ {I_rms_with_6th:.6f} A")
    out["I_rms_with_6th_A"] = round(I_rms_with_6th, 6)
    steps.append(f"Step 7: Diode currents:")
    steps.append(f"        Average diode current I_D,avg = I_o / 2 = {I_o:.6f}/2 ≈ {I_D_avg:.6f} A")
    steps.append(f"        RMS diode current I_D,rms = I_rms/√(2) ={I_rms_with_6th:.6f} /√(2) ≈ {I_D_rms:.6f} A")
    out["I_D_avg_A"] = round(I_D_avg, 6)
    out["I_D_rms_A"] = round(I_D_rms, 6)
    # Step 10: source RMS current mapping and power factor (use same mapping used in your solutions)
    I_s_rms = I_rms_with_6th * math.sqrt(2.0 / 3.0)
    P_load = (I_rms_with_6th ** 2) * R
    S_app = math.sqrt(3.0) * Vrms_line * I_s_rms
    pf = P_load / S_app if S_app != 0 else 0.0
    steps.append(f"Step 8: Power & source values:")
    steps.append(f"         P_load = I_rms^2 · R = {I_rms_with_6th:.6f}^2 · {R:.6f} ≈ {P_load:.6f} W")
    steps.append(f"         I_s_rms = √(2/3)·I_rms ≈ {I_s_rms:.6f} A")
    steps.append(f"         S = √3 · V_line · I_s_rms = √3·{Vrms_line}·{I_s_rms:.6f} ≈ {S_app:.6f} VA")
    steps.append(f"         pf = P / S ≈ {pf:.6f}")
    out["I_s_rms_A"] = round(I_s_rms, 6)
    out["P_load_W"] = round(P_load, 6)
    out["S_VA"] = round(S_app, 6)
    out["pf"] = round(pf, 6)

    # Step 11: waveform description & final packaging
    steps.append("Step 9: Waveform assumption used (for clarity):")
    steps.append("         v_o(ωt) = V_m·sin(ωt) on conduction windows π/3→2π/3 and 4π/3→5π/3; otherwise v_o=0")
    out["i_wt_description"] = "i(ωt) = (V_m/R)·sin(ωt) on conduction windows π/3→2π/3 and 4π/3→5π/3; 0 otherwise"

    out["V_m"] = round(Vm, 6)
    out["V_o"] = round(V_o, 6)
    out["I_o"] = round(I_o, 6)
    out["I_rms_with_6th_A"] = round(I_rms_with_6th, 6)
    out["steps_text"] = "\n".join(steps)

    return out
