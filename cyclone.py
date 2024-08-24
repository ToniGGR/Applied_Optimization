import numpy as np
import math
import latexify

def calculation_mothes(x, cyclone, fluid):
    ra = cyclone['Da'] / 2
    ri = cyclone['Dt'] / 2
    rx = ri
    h = cyclone['H']
    ht = cyclone['Ht']

    he = cyclone['He']
    be = cyclone['Be']
    hk = h - he

    ra_alt = ra

    e = math.atan((ra - rx) / hk)

    vp = fluid['Vp']
    rhop = fluid['Rhop']
    croh = fluid['Croh']
    rhof = fluid['Rhof']
    mu = fluid['Mu']
    dp = 0.0125  # Diffusion coefficient from Abraham
    rb = 0.0075  # Friction coefficient from Meissner

    vis = mu
    hz = he

    vk = math.pi * (h - hz) * (ra**2 + rx**2 + ra * rx) / 3
    vk += math.pi * ra**2 * hz
    vr = vp / (2 * math.pi * ri * (h - ht))
    u = 1 - be / ra

    # Ensure that u is within [0,1]
    u = min(1, max(0, u))

    ar = -1 * math.atan(u / math.sqrt(-u * u + 1)) + math.pi / 2
    vd = vp / (math.pi * ra**2)
    bt = -0.204 * be / ra + 0.889
    ve = math.pi * ra**2 / (be * he * bt) * vd
    hc = (2 * math.pi - ar) / (2 * math.pi) - 1
    hc = hz / ra + he / ra * hc
    hc = vd / (rb * hc)
    ve = hc * (math.sqrt(0.25 + ve / hc) - 0.5)
    dm = ve / vd * (rb + rb / math.sin(e))
    vt = ve / ((ri / ra) * (1 + dm * (1 - ri / ra)))
    rf = ra
    ra = math.sqrt(vk / (math.pi * h))
    ve = ve / (ra / rf * (1 + dm * (1 - ra / rf)))

    wi = rhop * x**2 * vt**2 / (18 * vis * ri)
    wa = rhop * x**2 * ve**2 / (18 * vis * ra)
    k0 = h - ht
    k1 = 2 * math.pi * ra * wa / vp
    k2 = 2 * math.pi * ri * dp / (vp * (ra - ri))
    k3 = 2 * math.pi * ri * (wi - vr) / vp

    if (wi - vr) <= 0:
        a = -k0 * (-k1 + k3 - k2) - 1
        b = -k0 * k2
        c = k0 * (k2 - k3)
        d = b - 1
    else:
        a = -1 + k0 * (k1 + k2)
        b = k0 * (k2 - k3)
        c = k0 * k2
        d = b - 1

    m3 = (d + a) / 2
    m4 = math.sqrt(m3**2 - (a * d - b * c))
    m1 = m3 + m4
    m2 = m3 - m4
    c0 = 1
    c2 = c0 * math.exp(-k1 * (ht - he / 2))

    r1 = c2
    r2 = 0

    c4 = r1 * (m1 - a) / b + r2 * (m2 - a) / b
    t = 1 - c4 / c0

    # Pressure drop calculations
    BE = be / ra_alt
    re = ra_alt - be / 2

    Fe = be * he
    Fi = (math.pi * ri**2)
    F = Fe / Fi

    B = croh / rhof
    lambda_g = fluid['lambdag'] * (1 + 2 * math.sqrt(B))
    alpha = 1.0 - (0.54 - 0.153 / F) * BE**(1 / 3)
    vi = vp / (math.pi * ri**2)
    U = 1 / (F * alpha * ri / re + lambda_g * h / ri)

    xi2 = (U**2 * (ri / ra_alt)) * (1 - lambda_g * (h / ri) * U)**(-1)
    xi3 = 2 + 3 * U**(4 / 3) + U**2
    deltaP = (rhof / 2) * vi**2 * (xi2 + xi3)

    return t, deltaP

# Example usage:
#cyclone = {'Da': 2.0, 'Dt': 1.0, 'H': 5.0, 'Ht': 3.0, 'He': 1.0, 'Be': 0.5}
#fluid = {'Vp': 1.2, 'Rhop': 1.2, 'Croh': 0.05, 'Rhof': 1.0, 'Mu': 0.01, 'lambdag': 0.02}
#x = 0.05
#efficiency, pressure_drop = calculation_mothes(x, cyclone, fluid)
#print("Collection Efficiency:", efficiency)
#print("Pressure Drop:", pressure_drop)


def calculation_barth_muschelknautz(cyclone, fluid, xmean, delta):
    ra = cyclone['Da'] / 2
    ri = cyclone['Dt'] / 2
    h = cyclone['H']
    ht = cyclone['Ht']
    he = cyclone['He']
    be = cyclone['Be']

    vp = fluid['Vp']
    croh = fluid['Croh']
    rhof = fluid['Rhof']
    rhop = fluid['Rhop']
    mu = fluid['Mu']

    BE = be / ra
    re = ra - be / 2

    Fe = be * he
    Fi = math.pi * ri ** 2
    F = Fe / Fi

    B = croh / rhof
    lambda_g = fluid['lambdag'] * (1 + 2 * math.sqrt(B))
    alpha = 1.0 - (0.54 - 0.153 / F) * BE ** (1 / 3)
    vi = vp / (math.pi * ri ** 2)
    vr = vp / (2 * math.pi * ri * (h - ht))
    U = 1 / (F * alpha * ri / re + lambda_g * h / ri)
    vphii = U * vi

    xGr = math.sqrt((18 * mu * vr * ri) / ((rhop - rhof) * vphii ** 2))

    def Tf(x):
        return (1 + 2 / (x / xGr) ** 3.564) ** -1.235

    xi2 = (U ** 2 * (ri / ra)) * (1 - lambda_g * (h / ri) * U) ** -1
    xi3 = 2 + 3 * U ** (4 / 3) + U ** 2
    deltaP = (rhof / 2) * vi ** 2 * (xi2 + xi3)

    Ew = sum(Tf(xmean) * delta)

    x503 = xmean[np.cumsum(delta) >= 0.5][0]  # median of the particle size distribution
    ve = vp / Fe
    vphia = ve * ((re / ra) * (1 / alpha))
    try:
        BGr = (lambda_g * mu * math.sqrt(ra * ri)) / ((1 - ri / ra) * rhop * x503 ** 2 * math.sqrt(vphia * vphii))
    
        E = Ew
        if B > BGr:
            E = 1 - BGr / B + (BGr / B) * Ew
        if deltaP > 1000:
            return E,Ew, deltaP
    except:
        return 0,0,0
    return E, Ew, deltaP

# Example usage:
#cyclone = {'Da': 2.0, 'Dt': 1.0, 'H': 5.0, 'Ht': 3.0, 'He': 1.0, 'Be': 0.5}
#fluid = {'Vp': 1.2, 'Croh': 0.05, 'Rhof': 1, 'Rhop': 1.2, 'Mu': 0.01, 'lambdag': 0.02}
#xmean = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
#delta = np.array([0.1, 0.2, 0.3, 0.25, 0.15])
#efficiency, ew, pressure_drop = calculation_barth_muschelknautz(cyclone, fluid, xmean, delta)
#print("Efficiency:", efficiency)
#print("Ew (weighted efficiency):", ew)
#print("Pressure Drop:", pressure_drop)

def fun_cyclone(x, deterministic=[True, True, True],
                cyclone={'Da': 1.26, 'H': 2.5, 'Dt': 0.42, 'Ht': 0.65, 'He': 0.6, 'Be': 0.2},
                fluid={'Mu': 1.85e-5, 'Ve': (50/36)/0.12, 'lambdag': 1/200, 'Rhop': 2000, 'Rhof': 1.2, 'Croh': 0.05},
                noise_level={'Vp': 0.1, 'Rhop': 0.05}, model="Barth-Muschelknautz",
                intervals=np.array([0, 2, 4, 6, 8, 10, 15, 20, 30]) * 1e-6,
                delta=np.array([0.0, 0.02, 0.03, 0.05, 0.1, 0.3, 0.3, 0.2])):
    if np.isnan(x).any():
        return [np.nan, np.nan]

    # Update cyclone parameters based on x
    keys = ['Da', 'H', 'Dt', 'Ht', 'He', 'Be']
    cyclone.update({key: val for key, val in zip(keys, x)})

    fluid['Vp'] = fluid['Ve'] * cyclone['He'] * cyclone['Be']

    # Apply stochastic elements if non-deterministic
    if not deterministic[0]:
        fluid['Vp'] *= 1 + noise_level['Vp'] * (2 * np.random.rand() - 1)
    if not deterministic[1]:
        fluid['Rhop'] *= 1 + noise_level['Rhop'] * (2 * np.random.rand() - 1)

    xmin = intervals[:-1]
    xmax = intervals[1:]
    xmean = (xmin + xmax) / 2 if deterministic[2] else xmin + np.random.rand(len(xmin)) * (xmax - xmin)

    # Model selection and calculation
    if model == "Barth-Muschelknautz":
        E, Ew, PressureDrop = calculation_barth_muschelknautz(cyclone, fluid, xmean, delta)
    else:  # Mothes model
        PressureDrop = calculation_mothes(1, cyclone, fluid)[1]
        E = np.sum([calculation_mothes(d, cyclone, fluid)[0] for d in xmean] * delta)
        Ew = E  # Mothes model doesn't differentiate

    # Inverting efficiency for minimization
    return [PressureDrop, -E, -Ew]

# Example usage
#example_result = fun_cyclone(np.array([1.5, 3.0,         0.3,        0.5, 0.5, 0.1       ]),
#                             fluid={'Mu': 1.85e-5, 'Ve': (50/36)/0.12, 'lambdag': 1/200, 'Rhop': 2000, 'Rhof': 1.2, 'Croh': 0.05})
#example_result = fun_cyclone(np.array([1.26, 2.5, 0.42, 0.65, 0.6, 0.2]),
#                             fluid={'Mu': 1.85e-5, 'Ve': (50/36)/0.12, 'lambdag': 1/200, 'Rhop': 2000, 'Rhof': 1.2, 'Croh': 0.05}, 
#                             deterministic=[False, False, False])
#print("Results:", example_result)




#Good Results:

#[1.3760064390596898, 2.9999999999939626, 0.3517084091228143, 0.5000000000013354, 0.5000000000004706, 0.10000000000031344]  --> 98,74

#print(fun_cyclone)