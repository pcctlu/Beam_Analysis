import streamlit as st
import numpy as np
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt

# ---------- Classes ----------
class PointLoad:
    def __init__(self, x, P):
        self.x = x
        self.P = P

class MomentLoad:
    def __init__(self, x, M):
        self.x = x
        self.M = M

class UDL:
    def __init__(self, x_start, x_end, w):
        self.x_start = x_start
        self.x_end = x_end
        self.w = w

class Beam:
    def __init__(self, L, EI, n=1001, beam_type='simply_supported'):
        self.L = L
        self.EI = EI
        self.n = n
        self.x = np.linspace(0, L, n)
        self.dx = self.x[1] - self.x[0]
        self.loads_point = []
        self.loads_udl = []
        self.loads_moment = []
        self.beam_type = beam_type

    def add_point_load(self, x, P):
        self.loads_point.append(PointLoad(x, P))

    def add_moment_load(self, x, M):
        self.loads_moment.append(MomentLoad(x, M))

    def add_udl(self, x0, x1, w):
        self.loads_udl.append(UDL(x0, x1, w))

    def assemble_q(self):
        q = np.zeros_like(self.x)
        for udl in self.loads_udl:
            mask = (self.x >= udl.x_start) & (self.x <= udl.x_end)
            q[mask] += udl.w
        return q

    def reactions_for_simply_supported(self):
        L = self.L
        total_P = sum([p.P for p in self.loads_point])
        total_P += sum([udl.w * (min(udl.x_end, L) - max(udl.x_start, 0)) for udl in self.loads_udl])
        M_about_left = 0.0
        for p in self.loads_point:
            M_about_left += p.P * p.x
        for udl in self.loads_udl:
            a = max(udl.x_start, 0)
            b = min(udl.x_end, L)
            if b > a:
                length = b - a
                centroid = (a + b) / 2.0
                M_about_left += udl.w * length * centroid
        for m in self.loads_moment:
            M_about_left -= m.M
        RB = M_about_left / L
        RA = total_P - RB
        return RA, RB

    def analyze(self):
        x = self.x
        q = self.assemble_q()
        V = np.zeros_like(x)
        M = np.zeros_like(x)
        point_loads_index = [(np.argmin(np.abs(x - p.x)), p.P) for p in self.loads_point]
        moment_loads_index = [(np.argmin(np.abs(x - m.x)), m.M) for m in self.loads_moment]

        if self.beam_type == 'simply_supported':
            RA, RB = self.reactions_for_simply_supported()
            V[0] = RA
            int_q = cumulative_trapezoid(q, x, initial=0.0)
            V = V[0] - int_q
            for idx, P in point_loads_index:
                V[idx:] -= P
            M = cumulative_trapezoid(V, x, initial=0.0)
            for idx, M0 in moment_loads_index:
                M[idx:] -= M0
        else:
            raise RuntimeError("Only simply supported beam implemented for Streamlit demo")

        d2w = -M / self.EI
        theta = cumulative_trapezoid(d2w, x, initial=0.0)
        w_raw = cumulative_trapezoid(theta, x, initial=0.0)
        C0 = -w_raw[0]
        C1 = (-w_raw[-1] - C0) / self.L
        w = w_raw + C0 + C1 * x
        theta = theta + C1

        self.q, self.V, self.M, self.theta, self.w = q, V, M, theta, w
        return {'x': x, 'q': q, 'V': V, 'M': M, 'theta': theta, 'w': w}

# ---------- Streamlit UI ----------
st.title("🧮 Beam Analysis App")

L = st.number_input("Beam length (m)", 1.0, 100.0, 10.0)
EI = st.number_input("Flexural Rigidity (EI)", 1e4, 1e10, 2e8)
beam_type = st.selectbox("Beam Type", ['simply_supported'])

beam = Beam(L=L, EI=EI, n=1001, beam_type=beam_type)

st.subheader("➕ Add Loads")

# Point Loads
st.markdown("#### Point Loads")
n_point = st.number_input("Number of point loads", 0, 5, 0)
for i in range(n_point):
    x = st.number_input(f"Location of Point Load {i+1} (m)", 0.0, L, 0.0, key=f"plx_{i}")
    P = st.number_input(f"Magnitude of Point Load {i+1} (kN, +ve down)", -1000.0, 1000.0, 0.0, key=f"plp_{i}")
    if P != 0:
        beam.add_point_load(x, P)

# UDLs
st.markdown("#### Uniformly Distributed Loads (UDL)")
n_udl = st.number_input("Number of UDLs", 0, 5, 0)
for i in range(n_udl):
    x0 = st.number_input(f"Start of UDL {i+1} (m)", 0.0, L, 0.0, key=f"udl_start_{i}")
    x1 = st.number_input(f"End of UDL {i+1} (m)", 0.0, L, L, key=f"udl_end_{i}")
    w = st.number_input(f"Intensity of UDL {i+1} (kN/m, +ve down)", -1000.0, 1000.0, 0.0, key=f"udl_w_{i}")
    if w != 0:
        beam.add_udl(x0, x1, w)

# Analyze
if st.button("Run Analysis"):
    results = beam.analyze()
    st.success(" Analysis Completed!")

    Mmax = np.max(np.abs(results['M']))
    wmax = np.max(np.abs(results['w']))
    st.write(f"**Maximum Bending Moment:** {Mmax:.3f} kN·m")
    st.write(f"**Maximum Deflection:** {wmax:.6f} m")

    # Plot Results
    fig, axs = plt.subplots(5, 1, figsize=(8, 14), sharex=True)
    axs[0].plot(results['x'], results['q']); axs[0].set_ylabel('q(x) [kN/m]')
    axs[1].plot(results['x'], results['V']); axs[1].set_ylabel('Shear V(x) [kN]')
    axs[2].plot(results['x'], results['M']); axs[2].set_ylabel('Moment M(x) [kN·m]')
    axs[3].plot(results['x'], results['theta']); axs[3].set_ylabel('Slope θ(x)')
    axs[4].plot(results['x'], results['w']); axs[4].set_ylabel('Deflection w(x) [m]')
    axs[4].set_xlabel('x (m)')
    for ax in axs: ax.grid(True)
    st.pyplot(fig)
