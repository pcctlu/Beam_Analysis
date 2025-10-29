import streamlit as st
import numpy as np
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt

# ---------- Data Classes ----------
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

        elif self.beam_type == 'cantilever':
            V[0] = -np.sum([p.P for p in self.loads_point]) - sum(
                [udl.w * (min(udl.x_end, self.L) - max(udl.x_start, 0)) for udl in self.loads_udl])
            int_q = cumulative_trapezoid(q, x, initial=0.0)
            V = V[0] - int_q
            for idx, P in point_loads_index:
                V[idx:] -= P
            M = cumulative_trapezoid(V, x, initial=0.0)
            for idx, M0 in moment_loads_index:
                M[idx:] -= M0

        else:
            st.warning("Only simply_supported and cantilever beams are supported for now.")
            return {}

        d2w = -M / self.EI
        theta = cumulative_trapezoid(d2w, x, initial=0.0)
        w_raw = cumulative_trapezoid(theta, x, initial=0.0)

        if self.beam_type == 'simply_supported':
            C0 = -w_raw[0]
            C1 = (-w_raw[-1] - C0) / self.L
            w = w_raw + C0 + C1 * x
            theta = theta + C1
        else:
            C1 = -theta[0]
            theta = theta + C1
            C0 = -w_raw[0]
            w = w_raw + C1 * x + C0

        return {'x': x, 'q': q, 'V': V, 'M': M, 'theta': theta, 'w': w}

# ---------- Streamlit UI ----------
st.title("🧮 Beam Analysis App")
st.write("Analyze bending moment, shear force, slope & deflection for various beam types.")

# Input parameters
L = st.number_input("Beam Length (m)", 1.0, 100.0, 10.0)
EI = st.number_input("Flexural Rigidity (EI)", 1e4, 1e10, 200e6, step=1e6, format="%.1e")
beam_type = st.selectbox("Beam Type", ["simply_supported", "cantilever"])

beam = Beam(L, EI, beam_type=beam_type)

# Loads
st.subheader("Add Loads")

# Point loads
with st.expander("➕ Point Loads"):
    n_p = st.number_input("Number of Point Loads", 0, 10, 0)
    for i in range(n_p):
        x = st.number_input(f"Location of Point Load {i+1} (m)", 0.0, L, L/2)
        P = st.number_input(f"Magnitude of Point Load {i+1} (kN)", -1000.0, 1000.0, 10.0)
        beam.add_point_load(x, P)

# UDLs
with st.expander("➕ Uniformly Distributed Loads (UDL)"):
    n_udl = st.number_input("Number of UDLs", 0, 10, 0)
    for i in range(n_udl):
        x0 = st.number_input(f"Start of UDL {i+1} (m)", 0.0, L, 0.0)
        x1 = st.number_input(f"End of UDL {i+1} (m)", 0.0, L, L)
        w = st.number_input(f"Load Intensity of UDL {i+1} (kN/m)", -1000.0, 1000.0, 5.0)
        beam.add_udl(x0, x1, w)

# Moment loads
with st.expander("➕ Moment Loads"):
    n_m = st.number_input("Number of Moment Loads", 0, 10, 0)
    for i in range(n_m):
        x = st.number_input(f"Location of Moment {i+1} (m)", 0.0, L, L/2)
        M = st.number_input(f"Magnitude of Moment {i+1} (kN·m)", -1000.0, 1000.0, 20.0)
        beam.add_moment_load(x, M)

# Analyze
if st.button("Run Analysis"):
    res = beam.analyze()
    if res:
        st.success("✅ Analysis Completed!")
        st.write(f"**Maximum Moment:** {np.max(np.abs(res['M'])):.3f} kN·m")
        st.write(f"**Maximum Deflection:** {np.max(np.abs(res['w'])):.6f} m")

        # Plots
        fig, axs = plt.subplots(5, 1, figsize=(8, 12), sharex=True)
        axs[0].plot(res['x'], res['q']); axs[0].set_ylabel('Load q(x)')
        axs[1].plot(res['x'], res['V']); axs[1].set_ylabel('Shear V(x)')
        axs[2].plot(res['x'], res['M']); axs[2].set_ylabel('Moment M(x)')
        axs[3].plot(res['x'], res['theta']); axs[3].set_ylabel('Slope θ(x)')
        axs[4].plot(res['x'], res['w']); axs[4].set_ylabel('Deflection w(x)'); axs[4].set_xlabel('x (m)')
        for ax in axs: ax.grid(True)
        st.pyplot(fig)
