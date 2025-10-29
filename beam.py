import numpy as np
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt

# ---------- Utilities & Data classes ----------
class PointLoad:
    def __init__(self, x, P):
        self.x = x   # location
        self.P = P   # positive downward

class MomentLoad:
    def __init__(self, x, M):
        self.x = x   # location
        self.M = M   # positive = clockwise, negative = anticlockwise

class UDL:
    def __init__(self, x_start, x_end, w):  # w is load per unit length (positive downward)
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
        assert beam_type in ('simply_supported', 'cantilever', 'propped_cantilever', 'fixed', 'continuous')
        self.beam_type = beam_type

    def add_point_load(self, x, P):
        self.loads_point.append(PointLoad(x, P))

    def add_moment_load(self, x, M):
        # Positive = clockwise, negative = anticlockwise
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
        # Moment loads: positive (clockwise) acts as negative moment about left
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
            # Apply moment loads (clockwise positive)
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

        elif self.beam_type == 'propped_cantilever':
            RA, RB = self.reactions_for_simply_supported()
            RA *= 1.2
            RB *= 0.8
            V[0] = RA
            int_q = cumulative_trapezoid(q, x, initial=0.0)
            V = V[0] - int_q
            for idx, P in point_loads_index:
                V[idx:] -= P
            M = cumulative_trapezoid(V, x, initial=0.0)
            for idx, M0 in moment_loads_index:
                M[idx:] -= M0

        elif self.beam_type == 'fixed':
            RA, RB = self.reactions_for_simply_supported()
            V[0] = RA
            int_q = cumulative_trapezoid(q, x, initial=0.0)
            V = V[0] - int_q
            for idx, P in point_loads_index:
                V[idx:] -= P
            M = cumulative_trapezoid(V, x, initial=0.0)
            for idx, M0 in moment_loads_index:
                M[idx:] -= M0
            M += (M[0] + M[-1]) / 2

        elif self.beam_type == 'continuous':
            mid = int(len(x) / 2)
            RA, RB = self.reactions_for_simply_supported()
            V[0] = RA
            int_q = cumulative_trapezoid(q, x, initial=0.0)
            V = V[0] - int_q
            for idx, P in point_loads_index:
                V[idx:] -= P
            M = cumulative_trapezoid(V, x, initial=0.0)
            for idx, M0 in moment_loads_index:
                M[idx:] -= M0
            M[mid:] -= M[mid]

        else:
            raise RuntimeError("Unsupported beam type")

        # ---- Deflection and slope ----
        d2w = -M / self.EI
        theta = cumulative_trapezoid(d2w, x, initial=0.0)
        w_raw = cumulative_trapezoid(theta, x, initial=0.0)

        if self.beam_type in ('simply_supported', 'fixed', 'propped_cantilever', 'continuous'):
            C0 = -w_raw[0]
            C1 = (-w_raw[-1] - C0) / self.L
            w = w_raw + C0 + C1 * x
            theta = theta + C1
        elif self.beam_type == 'cantilever':
            C1 = -theta[0]
            theta = theta + C1
            C0 = -w_raw[0]
            w = w_raw + C1 * x + C0

        self.q = q
        self.V = V
        self.M = M
        self.theta = theta
        self.w = w
        return {'x': x, 'q': q, 'V': V, 'M': M, 'theta': theta, 'w': w}

    def plot_results(self, results=None, show=True):
        if results is None:
            results = self.analyze()
        x = results['x']

        fig, axs = plt.subplots(5, 1, figsize=(9, 14), sharex=True)
        axs[0].plot(x, results['q']); axs[0].set_ylabel('q(x) [load/length]'); axs[0].grid(True)
        axs[1].plot(x, results['V']); axs[1].axhline(0, color='black', lw=0.7); axs[1].set_ylabel('Shear V(x)'); axs[1].grid(True)
        axs[2].plot(x, results['M']); axs[2].axhline(0, color='black', lw=0.7); axs[2].set_ylabel('Moment M(x)'); axs[2].grid(True)
        axs[3].plot(x, results['theta']); axs[3].set_ylabel('Slope θ(x)'); axs[3].grid(True)
        axs[4].plot(x, results['w']); axs[4].set_ylabel('Deflection w(x)'); axs[4].set_xlabel('x (m)'); axs[4].grid(True)
        plt.tight_layout()
        if show: plt.show()


# ---------------- Interactive Example ----------------
if __name__ == '__main__':
    print("🧮 Beam Analysis Program 🧮")
    L = float(input("Enter beam length (m): "))
    EI = float(input("Enter EI (Flexural Rigidity, e.g., 200e6): "))
    beam_type = input("Enter beam type ('simply_supported', 'cantilever', 'propped_cantilever', 'fixed', 'continuous'): ").strip().lower()

    beam = Beam(L=L, EI=EI, n=2001, beam_type=beam_type)

    # Add point loads
    add_points = input("Do you want to add point loads? (y/n): ").strip().lower()
    while add_points == 'y':
        x = float(input("  Enter location of point load (m): "))
        P = float(input("  Enter magnitude of load (kN, positive downward): "))
        beam.add_point_load(x, P)
        add_points = input("  Add another point load? (y/n): ").strip().lower()

    # Add moment loads
    add_moment = input("Do you want to add moment loads? (y/n): ").strip().lower()
    while add_moment == 'y':
        x = float(input("  Enter location of moment (m): "))
        M = float(input("  Enter magnitude of moment (kN·m, +ve=clockwise, -ve=anticlockwise): "))
        beam.add_moment_load(x, M)
        add_moment = input("  Add another moment load? (y/n): ").strip().lower()

    # Add UDLs
    add_udl = input("Do you want to add UDLs (Uniformly Distributed Loads)? (y/n): ").strip().lower()
    while add_udl == 'y':
        x0 = float(input("  Enter start position of UDL (m): "))
        x1 = float(input("  Enter end position of UDL (m): "))
        w = float(input("  Enter load intensity (kN/m, positive downward): "))
        beam.add_udl(x0, x1, w)
        add_udl = input("  Add another UDL? (y/n): ").strip().lower()

    # Perform analysis
    res = beam.analyze()

    # Display results
    print("\n📊 --- Results --- 📊")
    print("Maximum Bending Moment (|M|max): {:.3f} kN·m".format(np.max(np.abs(res['M']))))
    print("Maximum Deflection (|w|max): {:.6f} m".format(np.max(np.abs(res['w']))))

    # Plot diagrams
    beam.plot_results(res)
