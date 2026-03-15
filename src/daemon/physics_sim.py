"""Toy physics simulations - dynamical models for climate research."""

from __future__ import annotations

import json
import logging
import textwrap
from typing import Any

from ulid import ULID

log = logging.getLogger("daemon.physics_sim")


SIM_TEMPLATES: dict[str, dict[str, Any]] = {
    "lorenz63": {
        "description": "Lorenz 1963 attractor - chaos, sensitivity to initial conditions",
        "requirements": ["numpy", "scipy", "matplotlib"],
        "parameters": {
            "sigma": 10.0, "rho": 28.0, "beta": 2.6667,
            "dt": 0.01, "n_steps": 10000, "ic": [1.0, 1.0, 1.0],
        },
    },
    "shallow_water": {
        "description": "1D shallow water equations - gravity waves, Rossby adjustment",
        "requirements": ["numpy", "matplotlib"],
        "parameters": {
            "nx": 200, "dx": 1000.0, "dt": 1.0, "n_steps": 5000,
            "g": 9.81, "H": 100.0, "perturbation": "gaussian",
        },
    },
    "barotropic_vorticity": {
        "description": "Barotropic vorticity equation - Rossby waves, jet stream dynamics",
        "requirements": ["numpy", "scipy", "matplotlib"],
        "parameters": {
            "nx": 64, "ny": 64, "beta": 1.6e-11,
            "dt": 3600.0, "n_steps": 240,
        },
    },
    "tc_potential_intensity": {
        "description": "Emanuel PI theory - maximum potential intensity of tropical cyclones",
        "requirements": ["numpy", "matplotlib"],
        "parameters": {
            "sst_range": [26, 32], "outflow_temp": 200.0,
            "surface_pressure": 1015.0,
        },
    },
    "simple_gcm": {
        "description": "Held-Suarez forcing test - idealized atmospheric circulation",
        "requirements": ["numpy", "scipy", "matplotlib"],
        "parameters": {
            "nlev": 20, "nlat": 32, "dt": 1800.0, "n_days": 30,
        },
    },
}


class PhysicsSimulator:
    def __init__(self, config: Any, db: Any, sandbox_runner: Any) -> None:
        self.config = config
        self.db = db
        self.sandbox = sandbox_runner

    def run_simulation(self, sim_type: str, parameters: dict[str, Any] | None = None,
                       description: str = "") -> dict[str, Any]:
        """Run a physics simulation via sandbox."""
        if sim_type not in SIM_TEMPLATES:
            return {"error": f"Unknown sim type: {sim_type}. Available: {list(SIM_TEMPLATES.keys())}"}

        template = SIM_TEMPLATES[sim_type]
        params = {**template["parameters"]}
        if parameters:
            params.update(parameters)

        # Generate script
        script_fn = getattr(self, f"_script_{sim_type}", None)
        if not script_fn:
            return {"error": f"No script template for sim type: {sim_type}"}

        script = script_fn(params)
        timeout = self.config.physics_sim_timeout_s

        sandbox_result = self.sandbox.execute(
            script=script,
            requirements=template["requirements"],
            timeout_s=timeout,
            description=description or f"Physics sim: {sim_type}",
        )

        sim_id = str(ULID())
        metrics = sandbox_result.metrics or {}
        artifacts = sandbox_result.artifacts or []

        self.db.insert_physics_sim({
            "id": sim_id,
            "sim_type": sim_type,
            "parameters": params,
            "execution_id": sandbox_result.execution_id,
            "summary": metrics.get("summary", ""),
            "artifacts": artifacts,
        })

        return {
            "sim_id": sim_id,
            "execution_id": sandbox_result.execution_id,
            "sim_type": sim_type,
            "metrics": metrics,
            "artifacts": artifacts,
            "stdout": sandbox_result.stdout[:2000],
            "returncode": sandbox_result.returncode,
        }

    def list_simulations(self, sim_type: str | None = None) -> list[dict[str, Any]]:
        return self.db.list_physics_sims(sim_type=sim_type)

    def get_simulation(self, sim_id: str) -> dict[str, Any] | None:
        return self.db.get_physics_sim(sim_id)

    def list_available_sims(self) -> list[dict[str, Any]]:
        """List available simulation types with descriptions and default params."""
        return [
            {
                "sim_type": name,
                "description": tmpl["description"],
                "default_parameters": tmpl["parameters"],
                "requirements": tmpl["requirements"],
            }
            for name, tmpl in SIM_TEMPLATES.items()
        ]

    # -- Script templates ------------------------------------------------

    def _script_lorenz63(self, params: dict[str, Any]) -> str:
        return textwrap.dedent(f"""\
            import json
            import numpy as np
            from scipy.integrate import odeint
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            sigma = {params['sigma']}
            rho = {params['rho']}
            beta = {params['beta']}
            dt = {params['dt']}
            n_steps = {params['n_steps']}
            ic = {params['ic']}

            def lorenz(state, t, sigma, rho, beta):
                x, y, z = state
                return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

            t = np.arange(0, n_steps * dt, dt)
            sol = odeint(lorenz, ic, t, args=(sigma, rho, beta))

            # Compute max Lyapunov exponent (simple estimation)
            ic2 = [ic[0] + 1e-8, ic[1], ic[2]]
            sol2 = odeint(lorenz, ic2, t, args=(sigma, rho, beta))
            d = np.sqrt(np.sum((sol - sol2)**2, axis=1))
            d = d[d > 0]
            if len(d) > 100:
                lyap = np.mean(np.log(d[100:] / d[0])) / (t[100])
            else:
                lyap = 0.0

            # Energy metrics
            mean_x = float(np.mean(sol[:, 0]))
            mean_z = float(np.mean(sol[:, 2]))
            std_x = float(np.std(sol[:, 0]))

            # Phase space plot
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
            ax.plot(sol[:, 0], sol[:, 1], sol[:, 2], lw=0.3, alpha=0.8)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title("Lorenz 63 Attractor")
            plt.savefig("lorenz63_phase.png", dpi=100, bbox_inches="tight")
            plt.close()

            # Time series
            fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
            for i, label in enumerate(["X", "Y", "Z"]):
                axes[i].plot(t[:5000], sol[:5000, i], lw=0.5)
                axes[i].set_ylabel(label)
            axes[-1].set_xlabel("Time")
            fig.suptitle("Lorenz 63 Time Series")
            plt.savefig("lorenz63_timeseries.png", dpi=100, bbox_inches="tight")
            plt.close()

            result = {{
                "max_lyapunov_exponent": round(float(lyap), 4),
                "mean_x": round(mean_x, 4),
                "mean_z": round(mean_z, 4),
                "std_x": round(std_x, 4),
                "n_steps": n_steps,
                "summary": f"Lorenz63: lyapunov={{lyap:.3f}}, mean_z={{mean_z:.1f}}",
            }}
            with open("results.json", "w") as f:
                json.dump(result, f)
            print(json.dumps(result, indent=2))
        """)

    def _script_shallow_water(self, params: dict[str, Any]) -> str:
        return textwrap.dedent(f"""\
            import json
            import numpy as np
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            nx = {params['nx']}
            dx = {params['dx']}
            dt = {params['dt']}
            n_steps = {params['n_steps']}
            g = {params['g']}
            H = {params['H']}

            # Initialize
            x = np.arange(nx) * dx
            h = np.zeros(nx) + H
            u = np.zeros(nx)

            # Gaussian perturbation in center
            center = nx // 2
            h += 1.0 * np.exp(-((x - x[center])**2) / (2 * (10 * dx)**2))

            # Store initial energy
            E0 = 0.5 * g * np.sum(h**2) * dx + 0.5 * H * np.sum(u**2) * dx

            # Leapfrog time stepping
            h_hist = [h.copy()]
            for step in range(n_steps):
                # Update u
                dhdx = np.zeros(nx)
                dhdx[1:-1] = (h[2:] - h[:-2]) / (2 * dx)
                u = u - dt * g * dhdx

                # Update h
                dudx = np.zeros(nx)
                dudx[1:-1] = (u[2:] - u[:-2]) / (2 * dx)
                h = h - dt * H * dudx

                # Boundary: zero gradient
                h[0] = h[1]
                h[-1] = h[-2]
                u[0] = u[1]
                u[-1] = u[-2]

                if step % 500 == 0:
                    h_hist.append(h.copy())

            Ef = 0.5 * g * np.sum(h**2) * dx + 0.5 * H * np.sum(u**2) * dx
            energy_error = abs(Ef - E0) / E0 if E0 > 0 else 0.0

            # Hovmoller diagram
            h_arr = np.array(h_hist)
            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.pcolormesh(x / 1000, np.arange(len(h_hist)), h_arr - H, cmap="RdBu_r")
            ax.set_xlabel("Distance (km)")
            ax.set_ylabel("Time step (x500)")
            ax.set_title("Shallow Water: Height Perturbation")
            plt.colorbar(im, label="h - H (m)")
            plt.savefig("shallow_water_hovmoller.png", dpi=100, bbox_inches="tight")
            plt.close()

            max_amplitude = float(np.max(np.abs(h - H)))
            wave_speed = float(np.sqrt(g * H))

            result = {{
                "energy_conservation_error": round(energy_error, 8),
                "max_amplitude_m": round(max_amplitude, 4),
                "theoretical_wave_speed_ms": round(wave_speed, 2),
                "n_steps": n_steps,
                "summary": f"Shallow water: energy_error={{energy_error:.2e}}, max_amp={{max_amplitude:.3f}}m",
            }}
            with open("results.json", "w") as f:
                json.dump(result, f)
            print(json.dumps(result, indent=2))
        """)

    def _script_barotropic_vorticity(self, params: dict[str, Any]) -> str:
        return textwrap.dedent(f"""\
            import json
            import numpy as np
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            nx = {params['nx']}
            ny = {params['ny']}
            beta = {params['beta']}
            dt = {params['dt']}
            n_steps = {params['n_steps']}

            Lx = 1e7  # 10,000 km domain
            Ly = 1e7
            dx = Lx / nx
            dy = Ly / ny

            # Wavenumbers for spectral derivatives
            kx = np.fft.fftfreq(nx, d=dx) * 2 * np.pi
            ky = np.fft.fftfreq(ny, d=dy) * 2 * np.pi
            KX, KY = np.meshgrid(kx, ky)
            K2 = KX**2 + KY**2
            K2[0, 0] = 1.0  # avoid division by zero

            # Initial vorticity: jet + perturbation
            x = np.linspace(0, Lx, nx)
            y = np.linspace(0, Ly, ny)
            X, Y = np.meshgrid(x, y)
            zeta = 1e-5 * np.sin(2 * np.pi * Y / Ly) + 1e-6 * np.random.randn(ny, nx)

            # Time integration (pseudo-spectral)
            enstrophy_hist = []
            for step in range(n_steps):
                zeta_hat = np.fft.fft2(zeta)
                psi_hat = -zeta_hat / K2
                psi_hat[0, 0] = 0

                # Velocities
                u = np.real(np.fft.ifft2(1j * KY * psi_hat))
                v = np.real(np.fft.ifft2(-1j * KX * psi_hat))

                # Advection: -u * dzeta/dx - v * dzeta/dy - beta * v
                dzeta_dx = np.real(np.fft.ifft2(1j * KX * zeta_hat))
                dzeta_dy = np.real(np.fft.ifft2(1j * KY * zeta_hat))

                tendency = -u * dzeta_dx - v * dzeta_dy - beta * v

                # Euler forward (simple)
                zeta = zeta + dt * tendency

                if step % 24 == 0:
                    enstrophy = float(0.5 * np.mean(zeta**2))
                    enstrophy_hist.append(enstrophy)

            # Final state plot
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.pcolormesh(X / 1e6, Y / 1e6, zeta, cmap="RdBu_r")
            ax.set_xlabel("X (1000 km)")
            ax.set_ylabel("Y (1000 km)")
            ax.set_title("Barotropic Vorticity (final state)")
            plt.colorbar(im, label="Vorticity (1/s)")
            plt.savefig("barotropic_vorticity.png", dpi=100, bbox_inches="tight")
            plt.close()

            # Enstrophy time series
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(enstrophy_hist)
            ax.set_xlabel("Time step (x24)")
            ax.set_ylabel("Enstrophy")
            ax.set_title("Enstrophy Evolution")
            plt.savefig("barotropic_enstrophy.png", dpi=100, bbox_inches="tight")
            plt.close()

            final_enstrophy = enstrophy_hist[-1] if enstrophy_hist else 0.0
            initial_enstrophy = enstrophy_hist[0] if enstrophy_hist else 0.0

            result = {{
                "initial_enstrophy": round(initial_enstrophy, 10),
                "final_enstrophy": round(final_enstrophy, 10),
                "enstrophy_ratio": round(final_enstrophy / initial_enstrophy, 4) if initial_enstrophy > 0 else 0,
                "max_vorticity": round(float(np.max(np.abs(zeta))), 8),
                "n_steps": n_steps,
                "summary": f"Barotropic vorticity: enstrophy_ratio={{final_enstrophy/max(initial_enstrophy,1e-20):.3f}}",
            }}
            with open("results.json", "w") as f:
                json.dump(result, f)
            print(json.dumps(result, indent=2))
        """)

    def _script_tc_potential_intensity(self, params: dict[str, Any]) -> str:
        return textwrap.dedent(f"""\
            import json
            import numpy as np
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            sst_range = {params['sst_range']}
            T_out = {params['outflow_temp']}  # K
            p_s = {params['surface_pressure']}  # hPa

            sst_values = np.linspace(sst_range[0], sst_range[1], 50)
            sst_K = sst_values + 273.15

            # Emanuel (1988) potential intensity
            # V_max^2 ~ (Ts/To) * (Ck/Cd) * (k*_s - k_a)
            # Simplified version using Clausius-Clapeyron
            Ck_Cd = 0.9  # ratio of exchange coefficients
            L_v = 2.5e6  # latent heat of vaporization (J/kg)
            R_v = 461.5  # gas constant for water vapor
            c_p = 1005.0  # specific heat at constant pressure

            # Saturation specific humidity (approx)
            e_s = 6.112 * np.exp(17.67 * sst_values / (sst_values + 243.5))  # hPa
            q_s = 0.622 * e_s / (p_s - e_s)

            # Environmental values (80% RH at SST - 2K)
            T_env = sst_values - 2.0
            e_env = 0.8 * 6.112 * np.exp(17.67 * T_env / (T_env + 243.5))
            q_env = 0.622 * e_env / (p_s - e_env)

            # Enthalpy disequilibrium
            delta_k = c_p * (sst_values - T_env) + L_v * (q_s - q_env)

            # PI calculation
            Ts_To = sst_K / T_out
            v_max_sq = Ck_Cd * Ts_To * delta_k
            v_max_ms = np.sqrt(np.maximum(v_max_sq, 0))
            v_max_kt = v_max_ms * 1.94384  # m/s to knots

            # Plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            ax1.plot(sst_values, v_max_kt, "b-", linewidth=2)
            ax1.axhline(y=64, color="r", linestyle="--", label="Hurricane threshold (64 kt)")
            ax1.axhline(y=96, color="orange", linestyle="--", label="Major hurricane (96 kt)")
            ax1.set_xlabel("SST (C)")
            ax1.set_ylabel("Max Potential Intensity (kt)")
            ax1.set_title("Tropical Cyclone Potential Intensity")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2.plot(sst_values, delta_k / 1000, "g-", linewidth=2)
            ax2.set_xlabel("SST (C)")
            ax2.set_ylabel("Enthalpy disequilibrium (kJ/kg)")
            ax2.set_title("Air-Sea Enthalpy Disequilibrium")
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig("tc_potential_intensity.png", dpi=100, bbox_inches="tight")
            plt.close()

            # Key metrics
            pi_at_28 = float(np.interp(28.0, sst_values, v_max_kt))
            pi_at_30 = float(np.interp(30.0, sst_values, v_max_kt))
            max_pi = float(np.max(v_max_kt))
            sst_at_max = float(sst_values[np.argmax(v_max_kt)])

            result = {{
                "max_potential_intensity_kt": round(max_pi, 1),
                "pi_at_28C_kt": round(pi_at_28, 1),
                "pi_at_30C_kt": round(pi_at_30, 1),
                "sst_at_max_pi": round(sst_at_max, 1),
                "sensitivity_kt_per_C": round(float((pi_at_30 - pi_at_28) / 2.0), 2),
                "summary": f"TC PI: max={{max_pi:.0f}}kt at SST={{sst_at_max:.1f}}C",
            }}
            with open("results.json", "w") as f:
                json.dump(result, f)
            print(json.dumps(result, indent=2))
        """)

    def _script_simple_gcm(self, params: dict[str, Any]) -> str:
        return textwrap.dedent(f"""\
            import json
            import numpy as np
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            nlev = {params['nlev']}
            nlat = {params['nlat']}
            dt = {params['dt']}
            n_days = {params['n_days']}
            n_steps = int(n_days * 86400 / dt)

            # Held-Suarez (1994) forcing
            p_levels = np.linspace(0.05, 1.0, nlev)  # sigma levels
            lats = np.linspace(-90, 90, nlat)
            lat_rad = np.deg2rad(lats)

            # Equilibrium temperature
            T_eq = np.zeros((nlat, nlev))
            for j in range(nlat):
                for k in range(nlev):
                    T_eq[j, k] = max(200.0,
                        315.0 - 60.0 * np.sin(lat_rad[j])**2
                        - 10.0 * np.log(p_levels[k]) * np.cos(lat_rad[j])**2)

            # Relaxation timescale (days)
            k_T = np.zeros((nlat, nlev))
            for j in range(nlat):
                for k in range(nlev):
                    sigma = p_levels[k]
                    k_a = 1.0 / 40.0  # 40-day timescale
                    k_s = 1.0 / 4.0   # 4-day timescale near surface
                    k_T[j, k] = k_a + (k_s - k_a) * max(0, (sigma - 0.7) / 0.3) * np.cos(lat_rad[j])**4

            # Initialize temperature
            T = T_eq.copy() + 5.0 * np.random.randn(nlat, nlev)

            # Simple Newtonian relaxation
            T_history = []
            for step in range(n_steps):
                dT = -k_T * (T - T_eq) * dt / 86400.0
                T = T + dT

                if step % (n_steps // min(n_days, 30)) == 0:
                    T_history.append(T.copy())

            # Zonal mean temperature plot
            T_final = T_history[-1] if T_history else T
            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.contourf(lats, p_levels, T_final.T, levels=20, cmap="RdYlBu_r")
            ax.set_xlabel("Latitude")
            ax.set_ylabel("Sigma level")
            ax.set_title("Held-Suarez: Zonal Mean Temperature (K)")
            ax.invert_yaxis()
            plt.colorbar(im, label="Temperature (K)")
            plt.savefig("held_suarez_temperature.png", dpi=100, bbox_inches="tight")
            plt.close()

            # Metrics
            equator_idx = nlat // 2
            pole_idx = 0
            equator_surface_T = float(T_final[equator_idx, -1])
            pole_surface_T = float(T_final[pole_idx, -1])
            meridional_gradient = equator_surface_T - pole_surface_T
            global_mean_T = float(np.mean(T_final))

            result = {{
                "global_mean_T_K": round(global_mean_T, 2),
                "equator_surface_T_K": round(equator_surface_T, 2),
                "pole_surface_T_K": round(pole_surface_T, 2),
                "meridional_gradient_K": round(meridional_gradient, 2),
                "n_days": n_days,
                "n_steps": n_steps,
                "summary": f"Held-Suarez: global_T={{global_mean_T:.1f}}K, grad={{meridional_gradient:.1f}}K",
            }}
            with open("results.json", "w") as f:
                json.dump(result, f)
            print(json.dumps(result, indent=2))
        """)
