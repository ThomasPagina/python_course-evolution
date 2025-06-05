"""
Wine Fermentation Simulation with Torulaspora delbrueckii Yeast

This script simulates yeast growth, sugar consumption,
and alcohol production during wine fermentation. It includes five
different scenarios:

A) High initial sugar concentration
B) Low initial sugar concentration
C) Hypothetical infinite alcohol tolerance (all sugar eventually consumed)
D) Addition of alcohol at a specified time
E) Addition of sugar at a specified time

In this version, yeast stops consuming sugar once the alcohol concentration
reaches or exceeds its tolerance threshold. Additionally, when sugar is
depleted, yeast experiences a maintenance‐driven death rate (starvation).

Results are plotted in a multi‐axis Matplotlib figure for each scenario:
- Yeast population over time
- Sugar concentration over time
- Alcohol concentration over time
- Alcohol tolerance threshold (horizontal line)
"""

import numpy as np
import matplotlib.pyplot as plt


def simulate_fermentation(
    t_max: float,
    dt: float,
    yeast0: float,
    sugar0: float,
    alcohol0: float,
    tol_alcohol: float,
    mu_max: float,
    Ks: float,
    yield_alcohol: float,
    consumption_coeff: float,
    death_rate: float,
    maintenance_rate: float,
    sugar_add_time: float = None,
    sugar_add_amount: float = 0.0,
    alcohol_add_time: float = None,
    alcohol_add_amount: float = 0.0,
):
    """
    Simulate the fermentation process over time.

    Parameters:
    - t_max: Total simulation time (hours)
    - dt: Time step increment (hours)
    - yeast0: Initial yeast population (arbitrary units)
    - sugar0: Initial sugar concentration (g/L)
    - alcohol0: Initial alcohol concentration (% ABV-equivalent)
    - tol_alcohol: Alcohol tolerance threshold (% ABV-equivalent)
    - mu_max: Maximum yeast growth rate (1/hour)
    - Ks: Half-saturation constant for sugar uptake (g/L)
    - yield_alcohol: Alcohol produced (g) per gram of sugar consumed
    - consumption_coeff: Sugar consumption rate coefficient
    - death_rate: Yeast death rate coefficient when alcohol ≥ tol_alcohol (1/hour)
    - maintenance_rate: Basal yeast death rate when sugar is zero (1/hour)
    - sugar_add_time: Time to add sugar (hours; None if no addition)
    - sugar_add_amount: Amount of sugar added at sugar_add_time (g/L)
    - alcohol_add_time: Time to add alcohol (hours; None if no addition)
    - alcohol_add_amount: Amount of alcohol added at alcohol_add_time (% ABV)

    Returns:
    - times: 1D numpy array of time points
    - yeast: 1D numpy array of yeast population over time
    - sugar: 1D numpy array of sugar concentration over time
    - alcohol: 1D numpy array of alcohol concentration over time
    """
    n_steps = int(t_max / dt) + 1
    times = np.linspace(0, t_max, n_steps)

    # Initialize state arrays
    yeast = np.zeros(n_steps)
    sugar = np.zeros(n_steps)
    alcohol = np.zeros(n_steps)

    # Set initial conditions
    yeast[0] = yeast0
    sugar[0] = sugar0
    alcohol[0] = alcohol0

    for i in range(1, n_steps):
        t = times[i]

        # Apply sugar addition if specified (at the closest timestep)
        if (sugar_add_time is not None) and (np.isclose(t, sugar_add_time, atol=dt / 2)):
            sugar[i - 1] += sugar_add_amount

        # Apply alcohol addition if specified (at the closest timestep)
        if (alcohol_add_time is not None) and (np.isclose(t, alcohol_add_time, atol=dt / 2)):
            alcohol[i - 1] += alcohol_add_amount

        Y_prev = yeast[i - 1]
        S_prev = max(sugar[i - 1], 0.0)   # ensure sugar is non-negative
        A_prev = max(alcohol[i - 1], 0.0) # ensure alcohol is non-negative

        # === Sugar uptake only if alcohol < tolerance ===
        if A_prev < tol_alcohol:
            sugar_uptake = consumption_coeff * Y_prev * (S_prev / (Ks + S_prev)) * dt
            sugar_uptake = min(sugar_uptake, S_prev)
        else:
            sugar_uptake = 0.0
        # ===========================================================

        # Update sugar and alcohol concentrations
        S_new = S_prev - sugar_uptake
        A_new = A_prev + yield_alcohol * sugar_uptake

        # Determine yeast growth or death
        if A_prev < tol_alcohol:
            # Monod growth term
            growth_rate = mu_max * (S_prev / (Ks + S_prev))
            if S_prev > 0:
                # Net growth minus maintenance
                net_growth = growth_rate - maintenance_rate
            else:
                # If no sugar remains, growth_rate = 0 → net death by maintenance
                net_growth = -maintenance_rate

            dY = net_growth * Y_prev * dt
        else:
            # Yeast dies at death_rate if alcohol ≥ tolerance
            dY = -death_rate * Y_prev * dt

        Y_new = max(Y_prev + dY, 0.0)  # prevent negative population

        # Store new values
        yeast[i] = Y_new
        sugar[i] = S_new
        alcohol[i] = A_new

    return times, yeast, sugar, alcohol


def plot_multi_axis_standard(
    times: np.ndarray,
    yeast: np.ndarray,
    sugar: np.ndarray,
    alcohol: np.ndarray,
    tol_alcohol: float,
    scenario_label: str,
):
    """
    Plot yeast, sugar, and alcohol curves on a single figure with three Y-axes.

    - Left y-axis (green): Yeast Population
    - Right y-axis #1 (blue): Sugar Concentration
    - Right y-axis #2 (red, offset): Alcohol Concentration
    - Horizontal dotted line for alcohol tolerance threshold
    """
    # Create figure and first axis
    fig, ax1 = plt.subplots(figsize=(8, 5))
    fig.subplots_adjust(right=0.75)  # leave room on the right for the third axis

    # Create second axis sharing the same x-axis (for sugar)
    ax2 = ax1.twinx()

    # Create third axis sharing the same x-axis (for alcohol) and offset it
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.2))
    ax3.spines["right"].set_visible(True)

    # Plot the curves
    p1, = ax1.plot(times, yeast, color="green", linestyle="-", label="Yeast Population")
    p2, = ax2.plot(times, sugar, color="blue", linestyle="--", label="Sugar (g/L)")
    p3, = ax3.plot(times, alcohol, color="red", linestyle="-", label="Alcohol (% ABV)")

    # Plot alcohol tolerance threshold on the third axis
    p4, = ax3.plot(
        times,
        np.full_like(times, tol_alcohol),
        color="red",
        linestyle=":",
        linewidth=1.0,
        label="Alcohol Tolerance Threshold",
    )

    # Set axis labels
    ax1.set_xlabel("Time (hours)")
    ax1.set_ylabel("Yeast Population (arb. units)", color="green")
    ax2.set_ylabel("Sugar Concentration (g/L)", color="blue")
    ax3.set_ylabel("Alcohol Concentration (% ABV)", color="red")

    # Set title
    plt.title(scenario_label)

    # Color the tick labels to match the curves
    ax1.tick_params(axis="y", colors="green")
    ax2.tick_params(axis="y", colors="blue")
    ax3.tick_params(axis="y", colors="red")

    # Combine legends from all plots
    lines = [p1, p2, p3, p4]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper right")

    ax1.grid(True)


def run_all_scenarios():
    """
    Run all five scenarios and generate corresponding plots.
    """
    # Common simulation parameters
    t_max = 120.0             # total hours to simulate
    dt = 0.1                  # time step (hours)
    yeast0 = 1e6              # initial yeast population (arbitrary units)
    mu_max = 0.4              # maximum growth rate (1/hour)
    Ks = 10.0                 # half-saturation constant for sugar (g/L)
    yield_alcohol = 0.51      # (g alcohol) per (g sugar) consumed
    consumption_coeff = 1e-7  # sugar consumption rate coefficient
    death_rate = 0.1          # yeast death rate coefficient when alcohol ≥ tolerance (1/hour)
    maintenance_rate = 0.02   # basal death rate when sugar is depleted (1/hour)

    # Scenario A: High initial sugar
    sugar0_A = 200.0   # g/L
    alcohol0_A = 0.0   # % ABV
    tol_alcohol_A = 12.0  # % ABV
    times_A, yeast_A, sugar_A, alcohol_A = simulate_fermentation(
        t_max,
        dt,
        yeast0,
        sugar0_A,
        alcohol0_A,
        tol_alcohol_A,
        mu_max,
        Ks,
        yield_alcohol,
        consumption_coeff,
        death_rate,
        maintenance_rate,
    )
    plot_multi_axis_standard(
        times_A,
        yeast_A,
        sugar_A,
        alcohol_A,
        tol_alcohol_A,
        "Scenario A: High Initial Sugar",
    )

    # Scenario B: Low initial sugar
    sugar0_B = 100.0   # g/L
    alcohol0_B = 0.0   # % ABV
    tol_alcohol_B = 12.0  # % ABV
    times_B, yeast_B, sugar_B, alcohol_B = simulate_fermentation(
        t_max,
        dt,
        yeast0,
        sugar0_B,
        alcohol0_B,
        tol_alcohol_B,
        mu_max,
        Ks,
        yield_alcohol,
        consumption_coeff,
        death_rate,
        maintenance_rate,
    )
    plot_multi_axis_standard(
        times_B,
        yeast_B,
        sugar_B,
        alcohol_B,
        tol_alcohol_B,
        "Scenario B: Low Initial Sugar",
    )

    # Scenario C: Infinite alcohol tolerance (all sugar eventually consumed)
    sugar0_C = 200.0   # g/L
    alcohol0_C = 0.0   # % ABV
    tol_alcohol_C = 1e6  # effectively infinite tolerance
    times_C, yeast_C, sugar_C, alcohol_C = simulate_fermentation(
        t_max,
        dt,
        yeast0,
        sugar0_C,
        alcohol0_C,
        tol_alcohol_C,
        mu_max,
        Ks,
        yield_alcohol,
        consumption_coeff,
        death_rate,
        maintenance_rate,
    )
    plot_multi_axis_standard(
        times_C,
        yeast_C,
        sugar_C,
        alcohol_C,
        tol_alcohol_C,
        "Scenario C: Infinite Alcohol Tolerance",
    )

    # Scenario D: Addition of alcohol at mid-fermentation (t = 50 h)
    sugar0_D = 200.0
    alcohol0_D = 0.0
    tol_alcohol_D = 12.0
    alcohol_add_time_D = 50.0    # hours
    alcohol_add_amount_D = 5.0   # % ABV added
    times_D, yeast_D, sugar_D, alcohol_D = simulate_fermentation(
        t_max,
        dt,
        yeast0,
        sugar0_D,
        alcohol0_D,
        tol_alcohol_D,
        mu_max,
        Ks,
        yield_alcohol,
        consumption_coeff,
        death_rate,
        maintenance_rate,
        sugar_add_time=None,
        sugar_add_amount=0.0,
        alcohol_add_time=alcohol_add_time_D,
        alcohol_add_amount=alcohol_add_amount_D,
    )
    plot_multi_axis_standard(
        times_D,
        yeast_D,
        sugar_D,
        alcohol_D,
        tol_alcohol_D,
        "Scenario D: Addition of Alcohol at t=50h",
    )

    # Scenario E: Addition of sugar at mid-fermentation (t = 50 h)
    sugar0_E = 100.0
    alcohol0_E = 0.0
    tol_alcohol_E = 12.0
    sugar_add_time_E = 50.0   # hours
    sugar_add_amount_E = 50.0 # g/L added
    times_E, yeast_E, sugar_E, alcohol_E = simulate_fermentation(
        t_max,
        dt,
        yeast0,
        sugar0_E,
        alcohol0_E,
        tol_alcohol_E,
        mu_max,
        Ks,
        yield_alcohol,
        consumption_coeff,
        death_rate,
        maintenance_rate,
        sugar_add_time=sugar_add_time_E,
        sugar_add_amount=sugar_add_amount_E,
        alcohol_add_time=None,
        alcohol_add_amount=0.0,
    )
    plot_multi_axis_standard(
        times_E,
        yeast_E,
        sugar_E,
        alcohol_E,
        tol_alcohol_E,
        "Scenario E: Addition of Sugar at t=50h",
    )

    # Display all figures
    plt.show()


if __name__ == "__main__":
    run_all_scenarios()
