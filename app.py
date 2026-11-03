
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="TrussVerify", page_icon="🏗️", layout="wide")
st.title("🏗️ TrussVerify")
st.caption("Independent 2D Truss Analysis and Verification")

# SAMPLE MODEL — replace with imported STAAD.Pro geometry later
nodes = {
    "A": (0.0, 0.0),
    "B": (10.0, 8.0),
    "C": (20.0, 0.0),
}
members = [("A", "B"), ("B", "C"), ("A", "C")]
node_names = list(nodes)
node_index = {name: i for i, name in enumerate(node_names)}

st.sidebar.header("Material and Section")
material = st.sidebar.selectbox(
    "Choose Material",
    ["Steel", "Aluminum", "Timber", "Custom"]
)
# Representative elastic modulus values in psi
material_properties = {
    "Steel": 29_000_000.0,
    "Aluminum": 10_000_000.0,
    "Timber": 1_200_000.0,
    "Custom": 1_000_000.0
}
default_E_psi = material_properties[material]
# Allow E to be entered in psi or ksi
E_unit = st.sidebar.selectbox(
    "Modulus Input Unit",
    ["psi", "ksi"]
)
default_E = (
    default_E_psi
    if E_unit == "psi"
    else default_E_psi / 1000
)
E_input = st.sidebar.number_input(
    f"Modulus of Elasticity, E ({E_unit})",
    min_value=1.0,
    value=float(default_E),
    key=f"E_{material}_{E_unit}"
)
# Convert internally to ksi
E = E_input / 1000 if E_unit == "psi" else E_input
A = st.sidebar.number_input(
    "Cross-Sectional Area, A (in²)",
    min_value=0.01,
    value=5.25 if material == "Timber" else 2.0,
    key=f"A_{material}"
)
EA = E * A
st.sidebar.metric(
    "Modulus E (ksi)",
    f"{E:,.2f}"
)
st.sidebar.metric(
    "Axial Stiffness EA (kip)",
    f"{EA:,.2f}"
)
st.sidebar.caption(
    "All calculations use kips and inches. "
    "Material values are illustrative and editable. "
    "All members currently share the selected properties."
)

st.sidebar.header("Actual Joint Loads")
load_joint = st.sidebar.selectbox("Loaded joint", node_names, index=1)
Fx = st.sidebar.number_input("Horizontal force Fx (kip)", value=0.0)
Fy = st.sidebar.number_input("Vertical force Fy (kip)", value=-10.0)

st.sidebar.header("Virtual Work")
virtual_joint = st.sidebar.selectbox(
    "Displacement at joint", node_names, index=1
)
virtual_direction = st.sidebar.selectbox(
    "Displacement direction", ["Vertical (Y)", "Horizontal (X)"]
)

# All coordinates are in feet, converted to inches for stiffness.
coords = np.array([nodes[n] for n in node_names]) * 12.0
ndof = 2 * len(nodes)
K = np.zeros((ndof, ndof))
member_data = []

for start, end in members:
    i = node_index[start]
    j = node_index[end]
    dx, dy = coords[j] - coords[i]
    L = np.hypot(dx, dy)
    c, s = dx / L, dy / L

    k = (E * A / L) * np.array([
        [c*c, c*s, -c*c, -c*s],
        [c*s, s*s, -c*s, -s*s],
        [-c*c, -c*s, c*c, c*s],
        [-c*s, -s*s, c*s, s*s],
    ])
    dofs = [2*i, 2*i+1, 2*j, 2*j+1]
    K[np.ix_(dofs, dofs)] += k
    member_data.append((start, end, L, c, s, dofs))

# A = pin (X,Y restrained); C = roller (Y restrained)
restrained = [0, 1, 5]
free = [i for i in range(ndof) if i not in restrained]

def solve(force_vector):
    displacement = np.zeros(ndof)
    displacement[free] = np.linalg.solve(
        K[np.ix_(free, free)], force_vector[free]
    )
    reactions = K @ displacement - force_vector

    axial_forces = []
    for start, end, L, c, s, dofs in member_data:
        u = displacement[dofs]
        N = (E * A / L) * np.dot(
            np.array([-c, -s, c, s]), u
        )
        axial_forces.append(N)

    return displacement, reactions, np.array(axial_forces)

F = np.zeros(ndof)
loaded_idx = node_index[load_joint]
F[2*loaded_idx] = Fx
F[2*loaded_idx+1] = Fy

unit = np.zeros(ndof)
virtual_idx = node_index[virtual_joint]
direction_idx = 1 if virtual_direction.startswith("Vertical") else 0
unit[2*virtual_idx+direction_idx] = 1.0

try:
    displacement, reactions, N = solve(F)
    virtual_displacement, _, n = solve(unit)
except np.linalg.LinAlgError:
    st.error("The truss is unstable or its stiffness matrix is singular.")
    st.stop()

# Drawing
st.subheader("Truss Diagram")
fig, ax = plt.subplots(figsize=(10, 4))
for index, (start, end) in enumerate(members):
    x1, y1 = nodes[start]
    x2, y2 = nodes[end]
    color = "red" if N[index] < -1e-8 else (
        "blue" if N[index] > 1e-8 else "gray"
    )
    ax.plot([x1, x2], [y1, y2], color=color, linewidth=3)
    ax.text(
        (x1+x2)/2, (y1+y2)/2,
        f"{start}{end}: {N[index]:.2f}",
        fontsize=9
    )

for name, (x, y) in nodes.items():
    ax.scatter(x, y, color="black", zorder=5)
    ax.annotate(name, (x, y), xytext=(5, 7),
                textcoords="offset points")


# Draw the actual applied load arrow
x, y = nodes[load_joint]

if abs(Fx) > 1e-10 or abs(Fy) > 1e-10:
    magnitude = np.hypot(Fx, Fy)

    # Arrow direction follows the applied force
    ux = Fx / magnitude
    uy = Fy / magnitude

    arrow_length = 2.0

    # Start behind the joint and point toward the joint
    start_x = x - arrow_length * ux
    start_y = y - arrow_length * uy

    ax.annotate(
        "",
        xy=(x, y),
        xytext=(start_x, start_y),
        arrowprops=dict(
            arrowstyle="-|>",
            color="green",
            lw=2.5,
            mutation_scale=18
        )
    )
    ax.text(
        start_x,
        start_y,
        f"{magnitude:g} kip",
        color="green",
        fontsize=9
    )

ax.set_aspect("equal", adjustable="datalim")
ax.set_xlabel("X (ft)")
ax.set_ylabel("Y (ft)")
ax.grid(alpha=0.25)
ax.set_title("Blue = Tension | Red = Compression | Green = Load")
st.pyplot(fig)
plt.close(fig)
tab1, tab2, tab3, tab4 = st.tabs([
    "Reactions", "Member Forces",
    "Displacements", "Virtual Work"
])

with tab1:
    st.subheader("Support Reactions")
    st.dataframe(pd.DataFrame([
        {"Support": "A", "Direction": "X", "Reaction (kip)": reactions[0]},
        {"Support": "A", "Direction": "Y", "Reaction (kip)": reactions[1]},
        {"Support": "C", "Direction": "Y", "Reaction (kip)": reactions[5]},
    ]), hide_index=True, use_container_width=True)
    st.write("Equilibrium checks:")
    st.write(f"ΣFx = {Fx + reactions[0]:.6f} kip")
    st.write(f"ΣFy = {Fy + reactions[1] + reactions[5]:.6f} kip")

with tab2:
    st.subheader("Axial Member Forces")
    rows = []
    for (start, end), force in zip(members, N):
        rows.append({
            "Member": f"{start}-{end}",
            "Force (kip)": force,
            "Classification": (
                "Tension" if force > 1e-8 else
                "Compression" if force < -1e-8 else "Zero"
            )
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True,
                 use_container_width=True)
    st.caption("Positive axial force = tension; negative = compression.")
with tab3:
    st.subheader("Joint Displacements")
    rows = []
    for name, idx in node_index.items():
        rows.append({
            "Joint": name,
            "Ux (in)": displacement[2*idx],
            "Uy (in)": displacement[2*idx+1]
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True,
                 use_container_width=True)
with tab4:
    st.subheader("Method of Virtual Work")
    st.latex(r"\delta = \sum_i \frac{N_i n_i L_i}{A_i E_i}")

    rows = []
    for index, (start, end, L, c, s, dofs) in enumerate(member_data):
        contribution = N[index] * n[index] * L / (A * E)
        rows.append({
            "Member": f"{start}-{end}",
            "N (kip)": N[index],
            "n (kip/unit load)": n[index],
            "L (in)": L,
            "A (in²)": A,
            "E (ksi)": E,
            "NnL/AE (in)": contribution
        })   
    # Create virtual work calculation table
    df = pd.DataFrame(rows)
    # Calculate the total displacement by summing
    # each member's individual contribution
    total = df["NnL/AE (in)"].sum()
    # Add a TOTAL row at the bottom
    total_row = {
        "Member": "TOTAL",
        "N (kip)": None,
        "n (kip/unit load)": None,
        "L (in)": None,
        "A (in²)": None,
        "E (ksi)": None,
        "NnL/AE (in)": total
    }
    df_display = pd.concat(
        [df, pd.DataFrame([total_row])],
        ignore_index=True
    )
    # Display organized calculation table
    st.dataframe(
        df_display,
        hide_index=True,
        use_container_width=True,
        column_config={
            "N (kip)": st.column_config.NumberColumn(
                "N (kip)", format="%.4f"
            ),
            "n (kip/unit load)": st.column_config.NumberColumn(
                "n (kip/unit load)", format="%.4f"
            ),
            "L (in)": st.column_config.NumberColumn(
                "L (in)", format="%.3f"
            ),
            "A (in²)": st.column_config.NumberColumn(
                "A (in²)", format="%.3f"
            ),
            "E (ksi)": st.column_config.NumberColumn(
                "E (ksi)", format="%.0f"
            ),
            "NnL/AE (in)": st.column_config.NumberColumn(
                "NnL/AE (in)", format="%.6f"
            )
        }
    )
    # Compare virtual work against stiffness method
    direct = displacement[2 * virtual_idx + direction_idx]
    difference = abs(total - direct)
    st.divider()
    st.subheader("Virtual Work Verification")
    st.latex(r"\delta = \sum_i \frac{N_i n_i L_i}{A_i E_i}")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Virtual Work",
            f"{total:.6f} in"
        )
    with col2:
        st.metric(
            "Stiffness Method",
            f"{direct:.6f} in"
        )
    with col3:
        st.metric(
            "Absolute Difference",
            f"{difference:.9f} in"
        )
    if np.isclose(total, direct, atol=1e-8, rtol=1e-6):
        st.success(
            "PASS: Virtual work and stiffness method agree."
        )
    else:
        st.error(
            "CHECK: The two displacement results do not agree."
        )
st.warning(
    "This is a sample 2D truss solver. STAAD.Pro import, "
    "editable geometry, detailed hand-calculation steps, "
    "and independent validation are still under development."
)
