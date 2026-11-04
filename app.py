
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="TrussVerify", page_icon="🏗️", layout="wide")
st.title("🏗️ TrussVerify")
st.caption("Independent 2D Truss Analysis and Verification")


# CUSTOM TRUSS GEOMETRY EDITOR
st.subheader("Custom Truss Geometry")
st.caption("Enter joint coordinates in inches.")

default_joints = pd.DataFrame([
    {"Joint": "A", "X (in)": 0.0, "Y (in)": 0.0},
    {"Joint": "B", "X (in)": 120.0, "Y (in)": 96.0},
    {"Joint": "C", "X (in)": 240.0, "Y (in)": 0.0},
])

default_members = pd.DataFrame([
    {"Start": "A", "End": "B"},
    {"Start": "B", "End": "C"},
    {"Start": "A", "End": "C"},
])

st.markdown("#### Joints")
joints_df = st.data_editor(
    default_joints,
    num_rows="dynamic",
    hide_index=True,
    use_container_width=True,
    key="joint_editor"
)

st.markdown("#### Members")
members_df = st.data_editor(
    default_members,
    num_rows="dynamic",
    hide_index=True,
    use_container_width=True,
    key="member_editor"
)

# Validate the entered geometry before analysis
try:
    if joints_df.empty or members_df.empty:
        raise ValueError("At least one joint and member are required.")

    if joints_df[["Joint", "X (in)", "Y (in)"]].isnull().any().any():
        raise ValueError("Every joint needs a name and X/Y coordinates.")

    if members_df[["Start", "End"]].isnull().any().any():
        raise ValueError("Every member needs a start and end joint.")

    joint_ids = joints_df["Joint"].astype(str).str.strip()

    if joint_ids.duplicated().any():
        raise ValueError("Joint names must be unique.")

    nodes = {
        str(row["Joint"]).strip(): (
            float(row["X (in)"]),
            float(row["Y (in)"])
        )
        for _, row in joints_df.iterrows()
    }

    members = [
        (str(row["Start"]).strip(), str(row["End"]).strip())
        for _, row in members_df.iterrows()
    ]

    for start, end in members:
        if start not in nodes or end not in nodes:
            raise ValueError(
                f"Member {start}-{end} references a missing joint."
            )

        if start == end or np.allclose(nodes[start], nodes[end]):
            raise ValueError(
                f"Member {start}-{end} has zero length."
            )

except (ValueError, TypeError) as error:
    st.error(f"Invalid truss geometry: {error}")
    st.stop()

node_names = list(nodes)
node_index = {name: i for i, name in enumerate(node_names)}

# CUSTOM SUPPORT EDITOR
st.subheader("Support Editor")
st.caption(
    "Choose which joints are restrained. "
    "Pin = X and Y; Horizontal roller = Y; "
    "Vertical roller = X."
)

default_supports = pd.DataFrame([
    {"Joint": "A", "Support Type": "Pin"},
    {"Joint": "C", "Support Type": "Horizontal Roller"},
])

supports_df = st.data_editor(
    default_supports,
    num_rows="dynamic",
    hide_index=True,
    use_container_width=True,
    key="support_editor",
    column_config={
        "Joint": st.column_config.SelectboxColumn(
            "Joint",
            options=node_names,
            required=True
        ),
        "Support Type": st.column_config.SelectboxColumn(
            "Support Type",
            options=[
                "Pin",
                "Horizontal Roller",
                "Vertical Roller",
                "Fixed",
                "Remove Support"
        ],
            required=True
        )
    }
)

# Convert support selections to restrained DOFs
restrained = []

try:
    if supports_df.empty:
        raise ValueError("At least one support is required.")

    if supports_df[["Joint", "Support Type"]].isnull().any().any():
        raise ValueError("Every support needs a joint and type.")

    if supports_df["Joint"].duplicated().any():
        raise ValueError("Each joint can have only one support.")

    for _, row in supports_df.iterrows():
        joint = str(row["Joint"])
        support_type = str(row["Support Type"])

        if joint not in node_index:
            raise ValueError(f"Unknown support joint: {joint}")

        idx = node_index[joint]

        if support_type in ["Pin", "Fixed"]:
            restrained.extend([2 * idx, 2 * idx + 1])

        elif support_type == "Horizontal Roller":
            restrained.append(2 * idx + 1)

        elif support_type == "Vertical Roller":
            restrained.append(2 * idx)

        elif support_type == "Remove Support":
            pass  # No restrained directions at this joint

        else:
            raise ValueError(
                f"Unknown support type: {support_type}"
            )

    restrained = sorted(set(restrained))

except ValueError as error:
    st.error(f"Invalid support configuration: {error}")
    st.stop()


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


# MULTIPLE JOINT LOAD EDITOR
st.subheader("Joint Load Editor")
st.caption(
    "Apply horizontal and vertical loads at any joint. "
    "Positive Fx = right, positive Fy = up. "
    "Negative Fx = left, negative Fy = down. "
    "All forces are in kips."
)

default_loads = pd.DataFrame([
    {"Joint": "B", "Fx (kip)": 0.0, "Fy (kip)": -10.0}
])

loads_df = st.data_editor(
    default_loads,
    num_rows="dynamic",
    hide_index=True,
    width="stretch",
    key="load_editor",
    column_config={
        "Joint": st.column_config.SelectboxColumn(
            "Joint",
            options=node_names,
            required=True
        ),
        "Fx (kip)": st.column_config.NumberColumn(
            "Fx (kip)",
            required=True,
            format="%.3f"
        ),
        "Fy (kip)": st.column_config.NumberColumn(
            "Fy (kip)",
            required=True,
            format="%.3f"
        )
    }
)

# Validate and collect all applied loads
try:
    if loads_df.isnull().any().any():
        raise ValueError("Every load needs a joint, Fx, and Fy.")

    for _, row in loads_df.iterrows():
        if str(row["Joint"]) not in node_index:
            raise ValueError(
                f"Load references an unknown joint: {row['Joint']}"
            )

except ValueError as error:
    st.error(f"Invalid load configuration: {error}")
    st.stop()

# Totals for force equilibrium checks
Fx = float(loads_df["Fx (kip)"].sum())
Fy = float(loads_df["Fy (kip)"].sum())


st.sidebar.header("Virtual Work")
virtual_joint = st.sidebar.selectbox(
    "Displacement at joint", node_names, index=1
)
virtual_direction = st.sidebar.selectbox(
    "Displacement direction", ["Vertical (Y)", "Horizontal (X)"]
)

# All coordinates are in feet, converted to inches for stiffness.
coords = np.array([nodes[n] for n in node_names])
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

# Use the restraints selected in the Support Editor
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


# Build the global force vector from ALL joint loads
F = np.zeros(ndof)

for _, row in loads_df.iterrows():
    joint = str(row["Joint"])
    idx = node_index[joint]

    F[2 * idx] += float(row["Fx (kip)"])
    F[2 * idx + 1] += float(row["Fy (kip)"])


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

# DRAW ALL APPLIED JOINT LOADS
# Geometry is in inches; forces are in kips.

model_width = max(
    max(x for x, y in nodes.values())
    - min(x for x, y in nodes.values()),
    1.0
)

model_height = max(
    max(y for x, y in nodes.values())
    - min(y for x, y in nodes.values()),
    1.0
)

arrow_length = 0.15 * max(model_width, model_height)

for _, row in loads_df.iterrows():
    joint = str(row["Joint"])
    fx = float(row["Fx (kip)"])
    fy = float(row["Fy (kip)"])

    magnitude = np.hypot(fx, fy)

    if magnitude < 1e-10:
        continue

    x, y = nodes[joint]

    # Unit vector in the applied load direction
    ux = fx / magnitude
    uy = fy / magnitude

    # Arrow points toward the loaded joint
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

# DRAW SUPPORT SYMBOLS
# Symbols update automatically from the Support Editor.

symbol_size = 0.035 * max(model_width, model_height)

for _, row in supports_df.iterrows():
    joint = str(row["Joint"])
    support_type = str(row["Support Type"])

    if support_type == "Remove Support":
        continue

    x, y = nodes[joint]

    if support_type in ["Pin", "Fixed"]:
        # Triangle below the supported joint
        ax.scatter(
            x, y,
            marker="^",
            s=230,
            facecolor="lightgray",
            edgecolor="black",
            linewidth=1.5,
            zorder=4
        )

    elif support_type == "Horizontal Roller":
        # Roller beneath the joint
        ax.scatter(
            x, y - symbol_size,
            marker="o",
            s=100,
            facecolor="white",
            edgecolor="black",
            linewidth=1.5,
            zorder=4
        )

        ax.plot(
            [x - symbol_size, x + symbol_size],
            [y - 2 * symbol_size, y - 2 * symbol_size],
            color="black",
            linewidth=2
        )

    elif support_type == "Vertical Roller":
        # Roller beside a vertical wall
        ax.scatter(
            x - symbol_size, y,
            marker="o",
            s=100,
            facecolor="white",
            edgecolor="black",
            linewidth=1.5,
            zorder=4
        )

        ax.plot(
            [x - 2 * symbol_size, x - 2 * symbol_size],
            [y - symbol_size, y + symbol_size],
            color="black",
            linewidth=2
        )



# Leave room around the truss for supports and load arrows
x_values = [p[0] for p in nodes.values()]
y_values = [p[1] for p in nodes.values()]

padding = 0.25 * max(model_width, model_height)

ax.set_xlim(min(x_values) - padding, max(x_values) + padding)
ax.set_ylim(
    min(y_values) - padding,
    max(y_values) + padding + arrow_length
)

ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("X (in)")
ax.set_ylabel("Y (in)")

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
    reaction_rows = []
    for dof in restrained:
        joint = node_names[dof // 2]
        direction = "X" if dof % 2 == 0 else "Y"
        reaction_rows.append({
            "Support": joint,
            "Direction": direction,
            "Reaction (kip)": reactions[dof]
        })
    st.dataframe(
        pd.DataFrame(reaction_rows),
        hide_index=True,
        use_container_width=True
    )
    
    # Check global equilibrium
    total_fx = float(F[0::2].sum() + reactions[0::2].sum())
    total_fy = float(F[1::2].sum() + reactions[1::2].sum())

    total_moment = 0.0

    for joint, idx in node_index.items():
        x, y = nodes[joint]

        # Moment = x*Fy - y*Fx (counterclockwise positive)
        total_moment += (
            x * (F[2 * idx + 1] + reactions[2 * idx + 1])
            - y * (F[2 * idx] + reactions[2 * idx])
        )

    st.write("Equilibrium checks:")
    st.write(f"ΣFx = {total_fx:.6f} kip")
    st.write(f"ΣFy = {total_fy:.6f} kip")
    st.write(f"ΣM = {total_moment:.6f} kip-in")

    if (
        abs(total_fx) < 1e-6
        and abs(total_fy) < 1e-6
        and abs(total_moment) < 1e-4
    ):
        st.success("PASS: Global equilibrium is satisfied.")
    else:
        st.error("CHECK: Global equilibrium is not satisfied.")


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
