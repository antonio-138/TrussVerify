
import streamlit as st
import numpy as np
import pandas as pd
import re
import hashlib
import matplotlib.pyplot as plt

st.set_page_config(page_title="TrussVerify", page_icon="🏗️", layout="wide")
st.title("🏗️ TrussVerify")
st.caption("Independent 2D Truss Analysis and Verification")

# STAAD.PRO FILE UPLOADER
st.subheader("Import STAAD.Pro Model")
uploaded_std = st.file_uploader(
    "Upload a STAAD.Pro .std file", type=["std"],
    help="Import compatible 2D XY trusses; supported units are converted to inches and kips."
)

imported_nodes = {}
imported_members = []
imported_supports = []
imported_loads = []
imported_properties = {}


def parse_staad_truss(std_text):
    """Parse a documented 2D, pin-jointed subset of STAAD.Pro input.

    Missing area/load information stays missing, rather than being invented.
    Unsupported structural commands raise an explicit error.
    """
    lines = [x.split("*", 1)[0].strip().upper() for x in std_text.splitlines()]
    lines = [x for x in lines if x]
    units = next((x for x in lines if x.startswith("UNIT ")), None)
    if not units:
        raise ValueError("Missing UNIT command; cannot safely interpret dimensions and forces.")
    up = units.split()
    length_factors = {"INCHES": 1, "INCH": 1, "FEET": 12, "FT": 12,
                      "CM": 1/2.54, "METER": 100/2.54, "METERS": 100/2.54,
                      "MM": 1/25.4}
    force_factors = {"KIP": 1, "KIPS": 1, "POUND": .001, "LB": .001,
                     "KN": .2248089431, "NEWTON": .0002248089431}
    if len(up) != 3 or up[1] not in length_factors or up[2] not in force_factors:
        raise ValueError(f"Unsupported units: {units}")
    lf, ff = length_factors[up[1]], force_factors[up[2]]
    heads = ("JOINT COORDINATES", "MEMBER INCIDENCES", "ELEMENT INCIDENCES",
             "DEFINE ", "END DEFINE", "MEMBER PROPERTY", "CONSTANTS", "SUPPORTS",
             "MEMBER TRUSS", "MEMBER RELEASE", "MEMBER OFFSET", "LOAD ",
             "LOAD COMB", "JOINT LOAD", "MEMBER LOAD", "ELEMENT LOAD", "SELFWEIGHT",
             "FLOOR LOAD", "REPEAT LOAD", "SUPPORT DISPLACEMENT", "PERFORM ANALYSIS",
             "FINISH", "PRINT", "PARAMETER", "CHECK CODE", "UNIT ")
    def section(name, required=False):
        indices = [i for i, line in enumerate(lines) if line == name]
        if not indices:
            if required:
                raise ValueError(f"Missing {name} section.")
            return []
        if len(indices) > 1:
            raise ValueError(f"Multiple {name} sections are not supported.")
        result = []
        for line in lines[indices[0]+1:]:
            if line.startswith(heads):
                break
            result.extend(x.strip() for x in line.split(';') if x.strip())
        return result
    def expand(tokens):
        result = []
        i = 0
        while i < len(tokens):
            t = tokens[i]
            if t == 'TO' or t == 'BY':
                raise ValueError(f"Malformed joint/member selection: {' '.join(tokens)}")
            if not t.isdigit():
                raise ValueError(f"Unsupported joint/member selection: {' '.join(tokens)}")
            if i + 2 < len(tokens) and tokens[i+1] == 'TO':
                start, end = int(t), int(tokens[i+2])
                step = 1
                i += 3
                if i + 1 < len(tokens) and tokens[i] == 'BY':
                    step = int(tokens[i+1]); i += 2
                if step <= 0 or end < start:
                    raise ValueError('Invalid TO/BY range.')
                result.extend(str(x) for x in range(start, end+1, step))
            else:
                result.append(str(int(t))); i += 1
        return result
    nodes = {}
    for entry in section('JOINT COORDINATES', True):
        t = entry.split()
        if len(t) % 4:
            raise ValueError(f"Invalid coordinates: {entry}")
        for j in range(0, len(t), 4):
            ident = str(int(t[j])); x, y, z = map(float, t[j+1:j+4])
            if abs(z) > 1e-8:
                raise ValueError('Only planar XY trusses (Z=0) are supported.')
            if ident in nodes:
                raise ValueError(f'Duplicate joint {ident}.')
            nodes[ident] = (x*lf, y*lf)
    members = []
    for entry in section('MEMBER INCIDENCES', True):
        t = entry.split()
        if len(t) % 3:
            raise ValueError(f'Invalid member incidence: {entry}')
        for j in range(0, len(t), 3):
            mid, start, end = t[j:j+3]
            if start not in nodes or end not in nodes:
                raise ValueError(f'Member {mid} references an unknown joint.')
            members.append((mid, start, end))
    if not nodes or not members:
        raise ValueError('No joints or members found.')
    member_ids = {m[0] for m in members}
    if len(member_ids) != len(members):
        raise ValueError('Duplicate member numbers.')
    supports = []
    for entry in section('SUPPORTS'):
        t = entry.split()
        pos = next((i for i, word in enumerate(t) if word in ('PINNED', 'FIXED', 'ENFORCED')), -1)
        if pos < 1:
            raise ValueError(f'Unsupported support: {entry}')
        ids = expand(t[:pos]); kind = t[pos]; rest = t[pos+1:]
        if kind == 'PINNED' and not rest:
            support_type = 'Pin'
        elif kind in ('FIXED', 'ENFORCED') and rest[:1] == ['BUT']:
            released = set(rest[1:])
            if not released <= {'FX','FY','FZ','MX','MY','MZ'}:
                raise ValueError(f'Unsupported support release: {entry}')
            if 'FX' in released and 'FY' in released:
                raise ValueError(f'Both in-plane directions released: {entry}')
            support_type = ('Horizontal Roller' if 'FX' in released else
                            'Vertical Roller' if 'FY' in released else 'Pin')
        elif kind in ('FIXED', 'ENFORCED') and not rest:
            support_type = 'Pin'  # rotational restraint irrelevant to truss DOFs
        else:
            raise ValueError(f'Unsupported support: {entry}')
        for ident in ids:
            if ident not in nodes:
                raise ValueError(f'Support references unknown joint {ident}.')
            supports.append({'Joint': ident, 'Support Type': support_type})
    if len({r['Joint'] for r in supports}) != len(supports):
        raise ValueError('Multiple support definitions for the same joint.')
    # Material E is converted from force/length² to ksi.
    elastic = {}
    current_material = None
    for line in lines:
        if line.startswith('ISOTROPIC '):
            current_material = line.split(None, 1)[1]
        elif line.startswith('END DEFINE'):
            current_material = None
        elif current_material and line.startswith('E '):
            elastic[current_material] = float(line.split()[1]) * ff / lf**2
    member_material = {}
    for entry in section('CONSTANTS'):
        t = entry.split()
        if len(t) >= 3 and t[0] == 'MATERIAL':
            mat = t[1]
            ids = list(member_ids) if t[2:] == ['ALL'] else expand(t[2:])
            for ident in ids:
                if ident not in member_ids:
                    raise ValueError(f'Material assignment references unknown member {ident}.')
                member_material[ident] = mat
    areas = {}
    for entry in section('MEMBER PROPERTY'):
        t = entry.split()
        if 'PRIS' not in t or 'AX' not in t:
            raise ValueError(f'Unsupported member property (need PRIS AX): {entry}')
        pos = t.index('PRIS')
        ids = expand(t[:pos]); prop = t[pos+1:]
        if 'AX' not in prop or prop.index('AX')+1 >= len(prop):
            raise ValueError(f'Missing AX area: {entry}')
        area = float(prop[prop.index('AX')+1]) * lf**2
        if area <= 0:
            raise ValueError(f'Nonpositive area: {entry}')
        for ident in ids:
            if ident not in member_ids:
                raise ValueError(f'Property references unknown member {ident}.')
            areas[ident] = area
    # Check for non-truss structural actions that the solver cannot reproduce.
    for line in lines:
        if line.startswith(('MEMBER RELEASE', 'MEMBER OFFSET', 'ELEMENT INCIDENCES',
                            'MEMBER LOAD', 'ELEMENT LOAD', 'SELFWEIGHT', 'FLOOR LOAD',
                            'REPEAT LOAD', 'SUPPORT DISPLACEMENT', 'LOAD COMB')):
            raise ValueError(f'Unsupported 2D truss feature: {line}')
    loads = []
    load_headers = [i for i, line in enumerate(lines) if re.match(r'^LOAD\s+\d+\b', line)]
    if len(load_headers) > 1:
        raise ValueError('Multiple load cases detected; select/export one primary load case.')
    if load_headers:
        totals = {}
        in_joint = False
        for line in lines[load_headers[0]+1:]:
            if line.startswith(('PERFORM ANALYSIS', 'FINISH', 'PRINT')):
                break
            if line == 'JOINT LOAD':
                in_joint = True; continue
            if line.startswith(heads):
                raise ValueError(f'Unsupported load-case command: {line}')
            if not in_joint:
                raise ValueError(f'Unsupported load-case command: {line}')
            for entry in line.split(';'):
                t = entry.split()
                if not t: continue
                pos = next((i for i, word in enumerate(t) if word in ('FX','FY','FZ','MX','MY','MZ')), -1)
                if pos < 1 or (len(t)-pos) % 2:
                    raise ValueError(f'Unsupported joint load: {entry}')
                ids = expand(t[:pos])
                for ident in ids:
                    if ident not in nodes:
                        raise ValueError(f'Load references unknown joint {ident}.')
                    values = totals.setdefault(ident, [0., 0.])
                    for j in range(pos, len(t), 2):
                        if t[j] not in ('FX','FY'):
                            raise ValueError(f'Unsupported load component: {t[j]}')
                        values[0 if t[j]=='FX' else 1] += float(t[j+1])*ff
        loads = [{'Joint': ident, 'Fx (kip)': f[0], 'Fy (kip)': f[1]}
                 for ident, f in totals.items()]
    properties = {}
    for mid, _, _ in members:
        mat = member_material.get(mid)
        properties[mid] = {'area': areas.get(mid), 'E': elastic.get(mat), 'material': mat}
    return nodes, members, supports, loads, properties


if uploaded_std is not None:
    st.success(f"File selected: {uploaded_std.name}")
    try:
        std_text = uploaded_std.getvalue().decode("utf-8", errors="replace")
        imported_nodes, imported_members, imported_supports, imported_loads, imported_properties = ( 
            parse_staad_truss(std_text)
        )
    except (ValueError, TypeError, IndexError) as exc:
        st.error(f"Could not import STAAD.Pro model: {exc}")
        st.stop()

    if not imported_loads:
        st.warning("No supported joint loads were defined. Add loads in the Joint Load Editor before analysis.")
    st.subheader("Imported STAAD.Pro Truss")
    fig_import, ax_import = plt.subplots(figsize=(9, 5))
    for member_id, start, end in imported_members:
        x1, y1 = imported_nodes[start]
        x2, y2 = imported_nodes[end]
        ax_import.plot([x1, x2], [y1, y2], color="steelblue", linewidth=3)
        ax_import.text((x1 + x2) / 2, (y1 + y2) / 2,
                       member_id, color="darkred", fontsize=10)
    for joint_id, (x, y) in imported_nodes.items():
        ax_import.plot(x, y, "ko", markersize=7)
        ax_import.annotate(joint_id, (x, y), xytext=(7, 7),
                           textcoords="offset points")
    ax_import.set_xlabel("X (in)")
    ax_import.set_ylabel("Y (in)")
    ax_import.set_title("STAAD.Pro Imported Geometry")
    ax_import.set_aspect("equal", adjustable="datalim")
    ax_import.grid(True, alpha=0.3)
    st.pyplot(fig_import)
    plt.close(fig_import)
    st.success(f"Imported {len(imported_nodes)} joints, "
               f"{len(imported_members)} members, "
               f"{len(imported_supports)} supports, "
               f"and {len(imported_loads)} loaded joints.")


# CUSTOM TRUSS GEOMETRY EDITOR
st.subheader("Custom Truss Geometry")
st.caption("Enter joint coordinates in inches.")

default_joints = pd.DataFrame([
    {"Joint": "A", "X (in)": 0.0, "Y (in)": 0.0},
    {"Joint": "B", "X (in)": 120.0, "Y (in)": 96.0},
    {"Joint": "C", "X (in)": 240.0, "Y (in)": 0.0},
])


default_members = pd.DataFrame([
    {
        "Start": "A",
        "End": "B",
        "Material": "Steel",
        "Area (in²)": 2.0,
        "E (ksi)": 29000.0
    },
    {
        "Start": "B",
        "End": "C",
        "Material": "Steel",
        "Area (in²)": 2.0,
        "E (ksi)": 29000.0
    },
    {
        "Start": "A",
        "End": "C",
        "Material": "Steel",
        "Area (in²)": 2.0,
        "E (ksi)": 29000.0
    }
])

# USE IMPORTED STAAD.PRO GEOMETRY WHEN AVAILABLE
if uploaded_std is not None and imported_nodes and imported_members:
    default_joints = pd.DataFrame([
        {
            "Joint": joint_id,
            "X (in)": x,
            "Y (in)": y
        }
        for joint_id, (x, y) in imported_nodes.items()
    ])

    default_members = pd.DataFrame([
        {
            "Start": start,
            "End": end,
            "Material": ("Aluminum" if "ALUMIN" in (imported_properties[member_id]['material'] or "") else "Timber" if "TIMBER" in (imported_properties[member_id]['material'] or "") or "WOOD" in (imported_properties[member_id]['material'] or "") else "Steel"),
            "Area (in²)": imported_properties[member_id]['area'],
            "E (ksi)": imported_properties[member_id]['E']
        }
        for member_id, start, end in imported_members
    ])

    st.success(
        "STAAD.Pro geometry loaded into the Joints and Members editors."
    )


st.markdown("#### Joints")
joints_df = st.data_editor(
    default_joints,
    num_rows="dynamic",
    hide_index=True,
    use_container_width=True,
    key=f"joint_editor_{hashlib.sha256(uploaded_std.getvalue()).hexdigest()[:12] if uploaded_std is not None else 'manual'}"
)

st.markdown("#### Members")

members_df = st.data_editor(
    default_members,
    num_rows="dynamic",
    hide_index=True,
    width="stretch",
    # Versioned widget key resets stale Streamlit editor edits from earlier deployments.
    # Otherwise old None area values can override newly imported AX values.
    key=f"member_editor_v3_{hashlib.sha256(uploaded_std.getvalue()).hexdigest()[:12] if uploaded_std is not None else 'manual'}",
    column_config={
        "Material": st.column_config.SelectboxColumn(
            "Material",
            options=["Steel", "Aluminum", "Timber"],
            required=True
        ),
        "E (ksi)": st.column_config.NumberColumn("E (ksi)", min_value=0.001, format="%.3f"),
        "Area (in²)": st.column_config.NumberColumn(
            "Area (in²)",
            min_value=0.000001,
            required=True,
            format="%.3f"
        )
    }
)


# Check the *effective editor values*, not just the raw STAAD import.
# The editor can contain user changes or retained session-state edits.
missing_area_count = int(members_df["Area (in²)"].isna().sum())
missing_E_count = int(members_df["E (ksi)"].isna().sum())
if missing_area_count:
    st.warning(
        f"Missing member areas for {missing_area_count} members. "
        "Enter actual areas in the Members Editor before analysis."
    )
if missing_E_count:
    st.warning(
        f"Missing elastic modulus for {missing_E_count} members. "
        "Enter actual E values in the Members Editor before analysis."
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

# Use imported STAAD.Pro supports when available.
# Otherwise, use the example supports for the default truss.

if uploaded_std is not None and imported_nodes:
    default_supports = pd.DataFrame(
        imported_supports, columns=["Joint", "Support Type"]
    )
else:
    default_supports = pd.DataFrame([
        {"Joint": "A", "Support Type": "Pin"},
        {"Joint": "C", "Support Type": "Horizontal Roller"},
    ])

supports_df = st.data_editor(
    default_supports,
    num_rows="dynamic",
    hide_index=True,
    use_container_width=True,
    key=f"support_editor_{uploaded_std.name if uploaded_std is not None else 'manual'}",
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


# MATERIAL AND SECTION INFORMATION
# Material and cross-sectional area are assigned
# individually to each member in the Members Editor.

st.sidebar.header("Material and Section")

st.sidebar.info(
    "Material and cross-sectional area are assigned "
    "individually to each truss member using the "
    "Members Editor."
)

st.sidebar.caption(
    "Calculations use kips and inches. "
    "Each member uses its own material properties "
    "and cross-sectional area."
)

# MULTIPLE JOINT LOAD EDITOR
st.subheader("Joint Load Editor")
st.caption(
    "Apply horizontal and vertical loads at any joint. "
    "Positive Fx = right, positive Fy = up. "
    "Negative Fx = left, negative Fy = down. "
    "All forces are in kips."
)

# Load defaults for STAAD.Pro imports or manual models
if uploaded_std is not None and imported_nodes:
    default_loads = pd.DataFrame(
        imported_loads,
        columns=["Joint", "Fx (kip)", "Fy (kip)"]
    )
else:
    default_loads = pd.DataFrame([
        {"Joint": "B", "Fx (kip)": 0.0, "Fy (kip)": -10.0}
    ])

loads_df = st.data_editor(
    default_loads,
    num_rows="dynamic",
    hide_index=True,
    width="stretch",
    key=f"load_editor_{uploaded_std.name if uploaded_std is not None else 'manual'}",
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
    if uploaded_std is not None and loads_df.empty:
        st.info("Enter at least one joint load to analyze this imported model.")
        st.stop()
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

# Coordinates are already in inches; E is in ksi and forces in kips.
coords = np.array([nodes[n] for n in node_names])
ndof = 2 * len(nodes)

# MEMBER MATERIAL PROPERTIES
# E values are stored internally in ksi.
material_E = {
    "Steel": 29000.0,
    "Aluminum": 10000.0,
    "Timber": 1200.0
}

K = np.zeros((ndof, ndof))
member_data = []

# Validate member properties before assembling stiffness
if members_df[["Start", "End", "Material", "Area (in²)", "E (ksi)"]].isnull().any().any():
    st.error("Every member must have joints, material, and area.")
    st.stop()

for _, row in members_df.iterrows():
    start = str(row["Start"]).strip()
    end = str(row["End"]).strip()
    material = str(row["Material"])
    member_A = float(row["Area (in²)"])

    if material not in material_E:
        st.error(f"Unsupported material: {material}")
        st.stop()

    if not np.isfinite(member_A) or member_A <= 0:
        st.error(f"Invalid area for member {start}-{end}.")
        st.stop()

    member_E = float(row["E (ksi)"])
    if not np.isfinite(member_E) or member_E <= 0:
        st.error(f"Invalid E for member {start}-{end}.")
        st.stop()

    i = node_index[start]
    j = node_index[end]

    dx, dy = coords[j] - coords[i]
    L = np.hypot(dx, dy)

    if L <= 0:
        st.error(f"Member {start}-{end} has zero length.")
        st.stop()

    c, s = dx / L, dy / L

    k = (member_E * member_A / L) * np.array([
        [c*c, c*s, -c*c, -c*s],
        [c*s, s*s, -c*s, -s*s],
        [-c*c, -c*s, c*c, c*s],
        [-c*s, -s*s, c*s, s*s],
    ])

    dofs = [2*i, 2*i+1, 2*j, 2*j+1]
    K[np.ix_(dofs, dofs)] += k

    member_data.append((
        start, end, L, c, s, dofs,
        member_E, member_A
    ))


# Use the restraints selected in the Support Editor
free = [i for i in range(ndof) if i not in restrained]

def solve(force_vector):
    displacement = np.zeros(ndof)
    displacement[free] = np.linalg.solve(
        K[np.ix_(free, free)], force_vector[free]
    )
    reactions = K @ displacement - force_vector

    axial_forces = []
    for start, end, L, c, s, dofs, member_E, member_A in member_data:
        u = displacement[dofs]
        N = (member_E * member_A / L) * np.dot(
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
st.caption(
    "Member axial forces are in kips. "
    "Positive (+) = tension; negative (-) = compression. "
    "Green arrows represent applied joint loads."
)
fig, ax = plt.subplots(figsize=(10, 4))
for index, (start, end) in enumerate(members):
    x1, y1 = nodes[start]
    x2, y2 = nodes[end]
    color = "red" if N[index] < -1e-8 else (
        "blue" if N[index] > 1e-8 else "gray"
    )
    ax.plot([x1, x2], [y1, y2], color=color, linewidth=3)
    # Position member force labels away from the member lines
    dx = x2 - x1
    dy = y2 - y1
    length = np.hypot(dx, dy)

    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2

    # Perpendicular offset from the member
    offset = 12.0

    label_x = mid_x - offset * dy / length
    label_y = mid_y + offset * dx / length

    ax.text(
        label_x,
        label_y,
        f"{N[index]:+.2f}",
        fontsize=8,
        ha="center",
        va="center",
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.85,
            pad=1.5
        ),
        zorder=10
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
    # STEP-BY-STEP GLOBAL EQUILIBRIUM
    # INDEPENDENT SUPPORT REACTIONS USING STATICS
    st.subheader("Independent Support Reactions — Statics")
    st.caption(
        "Calculates reactions using only applied loads, support locations, "
        "and the three equations of global equilibrium. "
        "Stiffness-method reactions are used only for comparison."
    )

    if len(restrained) == 3:
        equilibrium_matrix = np.zeros((3, 3))

        for col, dof in enumerate(restrained):
            joint = node_names[dof // 2]
            x, y = nodes[joint]

            if dof % 2 == 0:
                # Horizontal reaction contributes to Fx and moment
                equilibrium_matrix[:, col] = [1.0, 0.0, -y]
            else:
                # Vertical reaction contributes to Fy and moment
                equilibrium_matrix[:, col] = [0.0, 1.0, x]

        applied_fx = float(F[0::2].sum())
        applied_fy = float(F[1::2].sum())

        applied_moment = 0.0
        for joint, idx in node_index.items():
            x, y = nodes[joint]
            applied_moment += (
                x * F[2 * idx + 1]
                - y * F[2 * idx]
            )

        equilibrium_rhs = -np.array([
            applied_fx,
            applied_fy,
            applied_moment
        ])

        if np.linalg.matrix_rank(equilibrium_matrix) == 3:
            statics_reactions = np.linalg.solve(
                equilibrium_matrix,
                equilibrium_rhs
            )

            comparison_rows = []

            for col, dof in enumerate(restrained):
                joint = node_names[dof // 2]
                direction = "X" if dof % 2 == 0 else "Y"

                statics_value = float(statics_reactions[col])
                stiffness_value = float(reactions[dof])
                difference = abs(statics_value - stiffness_value)

                comparison_rows.append({
                    "Support": joint,
                    "Direction": direction,
                    "Statics (kip)": statics_value,
                    "Stiffness (kip)": stiffness_value,
                    "Difference (kip)": difference
                })

            st.dataframe(
                pd.DataFrame(comparison_rows).style.format({
                    "Statics (kip)": "{:.4f}",
                    "Stiffness (kip)": "{:.4f}",
                    "Difference (kip)": "{:.6f}"
                }),
                hide_index=True,
                use_container_width=True
            )

            max_difference = max(
                row["Difference (kip)"]
                for row in comparison_rows
            )

            if max_difference < 1e-5:
                st.success(
                    "PASS: Independent statics reactions agree "
                    "with the stiffness-method reactions."
                )
            else:
                st.warning(
                    "CHECK: Independent statics reactions differ "
                    "from the stiffness-method reactions."
                )

        else:
            st.warning(
                "The support reactions cannot be uniquely determined "
                "from global equilibrium for this support arrangement."
            )

    else:
        st.info(
            "Independent reaction calculation requires exactly three "
            "restrained translational degrees of freedom. "
            "Other support arrangements may require additional equations."
        )
    st.subheader("Step-by-Step Equilibrium Verification")
    st.caption(
        "Positive X is right, positive Y is up, "
        "and positive moment is counterclockwise."
    )

    force_x_terms = []
    force_y_terms = []
    moment_terms = []

    for joint, idx in node_index.items():
        x, y = nodes[joint]

        fx = float(F[2 * idx])
        fy = float(F[2 * idx + 1])

        rx = float(reactions[2 * idx])
        ry = float(reactions[2 * idx + 1])

        if abs(fx) > 1e-10:
            force_x_terms.append(f"({fx:+.3f})")

        if abs(fy) > 1e-10:
            force_y_terms.append(f"({fy:+.3f})")

        if abs(rx) > 1e-10:
            force_x_terms.append(f"({rx:+.3f})")

        if abs(ry) > 1e-10:
            force_y_terms.append(f"({ry:+.3f})")

        # Moments about the global origin (0, 0)
        if abs(fx) > 1e-10 or abs(fy) > 1e-10:
            moment_terms.append(
                f"[({x:.2f})({fy:+.3f}) "
                f"- ({y:.2f})({fx:+.3f})]"
            )

        if abs(rx) > 1e-10 or abs(ry) > 1e-10:
            moment_terms.append(
                f"[({x:.2f})({ry:+.3f}) "
                f"- ({y:.2f})({rx:+.3f})]"
            )

    st.markdown("**1. Horizontal Force Equilibrium**")
    st.latex(r"\sum F_x = 0")
    st.code(
        " + ".join(force_x_terms or ["0"])
        + f" = {total_fx:.6f} kip"
    )

    st.markdown("**2. Vertical Force Equilibrium**")
    st.latex(r"\sum F_y = 0")
    st.code(
        " + ".join(force_y_terms or ["0"])
        + f" = {total_fy:.6f} kip"
    )

    st.markdown("**3. Moment Equilibrium About the Origin**")
    st.latex(r"\sum M_O = \sum(xF_y-yF_x)=0")
    st.code(
        " + ".join(moment_terms or ["0"])
        + f" = {total_moment:.6f} kip-in"
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

    
    # STEP-BY-STEP METHOD OF JOINTS
    st.divider()
    st.subheader("Step-by-Step Method of Joints")
    st.caption(
        "Joint equilibrium using member lengths. "
        "Positive axial force means tension; negative means compression."
    )

    # Build geometry and joint connections independently of stiffness.
    member_info = []
    connected = {joint: [] for joint in nodes}

    for j, (start, end) in enumerate(members):
        x1, y1 = nodes[start]
        x2, y2 = nodes[end]
        dx = x2 - x1
        dy = y2 - y1
        L = float(np.hypot(dx, dy))

        if L <= 1e-12:
            st.error(f"Member {start}-{end} has zero length.")
            st.stop()

        member_info.append((start, end, dx, dy, L))
        connected[start].append((j, dx / L, dy / L))
        connected[end].append((j, -dx / L, -dy / L))

    # METHOD OF JOINTS — select valid reaction source without stopping other tabs.
    # Three independent reactions can be determined from global statics.
    # Redundant support arrangements need compatibility/stiffness to find reactions.
    joint_reactions = np.zeros(ndof)
    independent_joint_reactions = False

    if len(restrained) == 3:
        equilibrium_matrix_moj = np.zeros((3, 3))
        for col, dof in enumerate(restrained):
            joint = node_names[dof // 2]
            x, y = nodes[joint]
            equilibrium_matrix_moj[:, col] = (
                [1.0, 0.0, -y] if dof % 2 == 0 else [0.0, 1.0, x]
            )
        if np.linalg.matrix_rank(equilibrium_matrix_moj) == 3:
            applied_moment_moj = sum(
                nodes[joint][0] * F[2*idx+1] - nodes[joint][1] * F[2*idx]
                for joint, idx in node_index.items()
            )
            statics_reactions_moj = np.linalg.solve(
                equilibrium_matrix_moj,
                -np.array([F[0::2].sum(), F[1::2].sum(), applied_moment_moj])
            )
            for col, dof in enumerate(restrained):
                joint_reactions[dof] = statics_reactions_moj[col]
            independent_joint_reactions = True

    if not independent_joint_reactions:
        # For statically indeterminate supports, the stiffness solution gives
        # the compatible reactions. This is NOT an independent verification.
        joint_reactions[restrained] = reactions[restrained]
        st.info(
            "Method of Joints uses stiffness-method support reactions for "
            "this support arrangement. Member equilibrium can be checked, "
            "but this is not an independent statics verification."
        )

    # Combine applied loads with the selected support reactions.
    joint_external = {}

    for joint, idx in node_index.items():
        joint_external[joint] = (
            float(F[2 * idx] + joint_reactions[2 * idx]),
            float(F[2 * idx + 1] + joint_reactions[2 * idx + 1])
        )

    solved_forces = {}
    steps = []
    tolerance = 1e-8

    # Repeatedly find joints with one or two unknown members.
    while len(solved_forces) < len(members):
        progress = False

        for joint in nodes:
            unknown = [
                (j, cx, cy)
                for j, cx, cy in connected[joint]
                if j not in solved_forces
            ]

            if not unknown or len(unknown) > 2:
                continue

            known_x = joint_external[joint][0]
            known_y = joint_external[joint][1]

            for j, cx, cy in connected[joint]:
                if j in solved_forces:
                    known_x += solved_forces[j] * cx
                    known_y += solved_forces[j] * cy

            A_joint = np.array(
                [[cx for _, cx, cy in unknown],
                 [cy for _, cx, cy in unknown]],
                dtype=float
            )
            b_joint = -np.array([known_x, known_y])

            if np.linalg.matrix_rank(A_joint) < len(unknown):
                continue

            values = np.linalg.lstsq(
                A_joint, b_joint, rcond=None
            )[0]

            if np.linalg.norm(
                A_joint @ values - b_joint
            ) > 1e-6:
                continue

            step = {
                "joint": joint,
                "unknown": unknown,
                "known_x": known_x,
                "known_y": known_y,
                "values": values
            }
            steps.append(step)

            for (j, cx, cy), value in zip(unknown, values):
                solved_forces[j] = float(value)

            progress = True
            break

        if not progress:
            break

    if not steps:
        st.warning(
            "No joint with a uniquely solvable set of one or two "
            "unknown member forces was found."
        )

    for number, step in enumerate(steps, start=1):
        joint = step["joint"]
        unknown = step["unknown"]
        known_x = step["known_x"]
        known_y = step["known_y"]
        values = step["values"]

        st.markdown(f"### Step {number}: Joint {joint}")

        # Show geometry only when a diagonal member is involved.
        for j, cx, cy in unknown:
            start, end, dx, dy, L = member_info[j]

            if abs(dx) > tolerance and abs(dy) > tolerance:
                st.write(
                    f"Member {start}-{end}: "
                    f"L = √({abs(dx):.3f}² + {abs(dy):.3f}²) "
                    f"= {L:.3f} in"
                )

        # Display the joint equilibrium equations.
        x_terms = [f"({known_x:+.4f})"]
        y_terms = [f"({known_y:+.4f})"]

        for j, cx, cy in unknown:
            start, end, dx, dy, L = member_info[j]
            name = f"F({start}-{end})"

            if abs(cx) > tolerance:
                x_terms.append(
                    f"({cx:+.4f}){name}"
                )
            if abs(cy) > tolerance:
                y_terms.append(
                    f"({cy:+.4f}){name}"
                )

        st.markdown("**Horizontal equilibrium**")
        st.latex(r"\sum F_x=0")
        st.code(" + ".join(x_terms) + " = 0")

        st.markdown("**Vertical equilibrium**")
        st.latex(r"\sum F_y=0")
        st.code(" + ".join(y_terms) + " = 0")

        # Show substituted numerical equations and solutions.
        st.markdown("**Solve the unknown member forces**")
        
        # SHOW METHOD OF JOINTS ALGEBRA
        # Use the equilibrium equation that isolates each force.
        remaining = list(range(len(unknown)))
        known_contributions = np.array([known_x, known_y], dtype=float)

        while remaining:
            found = False

            for pos in remaining:
                other_positions = [k for k in remaining if k != pos]

                for axis, axis_name in [(0, "x"), (1, "y")]:
                    coefficient = unknown[pos][1 + axis]

                    # The chosen equation must contain the target force,
                    # but no other force that is still unknown.
                    if abs(coefficient) <= tolerance:
                        continue

                    if any(
                        abs(unknown[k][1 + axis]) > tolerance
                        for k in other_positions
                    ):
                        continue

                    j, cx, cy = unknown[pos]
                    start, end, dx, dy, L = member_info[j]

                    numerator = -known_contributions[axis]
                    calculated = numerator / coefficient

                    st.markdown(
                        f"**Solve member {start}-{end} using ΣF{axis_name} = 0:**"
                    )

                    st.latex(
                        rf"F_{{{start}-{end}}}"
                        rf"=\frac{{{numerator:+.4f}}}"
                        rf"{{{coefficient:+.4f}}}"
                        rf"={calculated:+.4f}\ \mathrm{{kip}}"
                    )

                    # Substitute this solved force into both equations.
                    known_contributions[0] += values[pos] * cx
                    known_contributions[1] += values[pos] * cy

                    remaining.remove(pos)
                    found = True
                    break

                if found:
                    break

            if not found:
                st.caption(
                    "These member forces are solved simultaneously "
                    "using the two joint equilibrium equations."
                )
                break

        for (j, cx, cy), value in zip(unknown, values):
            start, end, dx, dy, L = member_info[j]
            classification = (
                "Tension" if value > tolerance
                else "Compression" if value < -tolerance
                else "Zero-force"
            )

            st.write(
                f"F({start}-{end}) = {value:+.6f} kip "
                f"({classification})"
            )

    if len(solved_forces) != len(members):
        st.warning(
            f"Method of Joints solved {len(solved_forces)} "
            f"of {len(members)} member forces. "
            "The remaining forces require a different equilibrium "
            "strategy or cannot be uniquely determined by this method."
        )
    else:
        if independent_joint_reactions:
            st.success("All member forces solved by independent Method of Joints.")
        else:
            st.success("All member forces satisfy joint equilibrium using stiffness-derived reactions.")

        comparison_rows = []
        for j, (start, end) in enumerate(members):
            joint_value = solved_forces[j]
            stiffness_value = float(N[j])

            comparison_rows.append({
                "Member": f"{start}-{end}",
                "Stiffness (kip)": stiffness_value,
                "Method of Joints (kip)": joint_value,
                "Difference (kip)": abs(
                    stiffness_value - joint_value
                )
            })

        st.markdown("**Member Force Comparison**")
        st.dataframe(
            pd.DataFrame(comparison_rows),
            hide_index=True,
            use_container_width=True
        )


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

    # STEP-BY-STEP DISPLACEMENT CALCULATIONS
    st.divider()
    st.subheader("Step-by-Step Joint Displacement Calculations")

    st.markdown("### Step 1: Member Stiffness Calculations")

    for j, (start, end, L, c, s, dofs, member_E, member_A) in enumerate(member_data):
        x1, y1 = nodes[start]
        x2, y2 = nodes[end]

        dx = x2 - x1
        dy = y2 - y1

        st.markdown(f"**Member {start}–{end}**")

        if abs(dx) > 1e-10 and abs(dy) > 1e-10:
            st.write(
                f"L = √({abs(dx):.3f}² + {abs(dy):.3f}²) "
                f"= {L:.3f} in"
            )
        else:
            st.write(f"L = {L:.3f} in")

        AE = member_A * member_E
        k = AE / L

        st.write(
            f"AE = ({member_A:.3f} in²)"
            f"({member_E:.3f} kip/in²)"
            f" = {AE:.3f} kip"
        )

        st.write(
            f"k = AE/L = {AE:.3f}/{L:.3f} "
            f"= {k:.6f} kip/in"
        )

        st.write(
            f"c = Δx/L = {dx:.3f}/{L:.3f} = {c:.6f}"
        )
        st.write(
            f"s = Δy/L = {dy:.3f}/{L:.3f} = {s:.6f}"
        )

    st.markdown("### Step 2: Global Stiffness Matrix")

    st.latex(r"K=\sum k_e")

    st.caption(
        "The global stiffness matrix is assembled by adding "
        "each member's stiffness contribution at its joint degrees "
        "of freedom. Matrix units: kip/in."
    )

    st.dataframe(
        pd.DataFrame(
            K,
            index=[
                f"{joint}-{direction}"
                for joint in node_index
                for direction in ("X", "Y")
            ],
            columns=[
                f"{joint}-{direction}"
                for joint in node_index
                for direction in ("X", "Y")
            ]
        ).round(4),
        use_container_width=True
    )

    st.markdown("### Step 3: Apply Supports and Loads")

    all_dofs = np.arange(len(displacement))
    free_dofs = np.array([
        dof for dof in all_dofs
        if dof not in restrained
    ], dtype=int)

    K_ff_steps = K[np.ix_(free_dofs, free_dofs)]
    F_f_steps = F[free_dofs]

    dof_labels = [
        f"{list(node_index.keys())[dof // 2]}-"
        f"{'X' if dof % 2 == 0 else 'Y'}"
        for dof in free_dofs
    ]

    st.write("Unknown displacement degrees of freedom:")
    st.write(", ".join(dof_labels))

    st.latex(r"K_{ff}u_f=F_f")

    st.markdown("**Reduced stiffness matrix (kip/in)**")
    st.dataframe(
        pd.DataFrame(
            K_ff_steps,
            index=dof_labels,
            columns=dof_labels
        ).round(5),
        use_container_width=True
    )

    st.markdown("**Applied load vector (kip)**")
    st.dataframe(
        pd.DataFrame({
            "Degree of Freedom": dof_labels,
            "Applied Load (kip)": F_f_steps
        }),
        hide_index=True,
        use_container_width=True
    )

    st.markdown("### Step 4: Solve the Displacements")

    st.latex(r"u_f=K_{ff}^{-1}F_f")

    st.caption(
        "The numerical solver computes the displacement vector "
        "by solving the reduced linear system."
    )

    displacement_steps = np.linalg.solve(
        K_ff_steps, F_f_steps
    )

    for label, value in zip(dof_labels, displacement_steps):
        st.write(
            f"{label}: u = {value:+.8f} in"
        )

    difference = np.max(
        np.abs(displacement[free_dofs] - displacement_steps)
    )

    st.write(
        f"Maximum difference from original solver: "
        f"{difference:.10f} in"
    )

with tab4:
    st.subheader("Method of Virtual Work")
    st.latex(r"\delta = \sum_i \frac{N_i n_i L_i}{A_i E_i}")

    rows = []
    
    for index, (start, end, L, c, s, dofs, member_E, member_A) in enumerate(member_data):
        contribution = (
            N[index] * n[index] * L
            / (member_A * member_E)
        )

        rows.append({
            "Member": f"{start}-{end}",
            "N (kip)": N[index],
            "n (kip/unit load)": n[index],
            "L (in)": L,
            "A (in²)": member_A,
            "E (ksi)": member_E,
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
     
    # STEP-BY-STEP VIRTUAL WORK CALCULATIONS
    st.divider()
    st.subheader("Step-by-Step Virtual Work Calculations")

    st.caption(
        "N = member force from actual loading (kip). "
        "n = member force from a unit load at the selected joint "
        "and direction (kip per unit load). "
        "L = member length (in), A = area (in²), E = modulus (ksi)."
    )

    st.latex(r"\delta=\sum_i\frac{N_i n_i L_i}{A_i E_i}")

    vw_contributions = []

    for i, (start, end, L, c, s, dofs, member_E, member_A) in enumerate(member_data):
        actual_force = float(N[i])
        unit_force = float(n[i])

        AE = member_A * member_E
        numerator = actual_force * unit_force * L
        contribution = numerator / AE

        vw_contributions.append(contribution)

        with st.expander(
            f"Member {start}–{end}: "
            f"{contribution:+.8f} in",
            expanded=True
        ):
            st.markdown("**Step 1 — Calculate AE**")

            st.latex(r"AE=A\times E")

            st.write(
                f"AE = ({member_A:.4f}) × ({member_E:.4f}) "
                f"= {AE:.4f} kip"
            )

            st.markdown("**Step 2 — Multiply N × n × L**")

            st.latex(r"NnL=N\times n\times L")

            st.write(
                f"NnL = ({actual_force:+.6f}) × "
                f"({unit_force:+.6f}) × ({L:.4f}) "
                f"= {numerator:+.8f}"
            )

            st.markdown("**Step 3 — Divide by AE**")

            st.latex(r"\delta_i=\frac{N_i n_i L_i}{A_i E_i}")

            st.write(
                f"δ = ({numerator:+.8f}) / ({AE:.4f}) "
                f"= {contribution:+.8f} in"
            )

    st.markdown("### Step 4 — Add All Member Contributions")

    total_steps = float(sum(vw_contributions))

    contribution_terms = [
        f"({value:+.8f})"
        for value in vw_contributions
    ]

    st.write("δ = " + " + ".join(contribution_terms))

    st.latex(r"\delta=\sum_i\delta_i")

    st.write(
        f"**Total Virtual Work Displacement = "
        f"{total_steps:+.8f} in**"
    )

    st.markdown("### Step 5 — Compare With Stiffness Method")

    st.write(
        f"Virtual Work: {total_steps:+.8f} in"
    )

    st.write(
        f"Direct Stiffness Method: {direct:+.8f} in"
    )

    st.write(
        f"Absolute Difference: "
        f"{abs(total_steps - direct):.10f} in"
    )

    if np.isclose(total_steps, direct, atol=1e-8, rtol=1e-6):
        st.success(
            "PASS: Virtual Work agrees with the stiffness method."
        )
    else:
        st.error(
            "CHECK: Virtual Work and stiffness results do not agree."
        )
   
st.warning(
    "This is a sample 2D truss solver. STAAD.Pro import, "
    "editable geometry, detailed hand-calculation steps, "
    "and independent validation are still under development."
)
