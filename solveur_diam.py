# +
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


model = gp.Model("timetable_discrete")

machines = 3
timeline = 48
ordres = [
    {'numéro': 1, 'duration': 2, 'due date': 10}, {'numéro': 2, 'duration': 2, 'due date': 10},
    {'numéro': 3, 'duration': 1, 'due date': 10}, {'numéro': 4, 'duration': 2, 'due date': 10},
    {'numéro': 5, 'duration': 3, 'due date': 15}, {'numéro': 6, 'duration': 4, 'due date': 15},
    {'numéro': 7, 'duration': 2, 'due date': 20}, {'numéro': 8, 'duration': 4, 'due date': 20},
    {'numéro': 9, 'duration': 2, 'due date': 30}, {'numéro': 10, 'duration': 2, 'due date': 30},
    {'numéro': 11, 'duration': 2, 'due date': 30}, {'numéro': 12, 'duration': 2, 'due date': 30},
    {'numéro': 13, 'duration': 10, 'due date': 40}, {'numéro': 14, 'duration': 2, 'due date': 45},
    {'numéro': 15, 'duration': 8, 'due date': 40}, {'numéro': 16, 'duration': 5, 'due date': 40},
    {'numéro': 17, 'duration': 2, 'due date': 35}, {'numéro': 18, 'duration': 12, 'due date': 30},
    {'numéro': 19, 'duration': 2, 'due date': 30}, {'numéro': 20, 'duration': 3, 'due date': 30}
]

# ---- PARAMETERS -----------------------------------------------------------
I        = range(len(ordres))          # order indices
M        = range(machines)             # machine indices
T        = range(timeline)             # time indices
K        = 10                          # tardiness weight
due_date = [o['due date'] for o in ordres]

# ---- DECISION VARIABLES ---------------------------------------------------
# assignment: x[i,t,m] == 1 if order i processed on machine m at time t
x = model.addVars(I, T, M, vtype=GRB.BINARY, name="x")

# completion times
C = model.addVars(I, vtype=GRB.INTEGER, name="C")

# tardiness ≥ max{0, C_i − d_i}, earliness ≥ max{0, d_i − C_i}
Tvar = model.addVars(I, vtype=GRB.CONTINUOUS, name="T")   # tardiness
Evar = model.addVars(I, vtype=GRB.CONTINUOUS, name="E")   # earliness

# ---- COMPLETION‑TIME DEFINITION -------------------------------------------
for i in I:
    for t in T:
        for m in M:
            # If any (t,m) is chosen, C_i must be at least that t
            model.addConstr(C[i] >= t * x[i, t, m])

# ---- TARDINESS / EARLINESS DEFINITION -------------------------------------
for i in I:
    # T_i ≥ C_i − d_i
    model.addConstr(Tvar[i] >= C[i] - due_date[i])
    # E_i ≥ d_i − C_i
    model.addConstr(Evar[i] >= due_date[i] - C[i])

# ---- OBJECTIVE ------------------------------------------------------------
model.setObjective(
    K * gp.quicksum(Tvar[i] for i in I) + gp.quicksum(Evar[i] for i in I),
    GRB.MINIMIZE
)

for i, ordre in enumerate(ordres):
    model.addConstr(
        gp.quicksum(x[i, t, m] for m in range(machines)
                    for t in range(ordre['due date'])) == ordre['duration'],name=f"assign_{i}")

for m in range(machines):
    for t in range(timeline):
        model.addConstr(
            gp.quicksum(
                x[i, t, m]
                for i, ordre in enumerate(ordres) )<= 1,
            name=f"machine_{m}time{t}")

model.optimize()


# x = {(i, t, m): var}  avec var.X > 0.5 si l’ordre i commence à t sur machine m
# ordres = {i: {'numéro': id, 'duration': d}}

schedule = []

if model.status == GRB.OPTIMAL:
    for (i, t, m), var in x.items():
        if var.X > 0.5:
            duration = 1  # chaque x[i, t, m] = 1 correspond à une unité de temps
            order_id = ordres[i]['numéro']
            schedule.append({
                'Machine': m,
                'Start': t,
                'Duration': duration,
                'Order': f"Ordre {order_id}",
                'Index': i
            })
else:
    print("Pas de solution optimale trouvée.")

# Regrouper par machine
machines_used = sorted(set(task['Machine'] for task in schedule))
machine_to_y = {m: idx for idx, m in enumerate(machines_used)}

# Palette de couleurs unique par ordre
unique_orders = sorted(set(task['Index'] for task in schedule))
order_to_color = {i: plt.cm.tab20(i % 20) for i in unique_orders}

# Tracer
fig, ax = plt.subplots(figsize=(12, 6))

for task in schedule:
    y = machine_to_y[task['Machine']]
    color = order_to_color[task['Index']]
    ax.barh(y, task['Duration'], left=task['Start'], height=0.4,
            color=color, edgecolor='black')
    # Affichage facultatif de l'étiquette sur chaque bloc :
    # ax.text(task['Start'] + task['Duration'] / 2, y, task['Order'],
    #         va='center', ha='center', fontsize=8, color='white')

# Axe y avec noms de machines
ax.set_yticks(range(len(machines_used)))
ax.set_yticklabels([f"Machine {m}" for m in machines_used])
ax.set_xlabel("Temps")
ax.set_title("Emploi du temps des ordres par machine")
ax.grid(True)

# Légende avec une seule entrée par ordre
legend_patches = []
already_seen = set()
for task in schedule:
    idx = task['Index']
    label = task['Order']
    if label not in already_seen:
        legend_patches.append(mpatches.Patch(color=order_to_color[idx], label=label))
        already_seen.add(label)

ax.legend(handles=legend_patches, title="Ordres", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()


# -


