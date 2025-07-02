import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import pandas as pd

# 1) Définir automatiquement origin = maintenant

# Début : au format(year,month,day,hour,minute)
start=(2025, 7, 7, 8, 0)
origin = datetime(start[0],start[1],start[2],start[3],start[4])
# Fin   : au format(year,month,day,hour,minute)
finish=(2025, 7, 12, 11, 0)
end    = datetime(finish[0],finish[1],finish[2],finish[3],finish[4])

# 2) Créneaux de 30 min
slot_duration = timedelta(minutes=30)
timeline = int((end - origin) / slot_duration)


model = gp.Model("timetable_discrete")

data = pd.read_csv('Machine_Cadence.csv', sep=';', encoding='cp1252')

data.drop( data [data['Cork Type']=='MGF'].index, inplace=True)


fire_machines=[]
all_machines=list(data['Machine'].unique())
for m in all_machines:
    if m[:3]!='LAS':#Non-laser Machine
        fire_machines.append(m)

machines=[]

for machine in fire_machines:
    dico={}
    a=data[data['Machine']==fire_machines[0]]['Cork Type']
    n=len(a)
    L=[]
    for i in range(n):
        L.append({a.iloc[i]:data[data['Machine']==fire_machines[0]]['Cadence'].iloc[i]})
    dico[f'{machine}']=L
    machines.append(dico)

# print(machines)


# supprimer une des machines

n_machines = 13
timeline = 300

data = pd.read_csv('DB_OF.csv', sep=';', encoding='cp1252')
commandes=data['OF'].unique()
orders=[]
for commande in commandes:
    dico={}
    dico["référence commande"]=commande
    dico["due date"]=pd.to_datetime(data[data['OF']==commande]['Date'].iloc[0])
    dico["quantité restante"]=data[data['OF']==commande]['Remaining_Qty'].iloc[0]
    dico["cork type"]=data[data['OF']==commande]['Family'].iloc[0]
    dico["double"]=1 if data[data['OF']==commande]['Type'].iloc[0][-5:] == "DOBLE" else 0
    dico["stamp"] = 1
    orders.append(dico)
print(orders)

# ---- PARAMETERS -----------------------------------------------------------
I        = range(len(orders))          # order indices
M        = range(n_machines)             # machine indices
T        = range(timeline)             # time indices
Kret        = 10
Kav = 1
Kchange = 1
due_date = [int((o['due date'] - origin) / slot_duration) for o in orders]
stamps   = [o['stamp'] for o in orders]

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

# ---- MAXIMUM MACHINES RUNNING AT THE SAME TIME ----------------------------

for i in I:
    for t in T:
        # Sum on m x_imt<= number of stamps
        model.addConstr(gp.quicksum(x[i,t,m] for m in M)<=stamps[i])

    
#-----------------------------------------------------------------------------

setup_time = [[1]*len(orders) for _ in orders]


# --- CHANGING TIME-----------------------------------------------------------
for m in M:
    for i in I:
        for j in I:
            if i == j: 
                continue
            s_ij = setup_time[i][j]
            # si setup_time=0, pas besoin de contraindre
            if s_ij > 0:
                # pour tout t où t + s_ij reste dans l'horizon
                for t in range(timeline - s_ij):
                    for tau in range(1, s_ij+1):
                        model.addConstr(
                            x[i, t, m] + x[j, t + tau, m] <= 1,
                            name=f"setup_m{m}_i{i}_j{j}_t{t}_tau{tau}"
                        )


echelle_temporelle=0.5

timeline=40

Cadences = {'simple':{'VT':8000, 'VE':6400},'double':{'VT':6750}}

N = len(orders)
for i in range(N):
    cork_type=orders[i]['cork type']
    double = "double" if orders[i]['double'] else "simple"
    quantite_restante=float(orders[i]["quantité restante"].replace(',', '.'))*1000
    if double=="simple":
        model.addConstr(gp.quicksum(x[i,t,m]*Cadences[double][cork_type]*echelle_temporelle for t in range(timeline) for m in M)==(quantite_restante//Cadences[double][cork_type]+1)*Cadences[double][cork_type])
    if double=='double':
        model.addConstr(gp.quicksum(0.5*x[i,t,m]*
                                     Cadences['simple'][cork_type]*echelle_temporelle 
                                     for t in range(timeline) 
                                     for m in M 
                                     if list(machines[m])[0][:3]=='MIX')+
                                     gp.quicksum(0.5*x[i,t,m]*Cadences['double'][cork_type]*echelle_temporelle 
                                                 for t in range(timeline) 
                                                 for m in M 
                                                 if list(machines[m])[0][:3]=='VTF')-quantite_restante>=0)
        model.addConstr(gp.quicksum(0.5*x[i,t,m]*
                                    Cadences['simple'][cork_type]*echelle_temporelle 
                                    for t in range(timeline) 
                                    for m in M if list(machines[m])[0][:3]=='MIX')+
                                    gp.quicksum(0.5*x[i,t,m]*Cadences['double'][cork_type]*echelle_temporelle 
                                                for t in range(timeline) 
                                                for m in M if list(machines[m])[0][:3]=='VTF')-quantite_restante>=
                                                    (Cadences['simple'][cork_type]*echelle_temporelle)*0.5+Cadences['double'][cork_type]*echelle_temporelle+1)

# ---- OBJECTIVE ------------------------------------------------------------
model.setObjective(
    Kret * gp.quicksum(Tvar[i]**2 for i in I) + Kav*gp.quicksum(Evar[i]**2 for i in I)+ Kchange*gp.quicksum((x[i,t,m]-x[i,t+1,m])*(x[i,t,m]-x[i,t+1,m]) for t in range(timeline-1) for m in M for i in I),
    GRB.MINIMIZE
)

# for i, ordre in enumerate(orders):
#     model.addConstr(
#         gp.quicksum(x[i, t, m] for m in range(n_machines)
#                     for t in range(ordre['due date'])) == ordre['duration'],name=f"assign_{i}")

for m in range(n_machines):
    for t in range(timeline):
        model.addConstr(
            gp.quicksum(
                x[i, t, m]
                for i, ordre in enumerate(orders) )<= 1,
            name=f"machine_{m}time{t}")

model.optimize()


# x = {(i, t, m): var}  avec var.X > 0.5 si l’ordre i commence à t sur machine m
# ordres = {i: {'numéro': id, 'duration': d}}

schedule = []

if model.status == GRB.OPTIMAL:
    for (i, t, m), var in x.items():
        if var.X > 0.5:
            duration = 1  # chaque x[i, t, m] = 1 correspond à une unité de temps
            order_id = orders[i]['référence commande']
            schedule.append({
                'Machine': m,
                'Start': t,
                'Duration': duration,
                'Order': f"Ordre {order_id}",
                'Index': i
            })
else:
    print("Pas de solution optimale trouvée.")
print(schedule)
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

finishing_times={}
for task in schedule:
    if task['Index'] in finishing_times.keys():
        if finishing_times[task['Index']]<task['Index']+task['Duration']:
            finishing_times[task['Index']]=task['Index']+task['Duration']
    else:
        finishing_times[task['Index']]=task['Index']+task['Duration']
        
avance=[]
for ordre in orders:
    print(ordre['référence commande'])
    avance.append(ordre['due date']-finishing_times[ordre['référence commande']-1])
    
indices = [i+1 for i in range(len(finishing_times))]
legendes = [f'Ordre{i+1}' for i in range(len(finishing_times))]
plt.bar(indices, avance, color='blue', edgecolor='black')

# Remplacer les valeurs de l'axe x par des labels
plt.xticks(indices, legendes, rotation=45)

plt.xlabel("Ordres")
plt.ylabel("Durée totale")
plt.title("Durée par ordre")
plt.tight_layout()
plt.show()