import json

lines = []
with open("logs/metrics.jsonl") as f:
    for line in f:
        line = line.strip()
        if line:
            lines.append(json.loads(line))

updates = [x for x in lines if "update" in x and "kill_rate" in x]
print("Total update records:", len(updates))

def fmt(x):
    upd = x.get("update", 0)
    stg = x.get("curriculum_stage", 0)
    kr = x.get("kill_rate", 0.0)
    ret = x.get("mean_return", 0.0)
    pac_sc = x.get("pacman_score", 0.0)
    deaths = x.get("ghost_deaths", 0.0)
    ttk = x.get("time_to_kill", 0.0)
    spd = x.get("speed_mean", 0.0)
    ent = x.get("entropy", 0.0)
    kl = x.get("approx_kl", 0.0)
    lr_sc = x.get("kl_lr_scale", 0.0)
    gn_a = x.get("grad_norm_a", 0.0)
    return (
        f"Upd {upd:>4}: stg={stg} kill_rate={kr:.1%} ret={ret:>5.1f} pac_sc={pac_sc:>6.1f} "
        f"deaths={deaths:>4.2f} ttk={ttk:>5.1f} spd={spd:>4.2f} ent={ent:>5.2f} kl={kl:>6.4f} "
        f"lr_sc={lr_sc:>4.2f} gn_a={gn_a:>5.2f}"
    )

print("--- Progression Every 25 Updates ---")
for x in updates[::25]:
    print(fmt(x))

print("\n--- Last 30 Updates ---")
for x in updates[-30:]:
    print(fmt(x))