import sys, time, json
sys.dont_write_bytecode = True
sys.path.insert(0, ".")
import sim
t0 = time.time()
m = sim.run_matcher(sim.BASE)
print({k: v for k, v in m.items() if k != "sel"}, "secs", round(time.time() - t0, 1))
short, tight, q, pf, av, log = sim.short_cells(sim.BASE)
print("short", len(short), "shortfall", sum(short.values()), "tight", len(tight))
print("milp nec", sim.milp_min_phrase(sim.BASE, sim.TG, q, False), "suf", sim.milp_min_phrase(sim.BASE, sim.TG, q, True))
print("G_C", sim.G_C, "G_U", sim.G_U, "G_F", sim.G_F)
