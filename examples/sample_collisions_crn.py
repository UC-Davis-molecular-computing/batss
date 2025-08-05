import math
import numpy as np
import ppsim as pp
import random
from tqdm import tqdm

def main():
    a,b,u = pp.species('A B U')
    rxns = [
        a+b >> 2*u,
        a+u >> 2*a,
        b+u >> 2*b,
    ]

    trials_exponent = 5
    pop_exponent = 7
    n = 10 ** pop_exponent
    p = 0.51 
    a_init = int(n * p)
    b_init = n - a_init
    inits = {a: a_init, b: b_init}
    trials = 10 ** trials_exponent

    sim = pp.Simulation(inits, rxns, simulator_method="crn")

    # r = math.ceil(math.sqrt(n))
    r = 0
    print(f'{n=}, {r=}')

    ls_pp = []
    ls_dir = []
    import time
    
    for i in tqdm(range(trials)):
        t1 = time.perf_counter()

        l_dir = sim.simulator.sample_collision_directly(n, r) # type: ignore
        ls_dir.append(l_dir)

        t2 = time.perf_counter()
        
        u = random.random()
        l_pp = sim.simulator.sample_collision(r, u, has_bounds=False) # type: ignore
        ls_pp.append(l_pp)

        t3 = time.perf_counter()

        # print(f'time to sample directly: {1000*(t2-t1):.6f}ms time to sample with pp: {1000*(t3-t2):.6f}ms')
        
    ls_pp = np.array(ls_pp)
    ls_dir = np.array(ls_dir)
    print(ls_pp)
    print(ls_dir)
    # print(f'fast:   { ls_pp.mean():7.2f}, stddev: { ls_pp.std():6.2f}')
    # print(f'direct: {ls_dir.mean():7.2f}, stddev: {ls_dir.std():6.2f}')
    print(f'fast:   { ls_pp.mean():7.2f}')
    print(f'direct: {ls_dir.mean():7.2f}')

if __name__ == "__main__":
    main()