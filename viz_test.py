from datahandling.visualisation import Plotter, look_slices
from datahandling.DNS_Loader import parse_n_coordinates
from gc import collect
import numpy as np
import os
import sys


def sizeof(x):
    return sys.getsizeof(x)


def middle_point(ent):
    return (ent[1:] + ent[:-1])/2


dns_filename_rho = os.path.join("dns_180", "dns_lata_1.sauv.lata.0.RHO")
dns_filename_pressure = os.path.join(
    "dns_180", "dns_lata_1.sauv.lata.0.PRESSURE")
dns_filename_coord = os.path.join(
    "dns_180", "dns_lata_1.sauv.lata.grid_geom2.coord")
coord = []
for direction in ["x", "y", "z"]:
    coord.append(np.fromfile(
        f"{dns_filename_coord}{direction}", dtype=np.float32))
rho_dns = np.fromfile(dns_filename_rho, dtype=np.float32)
p_dns = np.fromfile(dns_filename_pressure, dtype=np.float32)

# p_th = np.mean(p_dns)

# Constants
# ath pressure is supposed to be 101 325 Pa
r_air = 287.058
p_ath = 1.5e5
n_vertex, n = parse_n_coordinates("dns_180/dns_lata_1.sauv.lata")
# Computing temperature and reshaping it
T_dns = p_ath / (rho_dns * r_air)
T_dns = T_dns.reshape(n, order="F")

# Release memory for rho and P
del rho_dns
del p_dns
collect()

plotter = Plotter(T_dns, coord)
# plotter.plot_pavement()
plotter.plot_slices()
