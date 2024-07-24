from mnr_cond_adapted import gen_current
from geometry_generator import generate
from os import mkdir, replace
import os.path as path

if not ( path.exists("./Results/") ):
    mkdir("./Results/")

for i in range(1):
    coord_file = generate(i)
    f_id = coord_file.split("geom_")[1].split(".dat")[0]

    geom_folder = f"./Results/{f_id}"
    if not ( path.exists(geom_folder) ):
        mkdir(geom_folder)

    replace(f"{coord_file}", f"{geom_folder}/{coord_file}")
    coord_file = f"{geom_folder}/{coord_file}"

    if not ( path.exists(f"./Results/{f_id}/Resistances") ):
        mkdir(f"./Results/{f_id}/Resistances")

    gen_current(coord_file)
