import numpy as np
import numpy.random as rng
from shapely.geometry import LineString
from shapely import intersection

# Constants in nm
#WIRE_SUPERFICIAL_DENSITY = .001
ELECTRODE_LENGTH = 22_000
ELECTRODES_DISTANCE = 20_000
WIRE_LENGTH_MEAN = 5_000
WIRE_LENGTH_STANDARD_DEVIATION = 2_400
NUMBER_OF_JUNCTIONS = 900
# leeway for wire generation around nodes
pad_x = ELECTRODES_DISTANCE * .01
pad_y = ELECTRODE_LENGTH * .05

# Functions

def insert_electrodes():
    ele_L = [(0, 0),
             (0, ELECTRODE_LENGTH)]
    ele_R = [(ELECTRODES_DISTANCE, 0),
             (ELECTRODES_DISTANCE, ELECTRODE_LENGTH)]
    return ele_L + ele_R


def insert_wire():
    x_grid_size = ELECTRODES_DISTANCE + 2*pad_x
    y_grid_size = ELECTRODE_LENGTH + 2*pad_y

    start = (rng.random()*x_grid_size - pad_x,
             rng.random()*y_grid_size - pad_y)


    wire_size = -1
    while wire_size < 0:
        wire_size = rng.normal(WIRE_LENGTH_MEAN,
                               WIRE_LENGTH_STANDARD_DEVIATION)
    
    angle = rng.random()*2*np.pi
    end = [start[0] + np.cos(angle)*wire_size,
           start[1] + np.sin(angle)*wire_size]
    
    if (end[0] < -pad_x):
          end[0] = -pad_x
    if (end[0] > x_grid_size -pad_x):
          end[0] = x_grid_size -pad_x
    if (end[1] < -pad_y):
          end[1] = -pad_y
    if (end[1] > y_grid_size -pad_y):
          end[1] = y_grid_size -pad_y
    
    end = (end[0], end[1])
    
    return [start, end]


# Main
def generate(id: int=0):
    num_junc = 0

    point_list = []
    wire_list = []
    point_list += insert_electrodes()
    wire_list += [LineString([point_list[0], point_list[1]])]
    wire_list += [LineString([point_list[2], point_list[3]])]

    while(num_junc < NUMBER_OF_JUNCTIONS):
        wire_points = insert_wire()
        new_wire = LineString(wire_points)

        this_wire_num_junc = 0
        for wire in wire_list:
            if new_wire.intersects(wire):
                inter_x = intersection(wire, new_wire).x
                if inter_x >= 0 and inter_x <= ELECTRODES_DISTANCE:
                    # only count if junction in between electrodes
                    this_wire_num_junc+=1

        if (num_junc + this_wire_num_junc) > NUMBER_OF_JUNCTIONS:
            pass
        else:
            num_junc += this_wire_num_junc
            wire_list += [new_wire]
            point_list += wire_points
        

    # Output
    # output_file = (f"geom"
    #                f"_wl{WIRE_LENGTH_MEAN}"
    #                f"_wsd{WIRE_LENGTH_STANDARD_DEVIATION}"
    #                f"_wc{WIRE_SUPERFICIAL_DENSITY}"
    #                f"_el{ELECTRODE_LENGTH}"
    #                f"_ed{ELECTRODES_DISTANCE}"
    #                f"_.dat")
    output_file = (f"./geom"
                #    f"_wl{WIRE_LENGTH_MEAN}"
                #    f"_wsd{WIRE_LENGTH_STANDARD_DEVIATION}"
                f"_nj{NUMBER_OF_JUNCTIONS}"
                #    f"_el{ELECTRODE_LENGTH}"
                #    f"_ed{ELECTRODES_DISTANCE}"
                f"_{id}.dat")
    #output_file = "test.dat"
    with open(output_file, "w") as file:
        for point in point_list:
            point_text = f"{point[0]} {point[1]}\n"
            file.write(point_text)
    return output_file

if __name__ == "__main__":
     generate()
