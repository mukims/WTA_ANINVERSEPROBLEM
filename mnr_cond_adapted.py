from shapely.geometry import MultiLineString, Point
import matplotlib.pyplot as plt
import networkx as nx
import math
import numpy as np
from collections import defaultdict
from itertools import combinations
from cvxopt.base import matrix as m
from cvxopt.blas import gemv

from minres import minres

Amatrix = None

def Gfun(x,y,trans='N'):
    ''' Function that passes matrix A to the symmlq routine which solves Ax=B.'''
    gemv(Amatrix,x,y,trans)
    #my_gemv(Amatrix,x,y)

def distance_scale(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt( (x1 - x2)**2 + (y1 - y2)**2  )

def gen_current(coord_file: str, i0: float=0.01, iif: float=2.5, istep: float=0.01, mean_EXPON = 1.1, std_dev_EXPON=0.05):
    global Amatrix
    # Parameters for solving the linear system of equations
    TOL = 1e-15
    #EXPON = abs(np.random.normal(mean_EXPON, std_dev_EXPON))
    SHOW = False
    MAXIT = None
    # proportionality constant for single junction G = alph*icc
    ALPH = 0.05
    # Resistances of junctions in off and on states
    R_OFF = 10000.0
    R_ON = 12.0   
    # Characteristic resistivity (silver)
    RHO_0 = 22.63676 # Nano Ohm m
    # Characteristic wire diameter
    D0 = 60.0 # Nm
    # Cross section areas
    A0 = math.pi * (D0 / 2.0)**2
    file_id = coord_file.split("geom_")[1].split(".dat")[0]

    # Parameters for current scan
    num_i = np.arange(i0,iif,istep)


    # Read coordinates
    coords = np.loadtxt(coord_file)
    nwires_plus_leads = int(len(coords)/2)

    # Separate data in odd and even lines corresponding to the
    # start and end coordinates of each wire sequentially
    even_lines = [tuple(coords[i]) for i in range(len(coords)) if i%2 == 0]
    odd_lines = [tuple(coords[i]) for i in range(len(coords)) if i%2 == 1]
    coords = [(even_lines[i], odd_lines[i]) for i in range(len(even_lines))]
    mlines = MultiLineString(coords).geoms

    # dictionary containing wire index: [list of wires connected to it]
    wire_touch_list = defaultdict(list)
    nintersections = 0
    for comb in combinations(enumerate(mlines), 2):
        id1, line1 = comb[0]
        id2, line2 = comb[1]
        if line1.intersects(line2):
            nintersections += 1
            wire_touch_list[id1].append(id2)
            wire_touch_list[id2].append(id1)

    # dictionary containing wire index: [label nodes following MNR mapping]
    wire_touch_label_list = defaultdict(list)
    each_wire_inter_point_storage = defaultdict(list)
    new_pos_vec = defaultdict(list)
    label = 2

    # Starting creating the new node labelling according to MNR mapping
    for i in wire_touch_list.items(): #iter(wire_touch_list.viewitems()):
        for j in range(len(i[1])):
            cpoint = mlines[i[0]].intersection(mlines[i[1][j]])
            npoint = (cpoint.x,cpoint.y)
            each_wire_inter_point_storage[i[0]].append(npoint)
            
            if i[0] > 1:
                wire_touch_label_list[i[0]].append(label)
                new_pos_vec[label].append(npoint)
                label += 1
            else:
                wire_touch_label_list[i[0]].append(i[0])
                new_pos_vec[i[0]].append(npoint)


    maxl = label # dimension of the resistance matrix

    # list containing the length segments of each wire (if it has a junction)
    each_wire_length_storage = [[] for _ in range(nwires_plus_leads)]  

    # Routine that obtains the segment lengths on each wire
    for i in each_wire_inter_point_storage:
        
        point_ini = Point(mlines[i].coords[0])  # Initial point of the wire
        point_fin = Point(mlines[i].coords[1])  # Final point of the wire
        wlength = point_ini.distance(point_fin) # Whole length
        wire_points = each_wire_inter_point_storage[i]

        dist = [0.0]*(len(wire_points)+1)
        for j in range(len(wire_points)):
            point = Point(wire_points[j])
            dist[j] = point_ini.distance(point)

        dist[-1] = wlength  # Whole length stored on the last component of dist vector.
        dist.sort() # Sorting in crescent order

        dist_sep = [0.0]*len(dist)
        dist_sep[0] = dist[0]
        dist_sep[1:len(dist)] = [dist[k]-dist[k-1] for k in range(1,len(dist))] # Segment lengths calculated for a particular wire
        each_wire_length_storage[i].append(dist_sep)

    # starting building resistance matrix
    mr_matrix = np.zeros((maxl, maxl))

    # matrix storing the labels of ONLY junctions
    mr_matrix_info = np.zeros((maxl,maxl))

    # Matrix storing coordinates of intersection points
    interpos_matrix = np.zeros((maxl,maxl),dtype=object)
    innerpos_matrix = np.zeros(maxl,dtype=object)
    pos_vec = np.zeros(maxl,dtype=object)

    # list to store all junction resistances
    resis_list = []

    # file containing information of mr_matrix
    
    matrix_mr = open(f'./Results/{file_id}/matrix_mr.dat', 'w')

    # start graph that will store only inner edge connections
    mnr_nodes = range(maxl)
    G = nx.Graph()
    G.add_nodes_from(mnr_nodes)

    # Procedure to build the resistance matrix (mr_matrix) which assumes that the wires are not in the same potential.
    # The procedure is:
    # For each iwire wire...
    for iwire in range(nwires_plus_leads):
        # if each_wire_inter_point_storage[iwire] is not empty, the procedure can start.
        if each_wire_inter_point_storage[iwire]:
            # First we obtain the capacitance matrix elements related to the internal "capacitances"...
            # This procedure is similar to the one used above for building multilinestring segments.
            # Scan all the junction coordinate points stored in each_wire_inter_point_storage[iwire].
            for j, pointj in enumerate(each_wire_inter_point_storage[iwire]):
                # Reserve a particular point of this list.
                point = Point(pointj)
                # Scan all the junction coordinate points stored in each_wire_inter_point_storage[iwire].
                for i, pointw in enumerate(each_wire_inter_point_storage[iwire]):
                    # Reserve another point of this list.
                    comp_pointw = Point(pointw)
                    # Calculate the distance between point - comp_pointw
                    inter_dist = point.distance(comp_pointw)
                    # A 4 digit precision for this distance must be imposed otherwise, a comparison between exact numbers can fail.
                    round_inter_dist = round(inter_dist, 4)
                    # Check if each_wire_length_storage[iwire] contains a segment length that matches round_inter_dist.
                    # If it does, we found a capacitance matrix element correspondent to an inner "capacitance".
                    for il in each_wire_length_storage[iwire][0]:
                        value = float(il)
                        value = round(value,4)
                        if value == round_inter_dist and value != 0:
                            if iwire != 0 and iwire != 1 and mr_matrix[wire_touch_label_list[iwire][i], wire_touch_label_list[iwire][j]] == 0.0:
                                inner_resis = (float(value) * RHO_0 / A0)
                                #inner_resis = inner_resis/1000.0
                                # ELEMENT FOR mr_matrix FOUND! Its labels are stored in wire_touch_label_list.
                                mr_matrix[wire_touch_label_list[iwire][i], wire_touch_label_list[iwire][j]] = -1.0/inner_resis
                                mr_matrix[wire_touch_label_list[iwire][j], wire_touch_label_list[iwire][i]] = -1.0/inner_resis
                                
                                innerpos_matrix[wire_touch_label_list[iwire][i]] = comp_pointw
                                innerpos_matrix[wire_touch_label_list[iwire][j]] = point
                                G.add_edge(wire_touch_label_list[iwire][i],wire_touch_label_list[iwire][j])

                                matrix_mr.write('%s %s %s %s\n' % (wire_touch_label_list[iwire][i], wire_touch_label_list[iwire][j], inner_resis, -1.0/inner_resis))
                                matrix_mr.write('%s %s %s %s\n' % (wire_touch_label_list[iwire][j], wire_touch_label_list[iwire][i], inner_resis, -1.0/inner_resis))

                # Procedure to find capacitance matrix elements for the junctions...
                # Scan the list (wire_touch_list) which stores the label of wires to which iwire is connected.
                for label in wire_touch_list[iwire]:
                    # For a particular wire (labelled as label) in wire_touch_list, scan its junction coordinate points stored in (each_wire_inter_point_storage[label].
                    for kk, pointk in enumerate(each_wire_inter_point_storage[label]):
                        # Reserve one of the junction points.
                        pointk = Point(pointk)
                        # Calculate the distance between point - pointk
                        inter_dist = point.distance(pointk)
                        # A 4 digit precision for this distance must be imposed otherwise, a comparison between exact numbers can fail.
                        round_inter_dist = round(inter_dist, 4)
                        # If round_inter_dist is ZERO, it means we FOUND a junction capacitance element that is stored in mr_matrix.
                        # Its value is computed from the Gaussian distribution.
                        if round_inter_dist == 0 and mr_matrix[wire_touch_label_list[iwire][j], wire_touch_label_list[label][kk]]== 0:

                            resis = R_OFF

                            mr_matrix[wire_touch_label_list[iwire][j], wire_touch_label_list[label][kk]] = -1.0/resis
                            mr_matrix[wire_touch_label_list[label][kk], wire_touch_label_list[iwire][j]] = -1.0/resis

                            mr_matrix_info[wire_touch_label_list[iwire][j], wire_touch_label_list[label][kk]] = 1.0
                            mr_matrix_info[wire_touch_label_list[label][kk], wire_touch_label_list[iwire][j]] = 1.0

                            if label == 0:
                                    interpos_matrix[wire_touch_label_list[iwire][j], wire_touch_label_list[label][kk]] = point
                            elif label == 1:
                                    interpos_matrix[wire_touch_label_list[iwire][j], wire_touch_label_list[label][kk]] = point
                            else:
                                    interpos_matrix[wire_touch_label_list[iwire][j], wire_touch_label_list[label][kk]] = pointk

                            if iwire == 0:
                                    interpos_matrix[wire_touch_label_list[iwire][j], wire_touch_label_list[label][kk]] = pointk
                            elif iwire == 1:
                                    interpos_matrix[wire_touch_label_list[iwire][j], wire_touch_label_list[label][kk]] = pointk
                            else:
                                    interpos_matrix[wire_touch_label_list[iwire][j], wire_touch_label_list[label][kk]] = point


                            pos_vec[wire_touch_label_list[label][kk]] = pointk
                            pos_vec[wire_touch_label_list[iwire][j]] = pointk

                            resis_list.append(resis)
                            matrix_mr.write('%s %s %s %s\n' % (wire_touch_label_list[iwire][j], wire_touch_label_list[label][kk], resis, -1.0/resis))
                            matrix_mr.write('%s %s %s %s\n' % (wire_touch_label_list[label][kk], wire_touch_label_list[iwire][j], resis, -1.0/resis))

    # array storing the index of the non-zero elements of mr_matrix_info (only on its upper diagonal part)
    neigh_mr = np.nonzero(np.triu(mr_matrix_info))

    # Sum the non-zero elements on each row of mr_matrix
    sum_rows_mr = mr_matrix.sum(1)

    # Place them in the diagonal and change their signal with abs()
    np.fill_diagonal(mr_matrix, abs(sum_rows_mr))


    for i in range(maxl):
        matrix_mr.write('%s %s %s %s\n' % (i, i, mr_matrix[i,i], mr_matrix[i,i]))

    matrix_mr.close()


    # file storing resistance versus ic
    resist_info = open(f'./Results/{file_id}/conductance_icc.dat','w')

    list_ic = []
    list_conductance = []
    values = np.abs(np.random.normal(mean_EXPON, std_dev_EXPON, len(neigh_mr[0])))
    values_list= list(values)
    np.savetxt(f"./Results/alphalist.dat",np.column_stack(values_list))
    
    
    
    inner_edges = G.edges()
    frame = 0
    files = []
    #dpi = 200
    fps = 1
    dpi = 200
    npot = 20


    # Starting current loop
    for ic in num_i:
        
        # Initiating fixed current vector
        iv = np.zeros(maxl)
        iv[0] = +ic
        iv[1] = -ic
        Imatrix = m(iv)

        mr_matrix_form = m(mr_matrix)
        Amatrix = mr_matrix_form
        elec_pot_mr = minres( Gfun, Imatrix, show=SHOW, rtol=TOL, itnlim=MAXIT)

        resistance = (elec_pot_mr[0][0] - elec_pot_mr[0][1])/ic
        conductance = 1.0/resistance

        resist_info.write('%s   %s\n' % (ic, conductance))

        list_ic.append(ic)
        list_conductance.append(conductance)    
        # current map (current information is saved point-by-point in an .txt that will be processed in gnuplot)
        curr_file = open("./Results/current_%04d.dat"%frame,"w")
        for ie in inner_edges:
        	pinix = innerpos_matrix[ie[0]].x
        	piniy = innerpos_matrix[ie[0]].y
        
        	pfinx = innerpos_matrix[ie[1]].x
        	pfiny = innerpos_matrix[ie[1]].y
        	aslope = (pfiny - piniy)/(pfinx - pinix)
        	binter = 0.5 * (pfiny+piniy - aslope*(pfinx + pinix) )
        	ix = np.linspace(pinix, pfinx, num=npot,endpoint=True)
        	iy = ix * aslope + binter
        
        	diff = abs(elec_pot_mr[0][ie[1]] - elec_pot_mr[0][ie[0]]) 
        	curr = diff*(abs(mr_matrix[ie[1],ie[0]]))

        	for ii in range(npot):
            # the minus sign in iy may be necessary depending on how the schematic figure is mirrored from the experimental image
            		curr_file.write('%s  %s  %s\n' % (ix[ii], -iy[ii],curr))
        curr_file.close()
        
        frame += 1


        new_resis_list = []
        # Loop over all inter-wire connections 
        for i in range(len(neigh_mr[0])):

            # ddp along the junction
            #diffmr = abs(pot[neigh_mr[0][i]]-pot[neigh_mr[1][i]])
            diffmr = abs(elec_pot_mr[0][int(neigh_mr[0][i])] - elec_pot_mr[0][int(neigh_mr[1][i])])

            # current passing through the junction
            jcurr = diffmr/(abs(1.0/mr_matrix[int(neigh_mr[0][i]),int(neigh_mr[1][i])]))

            # resistance updated as a function of its current
            new_resis=1.0/(ALPH*jcurr**values_list[i])
            

            # thresholds (the junction resistance cannot be bigger than Roff or smaller than Ron)
            if new_resis > R_OFF:
                new_resis = R_OFF
            if new_resis < R_ON:
                new_resis = R_ON

            new_resis_list.append(new_resis)
            
            # modify resistance of the junction
            mr_matrix[neigh_mr[0][i],neigh_mr[1][i]] = -1.0/new_resis
            mr_matrix[neigh_mr[1][i],neigh_mr[0][i]] = -1.0/new_resis

        np.savetxt(f"./Results/{file_id}/Resistances/res_{ic:.2f}.dat", np.column_stack(new_resis_list))

        ###### uncomment to get the dynamics of resistances in a histogram #######
        #new_resis_hist = np.asarray(new_resis_list)

        #figr = plt.figure(facecolor='w')
        #n3, bins3, patches3 = plt.hist(new_resis_hist, bins=100, facecolor='red', alpha=0.75)
        #plt.ylim([0, nintersections*0.4])
        #plt.xlim([Ron-5, Roff+1000])
        #plt.xlabel('Resistance')
        #plt.ylabel('Occurrence')
        #plt.grid(True)

        #fdist = 'dist_%04d.png'%dframe
        #plt.savefig(fdist, bbox_inches='tight', dpi=dpi)
        #files_dist.append(fdist)
        #plt.close(figr)
        #figr.clf()
        ###### uncomment to get the dynamics of resistances in a histogram #######

        # Reset diagonal elements before updating Mr!
        np.fill_diagonal(mr_matrix, 0.0)

        # Adds the non-zero elements on each row of Mr.
        sum_rows_mr = mr_matrix.sum(1)
        
        # Place the sum in the diagonal of Mr.
        np.fill_diagonal(mr_matrix, abs(sum_rows_mr)) 


    resist_info.close()
    print(f'{file_id} - done')


