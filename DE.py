# -*- coding: utf-8 -*-
# @Time    : 2021/11/10 8:54
# @Author  : Praise
# @File    : DE_MDVRPTW.py
# obj:
import math
import random
import numpy as np
import copy
import xlsxwriter
import matplotlib.pyplot as plt
import csv
import sys
import pandas as pd






# DE/rand/1/bin
def muSol(model,v1):
    x1=model.sol_list[v1].node_id_list
    while True:
        v2=random.randint(0,len(model.demand_id_list)-1)
        if v2!=v1:
            break
    while True:
        v3=random.randint(0,len(model.demand_id_list)-1)
        if v3!=v2 and v3!=v1:
            break
    x2=model.sol_list[v2].node_id_list
    x3=model.sol_list[v3].node_id_list
    mu_x=[min(int(x1[i]+model.F*(x2[i]-x3[i])),len(model.demand_id_list)-1) for i in range(len(model.demand_id_list)) ]
    return mu_x

def crossSol(model,vx,vy):
    cro_x=[]
    for i in range(len(model.demand_id_list)):
        if random.random()<model.Cr:
            cro_x.append(vy[i])
        else:
            cro_x.append(vx[i])
    cro_x=adjustRoutes(cro_x,model)
    return cro_x

def plotObj(obj_list):
    plt.rcParams['font.sans-serif'] = ['SimHei'] #show chinese
    plt.rcParams['axes.unicode_minus'] = False  # Show minus sign
    plt.plot(np.arange(1,len(obj_list)+1),obj_list)
    plt.xlabel('Iterations')
    plt.ylabel('Obj Value')
    plt.grid()
    plt.xlim(1,len(obj_list)+1)
    plt.show()

def outPut(model):
    work=xlsxwriter.Workbook('result.xlsx')
    worksheet=work.add_worksheet()
    worksheet.write(0, 0, 'cost_of_time')
    worksheet.write(0, 1, 'cost_of_distance')
    worksheet.write(0, 2, 'opt_type')
    worksheet.write(0, 3, 'obj')
    worksheet.write(1,0,model.best_sol.cost_of_time)
    worksheet.write(1,1,model.best_sol.cost_of_distance)
    worksheet.write(1,2,model.opt_type)
    worksheet.write(1,3,model.best_sol.obj)
    worksheet.write(2,0,'vehicleID')
    worksheet.write(2,1,'route')
    worksheet.write(2,2,'timetable')
    for row,route in enumerate(model.best_sol.route_list):
        worksheet.write(row+3,0,'v'+str(row+1))
        r=[str(i)for i in route]
        worksheet.write(row+3,1, '-'.join(r))
        r=[str(i)for i in model.best_sol.timetable_list[row]]
        worksheet.write(row+3,2, '-'.join(r))
    work.close()

def plotRoutes(model):
    for route in model.best_sol.route_list:
        x_coord=[model.depot_dict[route[0]].x_coord]
        y_coord=[model.depot_dict[route[0]].y_coord]
        for node_id in route[1:-1]:
            x_coord.append(model.demand_dict[node_id].x_coord)
            y_coord.append(model.demand_dict[node_id].y_coord)
        x_coord.append(model.depot_dict[route[-1]].x_coord)
        y_coord.append(model.depot_dict[route[-1]].y_coord)
        plt.grid()
        if route[0]=='d1':
            plt.plot(x_coord,y_coord,marker='o',color='black',linewidth=0.5,markersize=5)
        elif route[0]=='d2':
            plt.plot(x_coord,y_coord,marker='o',color='orange',linewidth=0.5,markersize=5)
        else:
            plt.plot(x_coord,y_coord,marker='o',color='b',linewidth=0.5,markersize=5)
    plt.xlabel('x_coord')
    plt.ylabel('y_coord')
    plt.show()

def to_csv(best_obj_list,store_result):
    array_format = pd.DataFrame(best_obj_list)
    array_format.to_csv(store_result,header=None)  # don't write header
    



def run(demand_file,depot_file,store_result,epochs,Cr,F,popsize,v_cap,v_speed,opt_type):
    """
    :param demand_file: demand file path
    :param depot_file: depot file path
    :param epochs: Iterations
    :param Cr: Differential crossover probability
    :param F: Scaling factor
    :param popsize: Population size
    :param v_cap: Vehicle capacity
    :param v_speed: Vehicle free speed
    :param opt_type: Optimization type:0:Minimize the number of vehicles,1:Minimize travel distance
    :return:
    """
    model=Model()
    model.vehicle_cap=v_cap
    model.vehicle_speed=v_speed
    model.Cr=Cr
    model.F=F
    model.popsize=popsize
    model.opt_type=opt_type

    readCSVFile(demand_file,depot_file,model)
    calDistanceTimeMatrix(model)
    best_sol = Sol()
    best_sol.obj = float('inf')
    model.best_sol = best_sol
    genInitialSol(model)
    history_best_obj = []
    for ep in range(epochs):
        for i in range(popsize):
            v1=random.randint(0,len(model.demand_id_list)-1)
            sol=model.sol_list[v1]
            mu_x=muSol(model,v1)
            u=crossSol(model,sol.node_id_list,mu_x)
            new_sol=Sol()
            new_sol.node_id_list=u
            calObj(new_sol,model)
            if new_sol.obj<=sol.obj:
                sol=copy.deepcopy(new_sol)
                if sol.obj<model.best_sol.obj:
                    model.best_sol=copy.deepcopy(sol)
            history_best_obj.append(model.best_sol.obj)
        print("%s/%s， best obj: %s" % (ep, epochs, model.best_sol.obj))
    to_csv(history_best_obj,store_result)
    # plotObj(history_best_obj)
    # plotRoutes(model)
    # outPut(model)

if __name__ == '__main__':

    # #pr01
    # demand_file10 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr01C.csv'
    # depot_file10 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr01D.csv'
    # store_result10 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/pr01.csv'
    
    # #pr02
    # demand_file11 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr02C.csv'
    # depot_file11 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr02D.csv'
    # store_result11 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/pr02.csv'
    
    # #pr03
    # demand_file12 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr03C.csv'
    # depot_file12 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr03D.csv'
    # store_result12 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/pr03.csv'
    
    # #pr04
    # demand_file13 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr04C.csv'
    # depot_file13 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr04D.csv'
    # store_result13 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/pr04.csv'
    
    # #pr05
    # demand_file14 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr05C.csv'
    # depot_file14 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr05D.csv'
    # store_result14 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/pr05.csv'
    
    # #pr06
    # demand_file15 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr06C.csv'
    # depot_file15 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr06D.csv'
    # store_result15 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/pr06.csv'
    
    # #pr07
    # demand_file16 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr07C.csv'
    # depot_file16 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr07D.csv'
    # store_result16 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/pr07.csv'
    
    # #pr08
    # demand_file17 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr08C.csv'
    # depot_file17 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr08D.csv'
    # store_result17 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/pr08.csv'
    
    # #pr09
    # demand_file18 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr09C.csv'
    # depot_file18 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr09D.csv'
    # store_result18 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/pr09.csv'
    
    # #pr10
    # demand_file19 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr10C.csv'
    # depot_file19 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr10D.csv'
    # store_result19 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/pr10.csv'
    
    # #pr11
    # demand_file20 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr11C.csv'
    # depot_file20 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr11D.csv'
    # store_result20 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/pr11.csv'
    
    # #pr12
    # demand_file21 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr12C.csv'
    # depot_file21 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr12D.csv'
    # store_result21 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/pr12.csv'
    
    # #pr13
    # demand_file22 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr13C.csv'
    # depot_file22 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr13D.csv'
    # store_result22 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/pr13.csv'
    
    # #pr14
    # demand_file23 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr14C.csv'
    # depot_file23 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr14D.csv'
    # store_result23 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/pr14.csv'
    
    # #pr15
    # demand_file24 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr15C.csv'
    # depot_file24 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr15D.csv'
    # store_result24 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/pr15.csv'
    
    # #pr16
    # demand_file25 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr16C.csv'
    # depot_file25 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr16D.csv'
    # store_result25 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/pr16.csv'
    
    # #pr17
    # demand_file26 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr17C.csv'
    # depot_file26 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr17D.csv'
    # store_result26 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/pr17.csv'
    
    # #pr18
    # demand_file27 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr18C.csv'
    # depot_file27 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr18D.csv'
    # store_result27 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/pr18.csv'
    
    # #pr19
    # demand_file28 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr19C.csv'
    # depot_file28 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr19D.csv'
    # store_result28 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/pr19.csv'
    
    # #pr20
    # demand_file29 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr20C.csv'
    # depot_file29 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr20D.csv'
    # store_result29 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/pr20.csv'
    
    #C1_20_n
    demand_file1 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C1_20_n.csv'
    depot_file1 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C1_5.csv'
    store_result1= '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/MDLRPSDPTW/C1_20_n.csv'
    #C1_20_w
    demand_file2 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C1_20_w.csv'
    depot_file2 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C1_5.csv'
    store_result2 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/MDLRPSDPTW/C1_20_w.csv'
    #C1_50_n
    demand_file3 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C1_50_n.csv'
    depot_file3 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C1_5.csv'
    store_result3 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/MDLRPSDPTW/C1_50_n.csv'
    #C1_50_w
    demand_file4 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C1_50_w.csv'
    depot_file4 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C1_5.csv'
    store_result4 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/MDLRPSDPTW/C1_50_w.csv'
    #C1_100_102_5
    demand_file5 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C1_100_102.csv'
    depot_file5 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C1_5.csv'
    store_result5 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/MDLRPSDPTW/C1_100_102_5.csv'
    #C1_100_102_10
    demand_file6 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C1_100_102.csv'
    depot_file6 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C1_10.csv'
    store_result6 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/MDLRPSDPTW/C1_100_102_10.csv'
    #C1_100_105_5
    demand_file7 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C1_100_105.csv'
    depot_file7 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C1_5.csv'
    store_result7 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/MDLRPSDPTW/C1_100_105_5.csv'
    #C1_100_105_10
    demand_file8 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C1_100_105.csv'
    depot_file8 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C1_10.csv'
    store_result8 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/MDLRPSDPTW/C1_100_105_10.csv'
    #C1_100_108_5
    demand_file9 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C1_100_108.csv'
    depot_file9 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C1_5.csv'
    store_result9 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/MDLRPSDPTW/C1_100_108_5.csv'
    #C1_100_108_10
    demand_file10 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C1_100_108.csv'
    depot_file10 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C1_10.csv'
    store_result10 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/MDLRPSDPTW/C1_100_108_10.csv'
 
    
    #C2_20_n
    demand_file11 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C2_20_n.csv'
    depot_file11 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C2_5.csv'
    store_result11 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/MDLRPSDPTW/C2_20_n.csv'
    #C2_20_w
    demand_file12 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C2_20_w.csv'
    depot_file12 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C2_5.csv'
    store_result12 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/MDLRPSDPTW/C2_20_w.csv'
    #C2_50_n
    demand_file13 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C2_50_n.csv'
    depot_file13 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C2_5.csv'
    store_result13 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/MDLRPSDPTW/C2_50_n.csv'
    #C2_50_w
    demand_file14 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C2_50_w.csv'
    depot_file14 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C2_5.csv'
    store_result14 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/MDLRPSDPTW/C2_50_w.csv'
    #C2_100_202_5
    demand_file15 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C2_100_202.csv'
    depot_file15 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C2_5.csv'
    store_result15 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/MDLRPSDPTW/C2_100_202_5.csv'
    #C2_100_202_10
    demand_file16 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C2_100_202.csv'
    depot_file16 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C2_10.csv'
    store_result16 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/MDLRPSDPTW/C2_100_202_10.csv'
    #C2_100_205_5
    demand_file17 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C2_100_205.csv'
    depot_file17 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C2_5.csv'
    store_result17 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/MDLRPSDPTW/C2_100_205_5.csv'
    #C2_100_205_10
    demand_file18 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C2_100_205.csv'
    depot_file18 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C2_10.csv'
    store_result18 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/MDLRPSDPTW/C2_100_205_10.csv'
    #C2_100_208_5
    demand_file19 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C2_100_208.csv'
    depot_file19 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C2_5.csv'
    store_result19 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/MDLRPSDPTW/C2_100_208_5.csv'
    #C2_100_208_10
    demand_file20 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C2_100_208.csv'
    depot_file20 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C2_10.csv'
    store_result20 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/MDLRPSDPTW/C2_100_208_10.csv'
 
    
    #R1_20_n
    demand_file21 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R1_20_n.csv'
    depot_file21 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R1_5.csv'
    store_result21= '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/MDLRPSDPTW/R1_20_n.csv'
    #R1_20_w
    demand_file22 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R1_20_w.csv'
    depot_file22 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R1_5.csv'
    store_result22 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/MDLRPSDPTW/R1_20_w.csv'
    #R1_50_n
    demand_file23 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R1_50_n.csv'
    depot_file23 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R1_5.csv'
    store_result23 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/MDLRPSDPTW/R1_50_n.csv'
    #R1_50_w
    demand_file24 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R1_50_w.csv'
    depot_file24 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R1_5.csv'
    store_result24 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/MDLRPSDPTW/R1_50_w.csv'
    #R1_100_101_5
    demand_file25 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R1_100_101.csv'
    depot_file25 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R1_5.csv'
    store_result25 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/MDLRPSDPTW/R1_100_101_5.csv'
    #R1_100_101_10
    demand_file26 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R1_100_101.csv'
    depot_file26 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R1_10.csv'
    store_result26 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/MDLRPSDPTW/R1_100_101_10.csv'
    #R1_100_104_5
    demand_file27 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R1_100_104.csv'
    depot_file27 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R1_5.csv'
    store_result27 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/MDLRPSDPTW/R1_100_104_5.csv'
    #R1_100_104_10
    demand_file28 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R1_100_104.csv'
    depot_file28 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R1_10.csv'
    store_result28 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/MDLRPSDPTW/R1_100_104_10.csv'
    #R1_100_107_5
    demand_file29 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R1_100_107.csv'
    depot_file29 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R1_5.csv'
    store_result29 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/MDLRPSDPTW/R1_100_107_5.csv'
    #R1_100_107_10
    demand_file30 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R1_100_107.csv'
    depot_file30 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R1_10.csv'
    store_result30 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/MDLRPSDPTW/R1_100_107_10.csv'
    #R1_100_110_5
    demand_file31 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R1_100_110.csv'
    depot_file31 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R1_5.csv'
    store_result31 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/MDLRPSDPTW/R1_100_110_5.csv'
    #R1_100_110_10
    demand_file32 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R1_100_110.csv'
    depot_file32 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R1_10.csv'
    store_result32 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/MDLRPSDPTW/R1_100_110_10.csv'
 
    #R2_20_n
    demand_file33 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R2_20_n.csv'
    depot_file33 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R2_5.csv'
    store_result33 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/MDLRPSDPTW/R2_20_n.csv'
    #R2_20_w
    demand_file34 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R2_20_w.csv'
    depot_file34 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R2_5.csv'
    store_result34 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/MDLRPSDPTW/R2_20_w.csv'
    #R2_50_n
    demand_file35 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R2_50_n.csv'
    depot_file35 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R2_5.csv'
    store_result35 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/MDLRPSDPTW/R2_50_n.csv'
    #R2_50_w
    demand_file36 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R2_50_w.csv'
    depot_file36 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R2_5.csv'
    store_result36 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/MDLRPSDPTW/R2_50_w.csv'
    #R2_100_201_5
    demand_file37 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R2_100_201.csv'
    depot_file37 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R2_5.csv'
    store_result37 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/MDLRPSDPTW/R2_100_201_5.csv'
    #R2_100_201_10
    demand_file38 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R2_100_201.csv'
    depot_file38 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R2_10.csv'
    store_result38 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/MDLRPSDPTW/R2_100_201_10.csv'
    #R2_100_204_5
    demand_file39 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R2_100_204.csv'
    depot_file39 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R2_5.csv'
    store_result39 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/MDLRPSDPTW/R2_100_204_5.csv'
    #R2_100_204_10
    demand_file40 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R2_100_204.csv'
    depot_file40 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R2_10.csv'
    store_result40 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/MDLRPSDPTW/R2_100_204_10.csv'
    #R2_100_207_5
    demand_file41 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R2_100_207.csv'
    depot_file41 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R2_5.csv'
    store_result41 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/MDLRPSDPTW/R2_100_207_5.csv'
    #R2_100_207_10
    demand_file42 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R2_100_207.csv'
    depot_file42 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R2_10.csv'
    store_result42 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/MDLRPSDPTW/R2_100_207_10.csv'
    #R2_100_210_5
    demand_file43 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R2_100_210.csv'
    depot_file43 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R2_5.csv'
    store_result43 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/MDLRPSDPTW/R2_100_210_5.csv'
    #R2_100_210_10
    demand_file44 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R2_100_210.csv'
    depot_file44 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R2_10.csv'
    store_result44 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/MDLRPSDPTW/R2_100_210_10.csv'
 
    #RC1_20_n
    demand_file45 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC1_20_n.csv'
    depot_file45 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC1_5.csv'
    store_result45= '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/MDLRPSDPTW/RC1_20_n.csv'
    #RC1_20_w
    demand_file46 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC1_20_w.csv'
    depot_file46 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC1_5.csv'
    store_result46 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/MDLRPSDPTW/RC1_20_w.csv'
    #RC1_50_n
    demand_file47 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC1_50_n.csv'
    depot_file47 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC1_5.csv'
    store_result47 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/MDLRPSDPTW/RC1_50_n.csv'
    #RC1_50_w
    demand_file48 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC1_50_w.csv'
    depot_file48 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC1_5.csv'
    store_result48 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/MDLRPSDPTW/RC1_50_w.csv'
    #RC1_100_103_5
    demand_file49 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC1_100_103.csv'
    depot_file49 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC1_5.csv'
    store_result49 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/MDLRPSDPTW/RC1_100_103_5.csv'
    #RC1_100_103_10
    demand_file50 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC1_100_103.csv'
    depot_file50 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC1_10.csv'
    store_result50 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/MDLRPSDPTW/RC1_100_103_10.csv'
    #RC1_100_106_5
    demand_file51 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC1_100_106.csv'
    depot_file51 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC1_5.csv'
    store_result51 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/MDLRPSDPTW/RC1_100_106_5.csv'
    #RC1_100_106_10
    demand_file52 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC1_100_106.csv'
    depot_file52 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC1_10.csv'
    store_result52 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/MDLRPSDPTW/RC1_100_106_10.csv'
    #RC2_20_n
    demand_file53 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC2_20_n.csv'
    depot_file53 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC2_5.csv'
    store_result53 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/MDLRPSDPTW/RC2_20_n.csv'
    #RC2_20_w
    demand_file54 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC2_20_w.csv'
    depot_file54 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC2_5.csv'
    store_result54 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/MDLRPSDPTW/RC2_20_w.csv'
    #RC2_50_n
    demand_file55 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC2_50_n.csv'
    depot_file55 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC2_5.csv'
    store_result55 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/MDLRPSDPTW/RC2_50_n.csv'
    #RC2_50_w
    demand_file56 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC2_50_w.csv'
    depot_file56 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC2_5.csv'
    store_result56 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/MDLRPSDPTW/RC2_50_w.csv'
    #RC2_100_203_5
    demand_file57 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC2_100_203.csv'
    depot_file57 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC2_5.csv'
    store_result57 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/MDLRPSDPTW/RC2_100_203_5.csv'
    #RC2_100_203_10
    demand_file58 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC2_100_203.csv'
    depot_file58 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC2_10.csv'
    store_result58 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/MDLRPSDPTW/RC2_100_203_10.csv'
    #RC2_100_206_5
    demand_file59 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC2_100_206.csv'
    depot_file59 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC2_5.csv'
    store_result59 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/MDLRPSDPTW/RC2_100_206_5.csv'
    #RC2_100_206_10
    demand_file60 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC2_100_206.csv'
    depot_file60 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC2_10.csv'
    store_result60 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/DE/MDLRPSDPTW/RC2_100_206_10.csv'
    
    
    file_tuple1 = (demand_file1,depot_file1,store_result1)
    file_tuple2 = (demand_file2,depot_file2,store_result2)
    file_tuple3 = (demand_file3,depot_file3,store_result3)
    file_tuple4 = (demand_file4,depot_file4,store_result4)
    file_tuple5 = (demand_file5,depot_file5,store_result5)
    file_tuple6 = (demand_file6,depot_file6,store_result6)
    file_tuple7 = (demand_file7,depot_file7,store_result7)
    file_tuple8 = (demand_file8,depot_file8,store_result8)
    file_tuple9 = (demand_file9,depot_file9,store_result9)
    file_tuple10 = (demand_file10,depot_file10,store_result10)
    file_tuple11 = (demand_file11,depot_file11,store_result11)
    file_tuple12 = (demand_file12,depot_file12,store_result12)
    file_tuple13 = (demand_file13,depot_file13,store_result13)
    file_tuple14 = (demand_file14,depot_file14,store_result14)
    file_tuple15 = (demand_file15,depot_file15,store_result15)
    file_tuple16 = (demand_file16,depot_file16,store_result16)
    file_tuple17 = (demand_file17,depot_file17,store_result17)
    file_tuple18 = (demand_file18,depot_file18,store_result18)
    file_tuple19 = (demand_file19,depot_file19,store_result19)
    file_tuple20 = (demand_file20,depot_file20,store_result20)
    file_tuple21 = (demand_file21,depot_file21,store_result21)
    file_tuple22 = (demand_file22,depot_file22,store_result22)
    file_tuple23 = (demand_file23,depot_file23,store_result23)
    file_tuple24 = (demand_file24,depot_file24,store_result24)
    file_tuple25 = (demand_file25,depot_file25,store_result25)
    file_tuple26 = (demand_file26,depot_file26,store_result26)
    file_tuple27 = (demand_file27,depot_file27,store_result27)
    file_tuple28 = (demand_file28,depot_file28,store_result28)
    file_tuple29 = (demand_file29,depot_file29,store_result29)
    file_tuple30 = (demand_file30,depot_file30,store_result30)
    file_tuple31 = (demand_file31,depot_file31,store_result31)
    file_tuple32 = (demand_file32,depot_file32,store_result32)
    file_tuple33 = (demand_file33,depot_file33,store_result33)
    file_tuple34 = (demand_file34,depot_file34,store_result34)
    file_tuple35 = (demand_file35,depot_file35,store_result35)
    file_tuple36 = (demand_file36,depot_file36,store_result36)
    file_tuple37 = (demand_file37,depot_file37,store_result37)
    file_tuple38 = (demand_file38,depot_file38,store_result38)
    file_tuple39 = (demand_file39,depot_file39,store_result39)
    file_tuple40 = (demand_file40,depot_file40,store_result40)
    file_tuple41 = (demand_file41,depot_file41,store_result41)
    file_tuple42 = (demand_file42,depot_file42,store_result42)
    file_tuple43 = (demand_file43,depot_file43,store_result43)
    file_tuple44 = (demand_file44,depot_file44,store_result44)
    file_tuple45 = (demand_file45,depot_file45,store_result45)
    file_tuple46 = (demand_file46,depot_file46,store_result46)
    file_tuple47 = (demand_file47,depot_file47,store_result47)
    file_tuple48 = (demand_file48,depot_file48,store_result48)
    file_tuple49 = (demand_file49,depot_file49,store_result49)
    file_tuple50 = (demand_file50,depot_file50,store_result50)
    file_tuple51 = (demand_file51,depot_file51,store_result51)
    file_tuple52 = (demand_file52,depot_file52,store_result52)
    file_tuple53 = (demand_file53,depot_file53,store_result53)
    file_tuple54 = (demand_file54,depot_file54,store_result54)
    file_tuple55 = (demand_file55,depot_file55,store_result55)
    file_tuple56 = (demand_file56,depot_file56,store_result56)
    file_tuple57 = (demand_file57,depot_file57,store_result57)
    file_tuple58 = (demand_file58,depot_file58,store_result58)
    file_tuple59 = (demand_file59,depot_file59,store_result59)
    file_tuple60 = (demand_file60,depot_file60,store_result60)
    
    file_list = [file_tuple1,file_tuple2,file_tuple3,file_tuple4,file_tuple5,file_tuple6,file_tuple7,file_tuple8,file_tuple9,file_tuple10,file_tuple11,file_tuple12,file_tuple13,file_tuple14,file_tuple15,file_tuple16,file_tuple17,file_tuple18,file_tuple19,file_tuple20,file_tuple21,file_tuple22,file_tuple23,file_tuple24,file_tuple25,file_tuple26,file_tuple27,file_tuple28,file_tuple29,file_tuple30,file_tuple31,file_tuple32,file_tuple33,file_tuple34,file_tuple35,file_tuple36,file_tuple37,file_tuple38,file_tuple39,file_tuple40,file_tuple41,file_tuple42,file_tuple43,file_tuple44,file_tuple45,file_tuple46,file_tuple47,file_tuple48,file_tuple49,file_tuple50,file_tuple51,file_tuple52,file_tuple53,file_tuple54,file_tuple55,file_tuple56,file_tuple57,file_tuple58,file_tuple59,file_tuple60]
     
    for demand_file,depot_file,store_result in  file_list:
        run(demand_file,depot_file, store_result,epochs=150, Cr=0.6,F=0.5, popsize=400, v_cap=80,v_speed=1, opt_type=0)
