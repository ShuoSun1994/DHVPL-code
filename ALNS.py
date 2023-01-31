# -*- coding: utf-8 -*-
# @Time    : 2021/11/3 14:46
# @Author  : Praise
# @File    : ALNS_MDVRPTW.py
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
import time


def createRandomDestory(model):
    d=random.uniform(model.rand_d_min,model.rand_d_max)
    reomve_list=random.sample(range(len(model.demand_id_list)),int(d*len(model.demand_id_list)))
    return reomve_list

def createWorseDestory(model,sol):
    deta_f=[]
    for node_id in sol.node_id_list:
        sol_=copy.deepcopy(sol)
        sol_.node_id_list.remove(node_id)
        calObj(sol_,model)
        deta_f.append(sol.obj-sol_.obj)
    sorted_id = sorted(range(len(deta_f)), key=lambda k: deta_f[k], reverse=True)
    d=random.randint(model.worst_d_min,model.worst_d_max)
    remove_list=sorted_id[:d]
    return remove_list

def createRandomRepair(remove_list,model,sol):
    unassigned_nodes_id=[]
    assigned_nodes_id = []
    # remove node from current solution
    for i in range(len(model.demand_id_list)):
        if i in remove_list:
            unassigned_nodes_id.append(sol.node_id_list[i])
        else:
            assigned_nodes_id.append(sol.node_id_list[i])
    # insert
    for node_id in unassigned_nodes_id:
        index=random.randint(0,len(assigned_nodes_id)-1)
        assigned_nodes_id.insert(index,node_id)
    new_sol=Sol()
    new_sol.node_id_list=copy.deepcopy(assigned_nodes_id)
    calObj(new_sol,model)
    return new_sol

def findGreedyInsert(unassigned_nodes_id,assigned_nodes_id,model):
    best_insert_node_id=None
    best_insert_index = None
    best_insert_cost = float('inf')
    sol_1=Sol()
    sol_1.node_id_list=assigned_nodes_id
    calObj(sol_1,model)
    for node_id in unassigned_nodes_id:
        for i in range(len(assigned_nodes_id)):
            sol_2=Sol()
            sol_2.node_id_list=copy.deepcopy(assigned_nodes_id)
            sol_2.node_id_list.insert(i, node_id)
            calObj(sol_2, model)
            deta_f = sol_2.obj -sol_1.obj
            if deta_f<best_insert_cost:
                best_insert_index=i
                best_insert_node_id=node_id
                best_insert_cost=deta_f
    return best_insert_node_id,best_insert_index

def createGreedyRepair(remove_list,model,sol):
    unassigned_nodes_id = []
    assigned_nodes_id = []
    # remove node from current solution
    for i in range(len(model.demand_id_list)):
        if i in remove_list:
            unassigned_nodes_id.append(sol.node_id_list[i])
        else:
            assigned_nodes_id.append(sol.node_id_list[i])
    #insert
    while len(unassigned_nodes_id)>0:
        insert_node_id,insert_index=findGreedyInsert(unassigned_nodes_id,assigned_nodes_id,model)
        assigned_nodes_id.insert(insert_index,insert_node_id)
        unassigned_nodes_id.remove(insert_node_id)
    new_sol=Sol()
    new_sol.node_id_list=copy.deepcopy(assigned_nodes_id)
    calObj(new_sol,model)
    return new_sol

def findRegretInsert(unassigned_nodes_id,assigned_nodes_id,model,regret_n):
    opt_insert_node_id = None
    opt_insert_index = None
    opt_insert_cost = -float('inf')
    sol_=Sol()
    for node_id in unassigned_nodes_id:
        n_insert_cost=np.zeros((len(assigned_nodes_id),3))
        for i in range(len(assigned_nodes_id)):
            sol_.node_id_list=copy.deepcopy(assigned_nodes_id)
            sol_.node_id_list.insert(i,node_id)
            calObj(sol_,model)
            n_insert_cost[i,0]=node_id
            n_insert_cost[i,1]=i
            n_insert_cost[i,2]=sol_.obj
        n_insert_cost= n_insert_cost[n_insert_cost[:, 2].argsort()]
        deta_f=0
        if model.regret_n > len(assigned_nodes_id):
            model.regret_n = random.randint(1,len(assigned_nodes_id))
            
        for i in range(1,model.regret_n):
            deta_f=deta_f+n_insert_cost[i,2]-n_insert_cost[0,2]
        model.regret_n = regret_n
        if deta_f>opt_insert_cost:
            opt_insert_node_id = int(n_insert_cost[0, 0])
            opt_insert_index=int(n_insert_cost[0,1])
            opt_insert_cost=deta_f
    return opt_insert_node_id,opt_insert_index

def createRegretRepair(remove_list,model,sol,regret_n):
    unassigned_nodes_id = []
    assigned_nodes_id = []
    # remove node from current solution
    for i in range(len(model.demand_id_list)):
        if i in remove_list:
            unassigned_nodes_id.append(sol.node_id_list[i])
        else:
            assigned_nodes_id.append(sol.node_id_list[i])
    # insert
    while len(unassigned_nodes_id)>0:
        insert_node_id,insert_index=findRegretInsert(unassigned_nodes_id,assigned_nodes_id,model,regret_n)
        assigned_nodes_id.insert(insert_index,insert_node_id)
        unassigned_nodes_id.remove(insert_node_id)
    new_sol = Sol()
    new_sol.node_id_list = copy.deepcopy(assigned_nodes_id)
    calObj(new_sol, model)
    return new_sol

def selectDestoryRepair(model):
    d_weight=model.d_weight
    d_cumsumprob = (d_weight / sum(d_weight)).cumsum()
    d_cumsumprob -= np.random.rand()
    destory_id= list(d_cumsumprob > 0).index(True)

    r_weight=model.r_weight
    r_cumsumprob = (r_weight / sum(r_weight)).cumsum()
    r_cumsumprob -= np.random.rand()
    repair_id = list(r_cumsumprob > 0).index(True)
    return destory_id,repair_id

def doDestory(destory_id,model,sol):
    if destory_id==0:
        reomve_list=createRandomDestory(model)
    else:
        reomve_list=createWorseDestory(model,sol)
    return reomve_list

def doRepair(repair_id,reomve_list,model,sol,regret_n):
    if repair_id==0:
        new_sol=createRandomRepair(reomve_list,model,sol)
    elif repair_id==1:
        new_sol=createGreedyRepair(reomve_list,model,sol)
    else:
        new_sol=createRegretRepair(reomve_list,model,sol,regret_n)
    return new_sol

def resetScore(model):

    model.d_select = np.zeros(2)
    model.d_score = np.zeros(2)

    model.r_select = np.zeros(3)
    model.r_score = np.zeros(3)

def updateWeight(model):

    for i in range(model.d_weight.shape[0]):
        if model.d_select[i]>0:
            model.d_weight[i]=model.d_weight[i]*(1-model.rho)+model.rho*model.d_score[i]/model.d_select[i]
        else:
            model.d_weight[i] = model.d_weight[i] * (1 - model.rho)
    for i in range(model.r_weight.shape[0]):
        if model.r_select[i]>0:
            model.r_weight[i]=model.r_weight[i]*(1-model.rho)+model.rho*model.r_score[i]/model.r_select[i]
        else:
            model.r_weight[i] = model.r_weight[i] * (1 - model.rho)
    model.d_history_select = model.d_history_select + model.d_select
    model.d_history_score = model.d_history_score + model.d_score
    model.r_history_select = model.r_history_select + model.r_select
    model.r_history_score = model.r_history_score + model.r_score

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

def run(demand_file,depot_file,store_result,rand_d_max,rand_d_min,worst_d_min,worst_d_max,regret_n,r1,r2,r3,rho,phi,epochs,pu,v_cap,
        v_speed,opt_type):
    """
    :param demand_file: demand file path
    :param depot_file: depot file path
    :param rand_d_max: max degree of random destruction
    :param rand_d_min: min degree of random destruction
    :param worst_d_max: max degree of worst destruction
    :param worst_d_min: min degree of worst destruction
    :param regret_n:  n next cheapest insertions
    :param r1: score if the new solution is the best one found so far.
    :param r2: score if the new solution improves the current solution.
    :param r3: score if the new solution does not improve the current solution, but is accepted.
    :param rho: reaction factor of action weight
    :param phi: the reduction factor of threshold
    :param epochs: Iterations
    :param pu: the frequency of weight adjustment
    :param v_cap: Vehicle capacity
    :param v_speed Vehicle free speed
    :param opt_type: Optimization type:0:Minimize the number of vehicles,1:Minimize travel distance
    :return:
    """
    model=Model()
    model.rand_d_max=rand_d_max
    model.rand_d_min=rand_d_min
    model.worst_d_min=worst_d_min
    model.worst_d_max=worst_d_max
    model.regret_n=regret_n
    model.r1=r1
    model.r2=r2
    model.r3=r3
    model.rho=rho
    model.vehicle_cap=v_cap
    model.opt_type=opt_type
    model.vehicle_speed=v_speed
    readCSVFile(demand_file,depot_file, model)
    calDistanceTimeMatrix(model)
    history_best_obj = []
    sol = Sol()
    sol.node_id_list = genInitialSol(model.demand_id_list)
    calObj(sol, model)
    model.best_sol = copy.deepcopy(sol)
    history_best_obj.append(sol.obj)
    for ep in range(epochs):
        T=sol.obj*0.2
        resetScore(model)
        for k in range(pu):
            destory_id,repair_id=selectDestoryRepair(model)
            model.d_select[destory_id]+=1
            model.r_select[repair_id]+=1
            reomve_list=doDestory(destory_id,model,sol)
            new_sol=doRepair(repair_id,reomve_list,model,sol,regret_n)
            if new_sol.obj<sol.obj:
                sol=copy.deepcopy(new_sol)
                if new_sol.obj<model.best_sol.obj:
                    model.best_sol=copy.deepcopy(new_sol)
                    model.d_score[destory_id]+=model.r1
                    model.r_score[repair_id]+=model.r1
                else:
                    model.d_score[destory_id]+=model.r2
                    model.r_score[repair_id]+=model.r2
            elif new_sol.obj-sol.obj<T:
                sol=copy.deepcopy(new_sol)
                model.d_score[destory_id] += model.r3
                model.r_score[repair_id] += model.r3
            T=T*phi
            print("%s/%s:%s/%s， best obj: %s" % (ep,epochs,k,pu, model.best_sol.obj))
            history_best_obj.append(model.best_sol.obj)
        updateWeight(model)
    to_csv(history_best_obj,store_result)
    # plotObj(history_best_obj)
    # plotRoutes(model)
    # outPut(model)
    print("random destory weight is {:.3f}\tselect is {}\tscore is {:.3f}".format(model.d_weight[0],
                                                                        model.d_history_select[0],
                                                                        model.d_history_score[0]))
    print("worse destory weight is {:.3f}\tselect is {}\tscore is {:.3f} ".format(model.d_weight[1],
                                                                        model.d_history_select[1],
                                                                        model.d_history_score[1]))
    print("random repair weight is {:.3f}\tselect is {}\tscore is {:.3f}".format(model.r_weight[0],
                                                                       model.r_history_select[0],
                                                                       model.r_history_score[0]))
    print("greedy repair weight is {:.3f}\tselect is {}\tscore is {:.3f}".format(model.r_weight[1],
                                                                       model.r_history_select[1],
                                                                       model.r_history_score[1]))
    print("regret repair weight is {:.3f}\tselect is {}\tscore is {:.3f}".format(model.r_weight[2],
                                                                       model.r_history_select[2],
                                                                       model.r_history_score[2]))

if __name__=='__main__':

    # #pr01
    # demand_file10 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr01C.csv'
    # depot_file10 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr01D.csv'
    # store_result10 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/pr01.csv'
    
    # #pr02
    # demand_file11 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr02C.csv'
    # depot_file11 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr02D.csv'
    # store_result11 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/pr02.csv'
    
    # #pr03
    # demand_file12 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr03C.csv'
    # depot_file12 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr03D.csv'
    # store_result12 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/pr03.csv'
    
    # #pr04
    # demand_file13 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr04C.csv'
    # depot_file13 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr04D.csv'
    # store_result13 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/pr04.csv'
    
    # #pr05
    # demand_file14 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr05C.csv'
    # depot_file14 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr05D.csv'
    # store_result14 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/pr05.csv'
    
    # #pr06
    # demand_file15 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr06C.csv'
    # depot_file15 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr06D.csv'
    # store_result15 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/pr06.csv'
    
    # #pr07
    # demand_file16 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr07C.csv'
    # depot_file16 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr07D.csv'
    # store_result16 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/pr07.csv'
    
    # #pr08
    # demand_file17 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr08C.csv'
    # depot_file17 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr08D.csv'
    # store_result17 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/pr08.csv'
    
    # #pr09
    # demand_file18 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr09C.csv'
    # depot_file18 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr09D.csv'
    # store_result18 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/pr09.csv'
    
    # #pr10
    # demand_file19 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr10C.csv'
    # depot_file19 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr10D.csv'
    # store_result19 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/pr10.csv'
    
    # #pr11
    # demand_file20 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr11C.csv'
    # depot_file20 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr11D.csv'
    # store_result20 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/pr11.csv'
    
    # #pr12
    # demand_file21 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr12C.csv'
    # depot_file21 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr12D.csv'
    # store_result21 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/pr12.csv'
    
    # #pr13
    # demand_file22 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr13C.csv'
    # depot_file22 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr13D.csv'
    # store_result22 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/pr13.csv'
    
    # #pr14
    # demand_file23 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr14C.csv'
    # depot_file23 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr14D.csv'
    # store_result23 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/pr14.csv'
    
    # #pr15
    # demand_file24 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr15C.csv'
    # depot_file24 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr15D.csv'
    # store_result24 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/pr15.csv'
    
    # #pr16
    # demand_file25 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr16C.csv'
    # depot_file25 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr16D.csv'
    # store_result25 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/pr16.csv'
    
    # #pr17
    # demand_file26 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr17C.csv'
    # depot_file26 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr17D.csv'
    # store_result26 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/pr17.csv'
    
    # #pr18
    # demand_file27 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr18C.csv'
    # depot_file27 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr18D.csv'
    # store_result27 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/pr18.csv'
    
    # #pr19
    # demand_file28 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr19C.csv'
    # depot_file28 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr19D.csv'
    # store_result28 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/pr19.csv'
    
    # #pr20
    # demand_file29 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr20C.csv'
    # depot_file29 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPPDTW/pr20D.csv'
    # store_result29 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/pr20.csv'
    
    #C1_20_n
    demand_file1 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C1_20_n.csv'
    depot_file1 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C1_5.csv'
    store_result1= '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/MDLRPSDPTW/C1_20_n.csv'
    #C1_20_w
    demand_file2 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C1_20_w.csv'
    depot_file2 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C1_5.csv'
    store_result2 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/MDLRPSDPTW/C1_20_w.csv'
    #C1_50_n
    demand_file3 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C1_50_n.csv'
    depot_file3 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C1_5.csv'
    store_result3 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/MDLRPSDPTW/C1_50_n.csv'
    #C1_50_w
    demand_file4 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C1_50_w.csv'
    depot_file4 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C1_5.csv'
    store_result4 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/MDLRPSDPTW/C1_50_w.csv'
    #C1_100_102_5
    demand_file5 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C1_100_102.csv'
    depot_file5 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C1_5.csv'
    store_result5 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/MDLRPSDPTW/C1_100_102_5.csv'
    #C1_100_102_10
    demand_file6 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C1_100_102.csv'
    depot_file6 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C1_10.csv'
    store_result6 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/MDLRPSDPTW/C1_100_102_10.csv'
    #C1_100_105_5
    demand_file7 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C1_100_105.csv'
    depot_file7 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C1_5.csv'
    store_result7 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/MDLRPSDPTW/C1_100_105_5.csv'
    #C1_100_105_10
    demand_file8 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C1_100_105.csv'
    depot_file8 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C1_10.csv'
    store_result8 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/MDLRPSDPTW/C1_100_105_10.csv'
    #C1_100_108_5
    demand_file9 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C1_100_108.csv'
    depot_file9 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C1_5.csv'
    store_result9 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/MDLRPSDPTW/C1_100_108_5.csv'
    #C1_100_108_10
    demand_file10 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C1_100_108.csv'
    depot_file10 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C1_10.csv'
    store_result10 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/MDLRPSDPTW/C1_100_108_10.csv'
 
    
    #C2_20_n
    demand_file11 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C2_20_n.csv'
    depot_file11 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C2_5.csv'
    store_result11 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/MDLRPSDPTW/C2_20_n.csv'
    #C2_20_w
    demand_file12 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C2_20_w.csv'
    depot_file12 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C2_5.csv'
    store_result12 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/MDLRPSDPTW/C2_20_w.csv'
    #C2_50_n
    demand_file13 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C2_50_n.csv'
    depot_file13 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C2_5.csv'
    store_result13 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/MDLRPSDPTW/C2_50_n.csv'
    #C2_50_w
    demand_file14 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C2_50_w.csv'
    depot_file14 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C2_5.csv'
    store_result14 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/MDLRPSDPTW/C2_50_w.csv'
    #C2_100_202_5
    demand_file15 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C2_100_202.csv'
    depot_file15 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C2_5.csv'
    store_result15 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/MDLRPSDPTW/C2_100_202_5.csv'
    #C2_100_202_10
    demand_file16 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C2_100_202.csv'
    depot_file16 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C2_10.csv'
    store_result16 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/MDLRPSDPTW/C2_100_202_10.csv'
    #C2_100_205_5
    demand_file17 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C2_100_205.csv'
    depot_file17 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C2_5.csv'
    store_result17 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/MDLRPSDPTW/C2_100_205_5.csv'
    #C2_100_205_10
    demand_file18 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C2_100_205.csv'
    depot_file18 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C2_10.csv'
    store_result18 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/MDLRPSDPTW/C2_100_205_10.csv'
    #C2_100_208_5
    demand_file19 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C2_100_208.csv'
    depot_file19 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C2_5.csv'
    store_result19 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/MDLRPSDPTW/C2_100_208_5.csv'
    #C2_100_208_10
    demand_file20 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C2_100_208.csv'
    depot_file20 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/C2_10.csv'
    store_result20 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/MDLRPSDPTW/C2_100_208_10.csv'
 
    
    #R1_20_n
    demand_file21 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R1_20_n.csv'
    depot_file21 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R1_5.csv'
    store_result21= '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/MDLRPSDPTW/R1_20_n.csv'
    #R1_20_w
    demand_file22 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R1_20_w.csv'
    depot_file22 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R1_5.csv'
    store_result22 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/MDLRPSDPTW/R1_20_w.csv'
    #R1_50_n
    demand_file23 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R1_50_n.csv'
    depot_file23 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R1_5.csv'
    store_result23 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/MDLRPSDPTW/R1_50_n.csv'
    #R1_50_w
    demand_file24 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R1_50_w.csv'
    depot_file24 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R1_5.csv'
    store_result24 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/MDLRPSDPTW/R1_50_w.csv'
    #R1_100_101_5
    demand_file25 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R1_100_101.csv'
    depot_file25 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R1_5.csv'
    store_result25 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/MDLRPSDPTW/R1_100_101_5.csv'
    #R1_100_101_10
    demand_file26 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R1_100_101.csv'
    depot_file26 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R1_10.csv'
    store_result26 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/MDLRPSDPTW/R1_100_101_10.csv'
    #R1_100_104_5
    demand_file27 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R1_100_104.csv'
    depot_file27 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R1_5.csv'
    store_result27 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/MDLRPSDPTW/R1_100_104_5.csv'
    #R1_100_104_10
    demand_file28 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R1_100_104.csv'
    depot_file28 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R1_10.csv'
    store_result28 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/MDLRPSDPTW/R1_100_104_10.csv'
    #R1_100_107_5
    demand_file29 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R1_100_107.csv'
    depot_file29 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R1_5.csv'
    store_result29 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/MDLRPSDPTW/R1_100_107_5.csv'
    #R1_100_107_10
    demand_file30 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R1_100_107.csv'
    depot_file30 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R1_10.csv'
    store_result30 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/MDLRPSDPTW/R1_100_107_10.csv'
    #R1_100_110_5
    demand_file31 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R1_100_110.csv'
    depot_file31 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R1_5.csv'
    store_result31 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/MDLRPSDPTW/R1_100_110_5.csv'
    #R1_100_110_10
    demand_file32 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R1_100_110.csv'
    depot_file32 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R1_10.csv'
    store_result32 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/MDLRPSDPTW/R1_100_110_10.csv'
 
    #R2_20_n
    demand_file33 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R2_20_n.csv'
    depot_file33 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R2_5.csv'
    store_result33 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/MDLRPSDPTW/R2_20_n.csv'
    #R2_20_w
    demand_file34 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R2_20_w.csv'
    depot_file34 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R2_5.csv'
    store_result34 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/MDLRPSDPTW/R2_20_w.csv'
    #R2_50_n
    demand_file35 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R2_50_n.csv'
    depot_file35 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R2_5.csv'
    store_result35 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/MDLRPSDPTW/R2_50_n.csv'
    #R2_50_w
    demand_file36 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R2_50_w.csv'
    depot_file36 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R2_5.csv'
    store_result36 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/MDLRPSDPTW/R2_50_w.csv'
    #R2_100_201_5
    demand_file37 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R2_100_201.csv'
    depot_file37 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R2_5.csv'
    store_result37 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/MDLRPSDPTW/R2_100_201_5.csv'
    #R2_100_201_10
    demand_file38 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R2_100_201.csv'
    depot_file38 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R2_10.csv'
    store_result38 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/MDLRPSDPTW/R2_100_201_10.csv'
    #R2_100_204_5
    demand_file39 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R2_100_204.csv'
    depot_file39 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R2_5.csv'
    store_result39 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/MDLRPSDPTW/R2_100_204_5.csv'
    #R2_100_204_10
    demand_file40 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R2_100_204.csv'
    depot_file40 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R2_10.csv'
    store_result40 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/MDLRPSDPTW/R2_100_204_10.csv'
    #R2_100_207_5
    demand_file41 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R2_100_207.csv'
    depot_file41 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R2_5.csv'
    store_result41 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/MDLRPSDPTW/R2_100_207_5.csv'
    #R2_100_207_10
    demand_file42 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R2_100_207.csv'
    depot_file42 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R2_10.csv'
    store_result42 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/MDLRPSDPTW/R2_100_207_10.csv'
    #R2_100_210_5
    demand_file43 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R2_100_210.csv'
    depot_file43 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R2_5.csv'
    store_result43 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/MDLRPSDPTW/R2_100_210_5.csv'
    #R2_100_210_10
    demand_file44 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R2_100_210.csv'
    depot_file44 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/R2_10.csv'
    store_result44 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/MDLRPSDPTW/R2_100_210_10.csv'
 
    #RC1_20_n
    demand_file45 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC1_20_n.csv'
    depot_file45 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC1_5.csv'
    store_result45= '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/MDLRPSDPTW/RC1_20_n.csv'
    #RC1_20_w
    demand_file46 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC1_20_w.csv'
    depot_file46 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC1_5.csv'
    store_result46 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/MDLRPSDPTW/RC1_20_w.csv'
    #RC1_50_n
    demand_file47 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC1_50_n.csv'
    depot_file47 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC1_5.csv'
    store_result47 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/MDLRPSDPTW/RC1_50_n.csv'
    #RC1_50_w
    demand_file48 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC1_50_w.csv'
    depot_file48 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC1_5.csv'
    store_result48 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/MDLRPSDPTW/RC1_50_w.csv'
    #RC1_100_103_5
    demand_file49 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC1_100_103.csv'
    depot_file49 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC1_5.csv'
    store_result49 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/MDLRPSDPTW/RC1_100_103_5.csv'
    #RC1_100_103_10
    demand_file50 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC1_100_103.csv'
    depot_file50 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC1_10.csv'
    store_result50 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/MDLRPSDPTW/RC1_100_103_10.csv'
    #RC1_100_106_5
    demand_file51 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC1_100_106.csv'
    depot_file51 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC1_5.csv'
    store_result51 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/MDLRPSDPTW/RC1_100_106_5.csv'
    #RC1_100_106_10
    demand_file52 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC1_100_106.csv'
    depot_file52 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC1_10.csv'
    store_result52 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/MDLRPSDPTW/RC1_100_106_10.csv'
    #RC2_20_n
    demand_file53 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC2_20_n.csv'
    depot_file53 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC2_5.csv'
    store_result53 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/MDLRPSDPTW/RC2_20_n.csv'
    #RC2_20_w
    demand_file54 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC2_20_w.csv'
    depot_file54 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC2_5.csv'
    store_result54 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/MDLRPSDPTW/RC2_20_w.csv'
    #RC2_50_n
    demand_file55 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC2_50_n.csv'
    depot_file55 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC2_5.csv'
    store_result55 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/MDLRPSDPTW/RC2_50_n.csv'
    #RC2_50_w
    demand_file56 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC2_50_w.csv'
    depot_file56 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC2_5.csv'
    store_result56 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/MDLRPSDPTW/RC2_50_w.csv'
    #RC2_100_203_5
    demand_file57 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC2_100_203.csv'
    depot_file57 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC2_5.csv'
    store_result57 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/MDLRPSDPTW/RC2_100_203_5.csv'
    #RC2_100_203_10
    demand_file58 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC2_100_203.csv'
    depot_file58 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC2_10.csv'
    store_result58 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/MDLRPSDPTW/RC2_100_203_10.csv'
    #RC2_100_206_5
    demand_file59 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC2_100_206.csv'
    depot_file59 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC2_5.csv'
    store_result59 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/MDLRPSDPTW/RC2_100_206_5.csv'
    #RC2_100_206_10
    demand_file60 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC2_100_206.csv'
    depot_file60 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDLRPSDPTW/RC2_10.csv'
    store_result60 = '/home/ss/Desktop/科研/code/code/paper_code/VPL/result/ALNS/MDLRPSDPTW/RC2_100_206_10.csv'
    
    
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
        start_time = time.time()
        run(demand_file=demand_file,depot_file=depot_file,store_result = store_result,rand_d_max=0.4,rand_d_min=0.1,
        worst_d_min=5,worst_d_max=20,regret_n=5,r1=30,r2=20,r3=10,rho=0.4,
        phi=0.9,epochs=20,pu=5,v_cap=80,v_speed=1,opt_type=0)
        end_time = time.time() - start_time
        print("Game Over , took {} s".format(time.strftime('%H:%M:%S', time.gmtime(end_time))))    
