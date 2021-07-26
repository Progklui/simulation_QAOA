#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 16:38:39 2021

@author: admin-klui
"""
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx # module useful for graph-visualization 

# QAA analysis

# visualize the instantaneous eigen-values
def plot_adiabatic_instantaneous_eigenstates(t, t_evals, image_name):
    font_size = 15 # label font size
    fig = plt.figure(figsize=(15,7), constrained_layout=True)

    for i in range(len(t_evals)):
        if i == 0:
            plt.plot(t, t_evals[i], label=r"$\varepsilon_{min} ($s = "+str(t[len(t)-1])+"$) = $"+str(round(t_evals[i][len(t)-1],2)))
        elif i == len(t_evals)-1:
            plt.plot(t, t_evals[i], label=r"$\varepsilon_{max} ($s = "+str(t[len(t)-1])+"$) = $"+str(round(t_evals[i][len(t)-1],2)))
        else:
            plt.plot(t, t_evals[i])
        
    plt.xlabel(r"$s$", fontsize=font_size)
    plt.ylabel(r"$\varepsilon (s) $", fontsize=font_size)

    plt.legend(fontsize=font_size,loc="lower right")
    plt.savefig("images/"+image_name+".png", dpi=200)

# QAOA analysis

# function to visualize the energy surface of the gamma,beta state with respect to H_p (just makes sense for p=1) / scanning results
def plot_qaoa_scan_p1(gamma, beta, exp_values, image_name):
    font_size = 15 # label font size
    fig, axs = plt.subplots(1,1, constrained_layout=True, figsize=(14,7), sharey=True) # create plot-layout

    cs = axs.contourf(gamma, beta, exp_values) # create contour-map
    cbar = fig.colorbar(cs, ax=axs) # get color-bar
    cbar.ax.set_ylabel(r'$\langle H_p \rangle$', fontsize=font_size)
    axs.locator_params(nbins=4)

    axs.set_xlabel(r"$\gamma$", fontsize=font_size)
    axs.set_ylabel(r"$\beta$", fontsize=font_size)

    plt.savefig("images/"+image_name+".png", dpi=200)

# function to facilitate plotting of the mean values
def plot_energies_entropies(axes, i, j, M, L, value_mean, value_std, p, quantity, error_bars, loc_legend, y_label, alpha_trans):
    if j != 10:
        if error_bars == "error":
            for n in range(len(p)):
                axes[i,j].plot(np.arange(M+1), value_mean[n], label=r"$p = $"+str(p[n])+", $L = $"+str(L)+", $M = $"+str(M)+quantity+str(round(value_mean[n,M-1],2)))
                axes[i,j].fill_between(np.arange(M+1), value_mean[n]-value_std[n], value_mean[n]+value_std[n], alpha=alpha_trans[n])
        elif error_bars == "no_error":
            for n in range(len(p)):
                axes[i,j].plot(np.arange(M+1), value_mean[n], label=r"$p = $"+str(p[n])+", $L = $"+str(L)+", $M = $"+str(M)+quantity+str(round(value_mean[n,M-1],2)))
        axes[i,j].legend(loc=loc_legend)
        axes[i,j].set_ylabel(y_label)
    elif j == 10:
        if error_bars == "error":
            for n in range(len(p)):
                axes[i].plot(np.arange(M+1), value_mean[n], label=r"$p = $"+str(p[n])+", $L = $"+str(L)+", $M = $"+str(M)+quantity+str(round(value_mean[n,M-1],2)))
                axes[i].fill_between(np.arange(M+1), value_mean[n]-value_std[n], value_mean[n]+value_std[n], alpha=alpha_trans[n])
        elif error_bars == "no_error":
            for n in range(len(p)):
                axes[i].plot(np.arange(M+1), value_mean[n], label=r"$p = $"+str(p[n])+", $L = $"+str(L)+", $M = $"+str(M)+quantity+str(round(value_mean[n,M-1],2)))
        axes[i].legend(loc=loc_legend)
        if i == 0:
            axes[i].set_ylabel(y_label)
    return axes

# this function comprises the main analysis of the monte-carlo simulation
def plot_monte_carlo_simulation_results(exp_values_1, exp_values_1_std, exp_values_2, exp_values_2_std, entropy_1, entropy_1_std, entropy_2, entropy_2_std, fidelity_1, fidelity_1_std, fidelity_2, fidelity_2_std, error_bars, p_array, M, L, y_labels, y_indices, opt_approach, image_name):
    fig, axs = plt.subplots(2,3, constrained_layout=True, figsize=(15,7), sharex=True)

    if opt_approach == "max": # plot either maximization or minimization dependend on the type of optimization problem
        h_p_string = r", $\overline{\langle H_p \rangle_{max}} = $"
    elif opt_approach == "min":
        h_p_string = r", $\overline{\langle H_p \rangle_{min}} = $"

    alpha_trans = np.array([0.5,0.4,0.3])
    # plot energy values if no additional entropy minimization has been performed
    axs = plot_energies_entropies(axs, 0, 0, M, L, exp_values_1, exp_values_1_std, p_array, 
                                  h_p_string, error_bars, "lower right", y_labels[0], alpha_trans)
    # plot energy values if additional entropy minimization has been performed
    axs = plot_energies_entropies(axs, 1, 0, M, L, exp_values_2, exp_values_2_std, p_array, 
                                  h_p_string, error_bars, "lower right", y_labels[0], alpha_trans)

    # plot entropy values if no additional entropy minimization has been performed
    axs = plot_energies_entropies(axs, 0, 1, M, L, entropy_1, entropy_1_std, p_array, 
                                  y_indices[0], error_bars, "upper right", y_labels[1], alpha_trans)
    # plot entropy values if additional entropy minimization has been performed
    axs = plot_energies_entropies(axs, 1, 1, M, L, entropy_2, entropy_2_std, p_array, 
                                  y_indices[1], error_bars, "upper right", y_labels[1], alpha_trans)

    # plot fidelity during iterations if no additional entropy minimzation has been performed
    axs = plot_energies_entropies(axs, 0, 2, M, L, fidelity_1, fidelity_1_std, p_array, 
                                  y_indices[2], error_bars, "lower right", y_labels[2], alpha_trans)
    # plot fidelity during iterations if additional entropy minimzation has been performed
    axs = plot_energies_entropies(axs, 1, 2, M, L, fidelity_2, fidelity_2_std, p_array, 
                                  y_indices[3], error_bars, "lower right", y_labels[2], alpha_trans)

    for i in range(3):
        axs[1,i].set_xlabel(r"number of iteration")

    plt.savefig("images/"+image_name+".png", dpi=200)
    return 

# this function splits up the results of the monte-carlo simulation into comprehensible plots
def plot_monte_carlo_simulation_results_individual(choose_quantity, exp_values_1, exp_values_1_std, exp_values_2, exp_values_2_std, entropy_1, entropy_1_std, entropy_2, entropy_2_std, fidelity_1, fidelity_1_std, fidelity_2, fidelity_2_std, error_bars, p_array, M, L, y_labels, y_indices, opt_approach, image_name):
    fig, axs = plt.subplots(1,2, constrained_layout=True, figsize=(15,6), sharey=True)

    if opt_approach == "max": # plot either maximization or minimization dependend on the type of optimization problem
        h_p_string = r", $\overline{\langle H_p \rangle_{max}} = $"
    elif opt_approach == "min":
        h_p_string = r", $\overline{\langle H_p \rangle_{min}} = $"

    alpha_trans = np.array([0.5,0.4,0.3])
    
    if choose_quantity == "energy":
        # plot energy values if no additional entropy minimization has been performed
        axs = plot_energies_entropies(axs, 0, 10, M, L, exp_values_1, exp_values_1_std, p_array, 
                                      h_p_string, error_bars, "lower right", y_labels[0], alpha_trans)
        # plot energy values if additional entropy minimization has been performed
        axs = plot_energies_entropies(axs, 1, 10, M, L, exp_values_2, exp_values_2_std, p_array, 
                                      h_p_string, error_bars, "lower right", y_labels[0], alpha_trans)
    elif choose_quantity == "entropy":
       # plot entropy values if no additional entropy minimization has been performed
       axs = plot_energies_entropies(axs, 0, 10, M, L, entropy_1, entropy_1_std, p_array, 
                                     y_indices[0], error_bars, "upper right", y_labels[1], alpha_trans)
       # plot entropy values if additional entropy minimization has been performed
       axs = plot_energies_entropies(axs, 1, 10, M, L, entropy_2, entropy_2_std, p_array, 
                                     y_indices[1], error_bars, "upper right", y_labels[1], alpha_trans)
    elif choose_quantity == "fidelity":
        # plot fidelity during iterations if no additional entropy minimzation has been performed
        axs = plot_energies_entropies(axs, 0, 10, M, L, fidelity_1, fidelity_1_std, p_array, 
                                      y_indices[2], error_bars, "lower right", y_labels[2], alpha_trans)
        # plot fidelity during iterations if additional entropy minimzation has been performed
        axs = plot_energies_entropies(axs, 1, 10, M, L, fidelity_2, fidelity_2_std, p_array, 
                                      y_indices[3], error_bars, "lower right", y_labels[2], alpha_trans)

    for i in range(2):
        axs[i].set_xlabel(r"number of iteration")

    plt.savefig("images/"+image_name+".png", dpi=200)
    return 

# this function plots the fock-distribution (amplitudes)
def plot_prob_amplitudes(state_probs_1, state_probs_1_std, state_probs_2, state_probs_2_std, p_array, M, L, rand_approach, opt_approach, image_name, y_max):
    font_size = 10 # label font size
    fig, axs = plt.subplots(2,int(len(p_array)), constrained_layout=True, figsize=(15,7), sharex=True, sharey=True)

    if len(p_array) == 1:
        for i in range(len(p_array)):
            axs[0].bar(np.arange(len(state_probs_1[i])), state_probs_1[i], yerr=state_probs_1_std[i], alpha=0.7, ecolor='black', capsize=3, label=r"$p = $"+str(p_array[i])+", $M = $"+str(M)+", $L = $"+str(L))
            axs[0].legend()
        for i in range(len(p_array)):
            axs[1].bar(np.arange(len(state_probs_2[i])), state_probs_2[i], yerr=state_probs_2_std[i], alpha=0.7, ecolor='black', capsize=3, label=r"$p = $"+str(p_array[i])+", $M = $"+str(M)+", $L = $"+str(L))
            axs[1].set_xlabel(r"qubit$_i$ ($|\gamma,\beta\rangle_{end}$)")
            axs[1].legend()
            
        axs[0].set_ylabel(r"$|\alpha_i|^2 = \alpha^*_i \alpha_i$")
        axs[1].set_ylabel(r"$|\alpha_i|^2 = \alpha^*_i \alpha_i$")

        axs[0].set_ylim(ymin=0, ymax=0.2)
    else:
        for i in range(len(p_array)):
            axs[0,i].bar(np.arange(len(state_probs_1[i])), state_probs_1[i], yerr=state_probs_1_std[i], alpha=0.7, ecolor='black', capsize=3, label=r"$p = $"+str(p_array[i])+", $M = $"+str(M)+", $L = $"+str(L))
            axs[0,i].legend()
        for i in range(len(p_array)):
            axs[1,i].bar(np.arange(len(state_probs_2[i])), state_probs_2[i], yerr=state_probs_2_std[i], alpha=0.7, ecolor='black', capsize=3, label=r"$p = $"+str(p_array[i])+", $M = $"+str(M)+", $L = $"+str(L))
            axs[1,i].set_xlabel(r"qubit$_i$ ($|\gamma,\beta\rangle_{end}$)")
            axs[1,i].legend()
        axs[0,0].set_ylabel(r"$|\alpha_i|^2 = \alpha^*_i \alpha_i$")
        axs[1,0].set_ylabel(r"$|\alpha_i|^2 = \alpha^*_i \alpha_i$")

        axs[0,0].set_ylim(ymin=0, ymax=y_max)

    plt.savefig("images/"+image_name+".png", dpi=200)

# this function plots the fock-distribution (amplitudes) separately
def plot_prob_amplitudes_individual(selection, state_probs_1, state_probs_1_std, state_probs_2, state_probs_2_std, p_array, M, L, rand_approach, opt_approach, image_name, y_max):
    font_size = 10 # label font size
    fig, axs = plt.subplots(1,int(len(p_array)), constrained_layout=True, figsize=(15,6), sharex=True, sharey=True)
    
    if selection == "probs_1":
        for i in range(len(p_array)):
            axs[i].bar(np.arange(len(state_probs_1[i])), state_probs_1[i], yerr=state_probs_1_std[i], alpha=0.7, ecolor='black', capsize=3, label=r"$p = $"+str(p_array[i])+", $M = $"+str(M)+", $L = $"+str(L))
            axs[i].set_xlabel(r"qubit$_i$ ($|\gamma,\beta\rangle_{end}$)")
            axs[i].legend()
    elif selection == "probs_2":
        for i in range(len(p_array)):
            axs[i].bar(np.arange(len(state_probs_2[i])), state_probs_2[i], yerr=state_probs_2_std[i], alpha=0.7, ecolor='black', capsize=3, label=r"$p = $"+str(p_array[i])+", $M = $"+str(M)+", $L = $"+str(L))
            axs[i].set_xlabel(r"qubit$_i$ ($|\gamma,\beta\rangle_{end}$)")
            axs[i].legend()
            
    axs[0].set_ylabel(r"$|\alpha_i|^2 = \alpha^*_i \alpha_i$")
    axs[0].set_ylim(ymin=0, ymax=y_max)

    plt.savefig("images/"+image_name+".png", dpi=200)
    
# helper function to visualize the MaxCut solution
def prepare_bipartite_graph(spin_orientations, J_adj):
    graph = nx.from_numpy_matrix(np.matrix(J_adj))
    node_colors = []
    for spin in spin_orientations:
        if spin == "0":
            node_colors.append('blue')
        elif spin == "1": 
            node_colors.append('green')   
    return graph, node_colors

# main function to visualize the MaxCut solution
def plot_solution_graphs(qubit_number, threshold, probs, J_int, image_size, image_name):
    solution_string = []
    probs_string = []

    for i in range(int(2**qubit_number)):
        if probs[i] >= threshold: # if the threshold (determined from graph is fulfilled 
            solution_string.append(format(i, "0"+str(int(qubit_number))+"b")) # for qubits number > 10 the 0 may be omitted
            probs_string.append(str(round(probs[i],2)))
    
    fig, axs = plt.subplots(int(np.ceil(len(solution_string)/2)),2, constrained_layout=True, figsize=image_size)

    j = 0
    for i in range(len(solution_string)):  
        graph, node_colors = prepare_bipartite_graph(solution_string[i], J_int)
        
        if np.ceil(len(solution_string)/2) != 1:
            nx.draw_circular(graph, node_size=700, node_color=node_colors, with_labels=True, ax=axs[int(np.floor(i/2)),j], label="t")
            axs[int(np.floor(i/2)),j].annotate(r"$|$"+solution_string[i]+r"$\rangle$, $|\alpha_i|^2 = $"+probs_string[i], xy=(0,0), xycoords='axes points',
                                               size=20, bbox=dict(boxstyle='round', fc='w'))
        else:
            nx.draw_circular(graph, node_size=700, node_color=node_colors, with_labels=True, ax=axs[j], label="t")
            axs[j].annotate(r"$|$"+solution_string[i]+r"$\rangle$", xy=(0,0), xycoords='axes points',
                            size=20, bbox=dict(boxstyle='round', fc='w'))
        j += 1
        if j == 2:
            j = 0
    plt.savefig("images/"+image_name+".png", dpi=200)
    return solution_string, probs_string
    # ["|{0}>".format(format(m, '04b')) for m in np.arange(16)] # generates all possible strings for qubit number (quite helpful)