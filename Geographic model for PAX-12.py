# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 18:07:39 2026

@author: Sirius
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, CheckButtons
import random

# =============================================
# EPIDEMIC SIMULATION ENGINE
# =============================================

class EpidemicSim:
    def __init__(self, n_people=900, width=100, height=100):
        self.n = n_people
        self.width, self.height = width, height
        
        # Initial Positions & Smooth Velocity
        self.pos = np.random.uniform(0, [width, height], (n_people, 2))
        self.vel = np.random.normal(0, 0.4, (n_people, 2))
        
        # States: 0=Susceptible, 1=Infected, 2=Dead
        self.states = np.zeros(n_people, dtype=int)
        self.infection_time = np.full(n_people, -1.0)
        
        # Seed infection in the center
        seed_idx = np.argmin(np.sum((self.pos - [width/2, height/2])**2, axis=1))
        self.states[seed_idx] = 1
        self.infection_time[seed_idx] = 0
        
        # Parameters (Optimized for 150-day slow burn)
        self.transmission_radius = 6.0  
        self.infectious_delay = 2.0
        self.time_to_death = 25.0       
        self.distancing = True
        
        self.time = 0.0
        self.dt = 0.1
        self.max_days = 220
        self.history = {'t': [0], 'S': [n_people-1], 'I': [1], 'D': [0]}
        
        self.ripples = []
        
        self.masks = True
        self.isolation = True
        self.hand_washing = True
        self.base_radius = 12.0
        
        self.isolated = np.zeros(self.n, dtype=bool)
        
        self.iso_active = False
        
    def update(self):
        if self.time >= self.max_days: return
        self.time += self.dt
        
        # --- Movement ---
        noise = np.random.normal(0, 0.35, (self.n, 2))
        self.vel += noise * 0.6
        self.vel *= 0.88 # Damping
        
        if self.distancing:
            self.vel *= 0.6
                
        self.vel[self.states == 2] = 0
        
        
        death_fraction = np.sum(self.states == 2) / self.n
        inf_fraction = np.sum(self.states == 1) / self.n
        
        #ISOLATION
        if self.isolation and (death_fraction >= 0.05 or inf_fraction >=0.3) and not self.iso_active:
            self.iso_active = True
        
        if self.iso_active:

            candidates = np.where((self.states == 1) & (~self.isolated))[0]
        
            if len(candidates) > 0:
                p = 0.02 * self.dt   # slow daily rate
                new_iso = candidates[np.random.rand(len(candidates)) < p]
                self.isolated[new_iso] = True
        
        self.pos += self.vel * self.dt * 10
        
        for dim in range(2):
            mask_low = self.pos[:, dim] < 0
            mask_high = self.pos[:, dim] > (self.width if dim==0 else self.height)
            self.vel[mask_low | mask_high, dim] *= -1
            self.pos[:, dim] = np.clip(self.pos[:, dim], 0, self.width if dim==0 else self.height)
        
        # --- Infection Spread ---
        if self.isolation:
            infected_idx = np.where((self.states == 1) & (~self.isolated))[0]
        else:
            infected_idx = np.where(self.states == 1)[0]
        
        prob = 0.001 if self.hand_washing else 0.004
        for i in infected_idx:
            if self.time - self.infection_time[i] < self.infectious_delay: continue
            
            diffs = self.pos - self.pos[i]
            dists = np.linalg.norm(diffs, axis=1)
            susceptibles = np.where((self.states == 0) & (dists < self.transmission_radius))[0]
            for j in susceptibles:
                if random.random() < prob: 
                    self.states[j] = 1
                    self.infection_time[j] = self.time
                    self.ripples.append([self.pos[j,0], self.pos[j,1], 0.0])
        # --- Death Logic ---
        for i in np.where(self.states == 1)[0]:
            if self.time - self.infection_time[i] > (self.infectious_delay + self.time_to_death):
                self.states[i] = 2
                self.vel[i] = 0
                self.isolated[i] = False

        for r in self.ripples:
            r[2] += 0.04 
        self.ripples = [r for r in self.ripples if r[2] < 1.0]

        self.history['t'].append(self.time)
        self.history['S'].append(np.sum(self.states == 0))
        self.history['I'].append(np.sum(self.states == 1))
        self.history['D'].append(np.sum(self.states == 2))
        

    def get_colors(self):
        colors = np.zeros((self.n, 3))
        colors[self.states == 0] = [0.1, 0.5, 0.95] # Blue
        colors[self.states == 1] = [1.0, 0.7, 0.1]  # Orange
        colors[self.states == 2] = [0.9, 0.1, 0.1]  # Red
        return colors
    
    def toggle_masks(self, enabled):
        """
        When masks are ON, the transmission radius is reduced by 50%.
        """
        self.masks = enabled
        if self.masks:
            self.transmission_radius = self.base_radius * 0.5
        else:
            self.transmission_radius = self.base_radius
# ========================== UI & PLOT SETUP ==========================
plt.rcParams['text.color'] = 'white'
fig = plt.figure(figsize=(16, 9), facecolor='#050505')

# Main Simulation Window
ax_sim = fig.add_axes([0.02, 0.15, 0.72, 0.78], facecolor='#080808')
ax_sim.set_xlim(0, 100); ax_sim.set_ylim(0, 100)
ax_sim.set_xticks([]); ax_sim.set_yticks([])
for spine in ax_sim.spines.values(): spine.set_edgecolor('#222')

# Header
fig.text(0.02, 0.96, "PAX-12 SIMULATION", fontsize=14, fontweight='bold')
fig.patches.append(plt.Circle((0.765, 0.36), 0.005, color='#1e90ff', transform=fig.transFigure))
fig.text(0.777, 0.355, "Susceptible", fontsize=10)
fig.patches.append(plt.Circle((0.765, 0.32), 0.005, color='#ffa500', transform=fig.transFigure))
fig.text(0.777, 0.315, "Infected", fontsize=10)
fig.patches.append(plt.Circle((0.765, 0.28), 0.005, color='#ff4500', transform=fig.transFigure))
fig.text(0.777, 0.275, "Dead", fontsize=10)

day_text = fig.text(0.5, 0.96, "DAY 0", fontsize=14, ha='center')

# Scatter and Transmission Lines
sim = EpidemicSim()
scatter = ax_sim.scatter(sim.pos[:,0], sim.pos[:,1], s=18, alpha=0.9, edgecolors='none', c=sim.get_colors(), zorder=3)
ripple_scatter = ax_sim.scatter([], [], s=[], facecolors='none', edgecolors='#ffa500', lw=1.5, zorder=2)
t_lines = [ax_sim.plot([], [], color='#ffa500', alpha=0.08, lw=0.6)[0] for _ in range(150)]

# Statistics Sidebar
ax_stats = fig.add_axes([0.76, 0.72, 0.22, 0.21], facecolor='#0a0a0a')
ax_stats.axis('off')
fig.text(0.77, 0.92, "STATISTICS", fontsize=10, fontweight='bold', color='#aaa')
stat_labels = [fig.text(0.77, y, "", fontsize=10) for y in [0.86, 0.82, 0.78, 0.74]]

# Graph Sidebar
ax_graph = fig.add_axes([0.765, 0.42, 0.22, 0.22], facecolor='#050505')
ax_graph.set_xlim(0, 220); ax_graph.set_ylim(0, 900)
line_s, = ax_graph.plot([], [], color='#1e90ff', lw=1.5)
line_i, = ax_graph.plot([], [], color='#ffa500', lw=1.5)
line_d, = ax_graph.plot([], [], color='#ff4500', lw=1.5)
ax_graph.tick_params(labelsize=7, colors='#555')
fig.text(0.77, 0.65, "POPULATION OVER TIME", fontsize=10, fontweight='bold', color='#aaa')

# Progress Bar
ax_prog = fig.add_axes([0.02, 0.12, 0.72, 0.005], facecolor='#222')
ax_prog.axis('off')
prog_bar, = ax_prog.plot([0, 0], [0.5, 0.5], color='white', lw=4, transform=ax_prog.transAxes)

# Bottom UI (No Sliders)
ax_reset = fig.add_axes([0.02, 0.04, 0.12, 0.05])
btn_reset = Button(ax_reset, 'Reset Simulation', color='#1a1a1a', hovercolor='#333')

#Distancing
dist_check = CheckButtons(fig.add_axes([0.27, 0.04, 0.08, 0.04], frameon=False), ['ON'], [True])
dist_check.labels[0].set_color('#2ecc71')
fig.text(0.21, 0.054, "DISTANCING", fontsize=12)

#Mask
mask_check = CheckButtons(fig.add_axes([0.37, 0.04, 0.08, 0.04], frameon=False), ['ON'], [True])
mask_check.labels[0].set_color('#2ecc71')
fig.text(0.34, 0.054, "MASKS", fontsize=12)

#Isolation
isol_check = CheckButtons(fig.add_axes([0.49, 0.04, 0.08, 0.04], frameon=False), ['ON'], [True])
isol_check.labels[0].set_color('#2ecc71')
fig.text(0.44, 0.054, "ISOLATION", fontsize=12)

#Hand washing
wash_check = CheckButtons(fig.add_axes([0.64, 0.04, 0.08, 0.04], frameon=False), ['ON'], [True])
wash_check.labels[0].set_color('#2ecc71')
fig.text(0.56, 0.054, "HAND WASHING", fontsize=12)

# Isolation Box
ax_iso_zone = fig.add_axes([0.76, 0.03, 0.22, 0.20], facecolor='#050510')
ax_iso_zone.set_xlim(0, 100); ax_iso_zone.set_ylim(0, 100)
ax_iso_zone.set_xticks([]); ax_iso_zone.set_yticks([])
for spine in ax_iso_zone.spines.values(): 
    spine.set_edgecolor('#444')
    spine.set_linewidth(2)

fig.text(0.87, 0.21, "QUARANTINE FACILITY", color='#ff4500', 
         ha='center', fontsize=9, fontweight='bold')

scatter_iso = ax_iso_zone.scatter([], [], s=15, alpha=0.9, edgecolors='none', zorder=3)

# ========================== ANIMATION ==========================
def update(frame):
    for _ in range(2): sim.update()
    
    is_iso = (sim.states == 1) & sim.isolation
    
    main_pos = sim.pos.copy()
    main_pos[is_iso] = np.nan
    
    scatter.set_offsets(main_pos)
    scatter.set_color(sim.get_colors())
    
    iso_pos = np.full((sim.n, 2), np.nan)
    iso_pos[is_iso] = sim.pos[is_iso]
    
    scatter_iso.set_offsets(iso_pos)
    scatter_iso.set_color(sim.get_colors())
    
    is_iso = sim.isolated
    
    main_display_pos = sim.pos.copy()
    main_display_pos[is_iso] = np.nan
    
    scatter.set_offsets(main_display_pos)
    scatter.set_color(sim.get_colors())
    
    iso_display_pos = np.full((sim.n, 2), np.nan)
    iso_display_pos[is_iso] = sim.pos[is_iso]
    
    scatter_iso.set_offsets(iso_display_pos)
    scatter_iso.set_color(sim.get_colors())
    
    if sim.ripples:
        r_data = np.array(sim.ripples)
        ripple_scatter.set_offsets(r_data[:, :2])
        ripple_scatter.set_sizes(r_data[:, 2] * 1200)
        alpha_colors = np.zeros((len(sim.ripples), 4))
        alpha_colors[:, :3] = [1.0, 0.65, 0.1]
        alpha_colors[:, 3] = np.clip(1.0 - r_data[:, 2], 0, 1)
        ripple_scatter.set_edgecolors(alpha_colors)
    else:
        ripple_scatter.set_offsets(np.empty((0, 2)))

    infected = np.where(sim.states == 1)[0]
    line_idx = 0
    for i in infected[:30]: 
        near = np.where((sim.states == 0) & (np.linalg.norm(sim.pos - sim.pos[i], axis=1) < sim.transmission_radius))[0]
        for j in near[:3]:
            if line_idx < len(t_lines):
                t_lines[line_idx].set_data([sim.pos[i,0], sim.pos[j,0]], [sim.pos[i,1], sim.pos[j,1]])
                line_idx += 1
    for k in range(line_idx, len(t_lines)): t_lines[k].set_data([], [])
    
    s, i_cnt, d = np.sum(sim.states==0), np.sum(sim.states==1), np.sum(sim.states==2)
    stat_labels[0].set_text(f"Susceptible:  {s} ({s/900*100:.1f}%)")
    stat_labels[1].set_text(f"Infected:     {i_cnt} ({i_cnt/900*100:.1f}%)")
    stat_labels[2].set_text(f"Dead:         {d} ({d/900*100:.1f}%)")
    stat_labels[3].set_text("Total:        900 (100%)")
    
    day_text.set_text(f"DAY {int(sim.time)}")
    prog_bar.set_data([0, sim.time/sim.max_days], [0.5, 0.5])
    
    line_s.set_data(sim.history['t'], sim.history['S'])
    line_i.set_data(sim.history['t'], sim.history['I'])
    line_d.set_data(sim.history['t'], sim.history['D'])
    
    
    return [scatter, ripple_scatter] + t_lines

def reset(event):
    global sim
    sim = EpidemicSim()
    # Sync simulation with current button states
    sim.distancing = dist_check.labels[0].get_text() == 'ON'
    sim.isolation = isol_check.labels[0].get_text() == 'ON'
    sim.hand_washing = wash_check.labels[0].get_text() == 'ON'
    sim.toggle_masks(mask_check.labels[0].get_text() == 'ON')


btn_reset.on_clicked(reset)


def dist_click(label):
    
    sim.distancing = not sim.distancing
    
    label_obj = dist_check.labels[0]
    new_text = 'OFF' if label_obj.get_text() == 'ON' else 'ON'
    label_obj.set_text(new_text)
    
    
    fig.canvas.draw_idle()


dist_check.on_clicked(dist_click)

def mask_click(label):
    
    sim.toggle_masks(not sim.masks)
    
    
    label_obj = mask_check.labels[0]
    new_text = 'OFF' if label_obj.get_text() == 'ON' else 'ON'
    label_obj.set_text(new_text)
    
    fig.canvas.draw_idle()
mask_check.on_clicked(mask_click)

def isol_click(label):
    
    sim.isolation = not sim.isolation
    
    label_obj = isol_check.labels[0]
    new_text = 'OFF' if label_obj.get_text() == 'ON' else 'ON'
    label_obj.set_text(new_text)
    
    fig.canvas.draw_idle()
isol_check.on_clicked(isol_click)

def wash_click(label):

    sim.hand_washing = not sim.hand_washing
    
    label_obj = wash_check.labels[0]
    new_text = 'OFF' if label_obj.get_text() == 'ON' else 'ON'
    label_obj.set_text(new_text)
    
    fig.canvas.draw_idle()
wash_check.on_clicked(wash_click)

ani = FuncAnimation(fig, update, frames=None, interval=30, blit=False)
plt.show()