import sys
import time
import random
import math 
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import networkx as nx
import heapq

class Graph:
    def __init__(self):
        self.graph = {}

    def add_edge(self, u: str, v: str, w: int):
        if u not in self.graph:
            self.graph[u] = {}
        if v not in self.graph:
            self.graph[v] = {}
        self.graph[u][v] = w
        self.graph[v][u] = w

def dijkstra(graph: Graph, start: str, end: str) -> List[str]:
    if start not in graph.graph or end not in graph.graph:
        return []

    distances = {node: float('infinity') for node in graph.graph}
    distances[start] = 0
    previous = {node: None for node in graph.graph}
    pq = [(0, start)]

    while pq:
        current_distance, current_node = heapq.heappop(pq)

        if current_node == end:
            path = []
            while current_node:
                path.append(current_node)
                current_node = previous[current_node]
            return path[::-1]

        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in graph.graph[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = current_node
                heapq.heappush(pq, (distance, neighbor))

    return []

def basic_delivery_auction(graph: Graph, deliveries: List[Tuple[int, str, int]]) -> Tuple[List[Tuple[int, str, int]], int]:
    selected_deliveries = []
    current_time = 0
    current_location = next(iter(graph.graph))
    total_bonus = 0

    sorted_deliveries = sorted(deliveries, key=lambda x: x[0])

    for start_time, destination, bonus in sorted_deliveries:
        if current_time <= start_time:
            path = dijkstra(graph, current_location, destination)
            if not path:
                continue

            travel_time = sum(graph.graph[path[i]][path[i+1]] for i in range(len(path)-1))
            arrival_time = max(current_time, start_time) + travel_time

            if arrival_time <= start_time + 10:
                selected_deliveries.append((arrival_time, destination, bonus))
                total_bonus += bonus
                current_time = arrival_time
                current_location = destination

                return_path = dijkstra(graph, destination, next(iter(graph.graph)))
                if return_path:
                    return_time = sum(graph.graph[return_path[i]][return_path[i+1]] for i in range(len(return_path)-1))
                    current_time += return_time
                    current_location = next(iter(graph.graph))

    return selected_deliveries, total_bonus

def simulated_annealing(graph: Graph, deliveries: List[Tuple[int, str, int]], initial_temp=1000, cooling_rate=0.995, iterations=1000):
    def calculate_total_bonus(solution):
        total_bonus = 0
        current_time = 0
        current_location = next(iter(graph.graph))

        for start_time, destination, bonus in solution:
            path = dijkstra(graph, current_location, destination)
            if not path:
                continue

            travel_time = sum(graph.graph[path[i]][path[i+1]] for i in range(len(path)-1))
            arrival_time = max(current_time, start_time) + travel_time

            if arrival_time <= start_time + 10:
                total_bonus += bonus
                current_time = arrival_time
                current_location = destination

                return_path = dijkstra(graph, destination, next(iter(graph.graph)))
                if return_path:
                    return_time = sum(graph.graph[return_path[i]][return_path[i+1]] for i in range(len(return_path)-1))
                    current_time += return_time
                    current_location = next(iter(graph.graph))
            else:
                break

        return total_bonus

    def generate_neighbor(solution):
        i, j = random.sample(range(len(solution)), 2)
        new_solution = solution[:]
        new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
        return new_solution

    current_solution = sorted(deliveries, key=lambda x: x[0])
    best_solution = current_solution
    current_bonus = calculate_total_bonus(current_solution)
    best_bonus = current_bonus
    temperature = initial_temp

    for _ in range(iterations):
        neighbor = generate_neighbor(current_solution)
        neighbor_bonus = calculate_total_bonus(neighbor)
        delta = neighbor_bonus - current_bonus

        if delta > 0 or random.random() < math.exp(delta / temperature):
            current_solution = neighbor
            current_bonus = neighbor_bonus

            if current_bonus > best_bonus:
                best_solution = current_solution
                best_bonus = current_bonus

        temperature *= cooling_rate

    return best_solution, best_bonus

class DeliveryAuctionSimulation:
    def __init__(self, master):
        self.master = master
        self.master.title("Simulação de Leilão de Entregas")

        self.graph = nx.Graph()
        self.deliveries = []

        self.create_widgets()
        self.update_graph()

    def create_widgets(self):
        self.canvas = FigureCanvasTkAgg(plt.figure(figsize=(8, 6)), master=self.master)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.add_edge_frame = ttk.Frame(self.master)
        self.add_edge_frame.pack(pady=10)

        ttk.Label(self.add_edge_frame, text="De:").grid(row=0, column=0)
        self.from_node = ttk.Entry(self.add_edge_frame, width=5)
        self.from_node.grid(row=0, column=1)

        ttk.Label(self.add_edge_frame, text="Para:").grid(row=0, column=2)
        self.to_node = ttk.Entry(self.add_edge_frame, width=5)
        self.to_node.grid(row=0, column=3)

        ttk.Label(self.add_edge_frame, text="Peso:").grid(row=0, column=4)
        self.weight = ttk.Entry(self.add_edge_frame, width=5)
        self.weight.grid(row=0, column=5)

        ttk.Button(self.add_edge_frame, text="Adicionar Conexão", command=self.add_edge).grid(row=0, column=6, padx=5)

        self.add_delivery_frame = ttk.Frame(self.master)
        self.add_delivery_frame.pack(pady=10)

        ttk.Label(self.add_delivery_frame, text="Início:").grid(row=0, column=0)
        self.start_time = ttk.Entry(self.add_delivery_frame, width=5)
        self.start_time.grid(row=0, column=1)

        ttk.Label(self.add_delivery_frame, text="Destino:").grid(row=0, column=2)
        self.destination = ttk.Entry(self.add_delivery_frame, width=5)
        self.destination.grid(row=0, column=3)

        ttk.Label(self.add_delivery_frame, text="Bônus:").grid(row=0, column=4)
        self.bonus = ttk.Entry(self.add_delivery_frame, width=5)
        self.bonus.grid(row=0, column=5)

        ttk.Button(self.add_delivery_frame, text="Adicionar Entrega", command=self.add_delivery).grid(row=0, column=6, padx=5)

        ttk.Button(self.master, text="Executar Simulação", command=self.run_simulation).pack(pady=10)

    def add_edge(self):
        try:
            from_node = self.from_node.get()
            to_node = self.to_node.get()
            weight = int(self.weight.get())
            if not from_node or not to_node:
                raise ValueError("Nós de origem e destino são obrigatórios")
            self.graph.add_edge(from_node, to_node, weight=weight)
            self.update_graph()
        except ValueError as e:
            messagebox.showerror("Erro", str(e))

    def add_delivery(self):
        try:
            start_time = int(self.start_time.get())
            destination = self.destination.get().lower()
            bonus = int(self.bonus.get())
            
            if destination not in self.graph.nodes():
                raise ValueError(f"Destino '{destination}' não existe no grafo.")
            
            self.deliveries.append((start_time, destination, bonus))
            self.update_graph()
            messagebox.showinfo("Sucesso", f"Entrega adicionada: Início {start_time}, Destino {destination}, Bônus {bonus}")
        except ValueError as e:
            messagebox.showerror("Erro", f"Entrada inválida: {str(e)}\nCertifique-se de que:\n- Início e Bônus são números inteiros\n- Destino existe no grafo")

    def update_graph(self):
        plt.clf()
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, font_weight='bold')
        edge_labels = nx.get_edge_attributes(self.graph, 'weight')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)
        
        for i, (_, dest, bonus) in enumerate(self.deliveries):
            if dest in pos:
                plt.annotate(f"D{i+1}: {bonus}", xy=pos[dest], xytext=(10, 10), 
                            textcoords="offset points", bbox=dict(boxstyle="round", fc="yellow", ec="none"))
            else:
                print(f"Aviso: Destino '{dest}' não encontrado no grafo.")

        self.canvas.draw()

    def run_simulation(self):
        if not self.deliveries:
            messagebox.showerror("Erro", "Adicione pelo menos uma entrega antes de executar a simulação.")
            return

        graph_obj = Graph()
        for u, v, data in self.graph.edges(data=True):
            graph_obj.add_edge(u, v, data['weight'])

        basic_results, basic_bonus = basic_delivery_auction(graph_obj, self.deliveries)
        
        if len(self.deliveries) < 2:
            ai_results, ai_bonus = self.deliveries, sum(bonus for _, _, bonus in self.deliveries)
        else:
            ai_results, ai_bonus = simulated_annealing(graph_obj, self.deliveries)

        result_window = tk.Toplevel(self.master)
        result_window.title("Resultados da Simulação")

        ttk.Label(result_window, text="Algoritmo Básico:").pack()
        for arrival_time, destination, bonus in basic_results:
            ttk.Label(result_window, text=f"({arrival_time}, {destination}; {bonus})").pack()
        ttk.Label(result_window, text=f"Lucro total: {basic_bonus}").pack()

        ttk.Label(result_window, text="\nAlgoritmo com IA:").pack()
        for arrival_time, destination, bonus in ai_results:
            ttk.Label(result_window, text=f"({arrival_time}, {destination}; {bonus})").pack()
        ttk.Label(result_window, text=f"Lucro total: {ai_bonus}").pack()

if __name__ == "__main__":
    root = tk.Tk()
    app = DeliveryAuctionSimulation(root)
    root.mainloop()
