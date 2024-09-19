import sys
import time
import random
import math  # Add this line to import the math module
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import networkx as nx

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

def read_input(filename: str) -> Tuple[Graph, List[Tuple[int, str, int]]]:
    with open(filename, 'r') as f:
        n_connections = int(f.readline().strip().split()[0])
        destinations = f.readline().strip().split(', ')
        
        graph = Graph()
        for i in range(n_connections):
            row = list(map(int, f.readline().strip().split(', ')))
            for j in range(n_connections):
                if row[j] != 0:
                    graph.add_edge(destinations[i], destinations[j], row[j])
        
        n_deliveries = int(f.readline().strip().split()[0])
        deliveries = []
        for _ in range(n_deliveries):
            start, dest, bonus = f.readline().strip().split(', ')
            deliveries.append((int(start), dest, int(bonus)))
    
    return graph, deliveries

# Versão 1: Algoritmo Básico
def basic_delivery_auction(graph: Graph, deliveries: List[Tuple[int, str, int]]) -> List[Tuple[int, str, int]]:
    selected_deliveries = []
    current_time = 0
    current_location = next(iter(graph.graph))  # Pega o primeiro nó do grafo
    total_bonus = 0

    for start_time, destination, bonus in deliveries:
        if current_time <= start_time:
            if current_location not in graph.graph or destination not in graph.graph[current_location]:
                print(f"Aviso: Não há conexão direta entre {current_location} e {destination}")
                continue
            travel_time = graph.graph[current_location][destination]
            arrival_time = max(current_time + travel_time, start_time + travel_time)
            selected_deliveries.append((arrival_time, destination, bonus))
            total_bonus += bonus
            current_time = arrival_time
            current_location = destination

    return selected_deliveries, total_bonus

# Versão 2: Algoritmo com IA (Simulated Annealing)
def simulated_annealing(graph: Graph, deliveries: List[Tuple[int, str, int]], initial_temp=1000, cooling_rate=0.995, iterations=1000):
    def calculate_total_bonus(solution):
        return sum(bonus for _, _, bonus in solution)

    def generate_neighbor(solution):
        i, j = random.sample(range(len(solution)), 2)
        new_solution = solution[:]
        new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
        return new_solution

    current_solution = deliveries[:]
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

# Função principal
def main(input_file: str):
    graph, deliveries = read_input(input_file)
    
    # Versão 1: Algoritmo Básico
    start_time = time.time()
    basic_results, basic_bonus = basic_delivery_auction(graph, deliveries)
    basic_time = time.time() - start_time

    # Versão 2: Algoritmo com IA
    start_time = time.time()
    ai_results, ai_bonus = simulated_annealing(graph, deliveries)
    ai_time = time.time() - start_time

    print("Resultados do Algoritmo Básico:")
    for arrival_time, destination, bonus in basic_results:
        print(f"({arrival_time}, {destination}; {bonus})")
    print(f"Lucro total: {basic_bonus}")
    print(f"Tempo de execução: {basic_time:.4f} segundos\n")

    print("Resultados do Algoritmo com IA:")
    for arrival_time, destination, bonus in ai_results:
        print(f"({arrival_time}, {destination}; {bonus})")
    print(f"Lucro total: {ai_bonus}")
    print(f"Tempo de execução: {ai_time:.4f} segundos")

    # Comparação de desempenho
    plt.figure(figsize=(10, 5))
    plt.bar(['Básico', 'IA'], [basic_bonus, ai_bonus])
    plt.title('Comparação de Lucro')
    plt.ylabel('Lucro Total')
    plt.savefig('lucro_comparacao.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(['Básico', 'IA'], [basic_time, ai_time])
    plt.title('Comparação de Tempo de Execução')
    plt.ylabel('Tempo (segundos)')
    plt.savefig('tempo_comparacao.png')
    plt.close()

# Simulação gráfica interativa
class DeliveryAuctionSimulation:
    def __init__(self, master):
        self.master = master
        self.master.title("Simulação de Leilão de Entregas")

        self.graph = nx.Graph()
        self.graph.add_nodes_from(['s', 'd', 'a', 'e', 'f'])
        self.graph.add_edge('s', 'd', weight=1)
        self.graph.add_edge('s', 'a', weight=1)
        self.graph.add_edge('d', 'a', weight=1)
        self.graph.add_edge('d', 'e', weight=1)
        self.graph.add_edge('d', 'f', weight=1)
        self.graph.add_edge('e', 'f', weight=1)
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
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        root = tk.Tk()
        app = DeliveryAuctionSimulation(root)
        root.mainloop()